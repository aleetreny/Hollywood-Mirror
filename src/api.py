from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.search_engine import (
    available_models,
    loaded_models,
    model_dimensions,
    preload_configured_models,
    search_similar_movies,
    warmup_model,
)
from src.settings import DEFAULT_MODEL, ModelId, SUPPORTED_MODELS, get_settings

SETTINGS = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    preload_configured_models()
    yield


app = FastAPI(
    title="Hollywood Mirror API",
    version="0.2.0",
    lifespan=lifespan,
)

# Pydantic Schemas
class SimilarMoviesRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User-provided script fragment or idea.")
    model: ModelId = Field(
        DEFAULT_MODEL,
        description="Embedding model used for the backend matrix.",
    )
    k: int = Field(5, ge=1, le=50, description="Number of similar movies to return.")

class SimilarMovie(BaseModel):
    title: str
    affinity: float

class SimilarMoviesResponse(BaseModel):
    results: list[SimilarMovie]


class ModelCapability(BaseModel):
    id: ModelId
    dimension: int
    available: bool


class SearchCapabilitiesResponse(BaseModel):
    default_model: ModelId
    max_k: int
    models: list[ModelCapability]


class WarmupRequest(BaseModel):
    model: ModelId = Field(
        SETTINGS.warmup_model,
        description="Model to warm up in memory.",
    )


class WarmupResponse(BaseModel):
    status: str
    model: ModelId
    loaded_now: bool
    loaded_models: list[ModelId]
    elapsed_ms: int


# Allow CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    enabled_models = set(available_models())
    return {
        "status": "ok",
        "default_model": SETTINGS.default_model,
        "available_models": sorted(enabled_models),
        "loaded_models": sorted(loaded_models()),
        "processed_dir": str(SETTINGS.processed_dir),
    }


@app.get("/api/capabilities", response_model=SearchCapabilitiesResponse)
def search_capabilities() -> SearchCapabilitiesResponse:
    enabled_models = set(available_models())
    dimensions = model_dimensions()
    return SearchCapabilitiesResponse(
        default_model=SETTINGS.default_model,
        max_k=SETTINGS.max_results,
        models=[
            ModelCapability(
                id=model_id,
                dimension=dimensions[model_id],
                available=model_id in enabled_models,
            )
            for model_id in SUPPORTED_MODELS
        ],
    )


@app.post("/api/similar-movies", response_model=SimilarMoviesResponse)
def similar_movies(payload: SimilarMoviesRequest) -> SimilarMoviesResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    if payload.model not in available_models():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Embeddings for model '{payload.model}' were not found in "
                f"{SETTINGS.processed_dir}. Generate them and redeploy."
            ),
        )

    try:
        results = search_similar_movies(
            text=text,
            model_id=payload.model,
            limit=min(payload.k, SETTINGS.max_results),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Backend failed to process model '{payload.model}': {exc}",
        ) from None

    return SimilarMoviesResponse(results=[SimilarMovie(**result) for result in results])


@app.post("/api/warmup", response_model=WarmupResponse)
def warmup(payload: WarmupRequest) -> WarmupResponse:
    if payload.model not in available_models():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Embeddings for model '{payload.model}' were not found in "
                f"{SETTINGS.processed_dir}. Generate them and redeploy."
            ),
        )

    started_at = perf_counter()
    try:
        loaded_now = warmup_model(payload.model)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Backend failed to warm up model '{payload.model}': {exc}",
        ) from None

    elapsed_ms = int((perf_counter() - started_at) * 1000)
    return WarmupResponse(
        status="ok",
        model=payload.model,
        loaded_now=loaded_now,
        loaded_models=sorted(loaded_models()),
        elapsed_ms=elapsed_ms,
    )
