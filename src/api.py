from __future__ import annotations

import os
import threading
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# App configuration
app = FastAPI(title="Hollywood Mirror API", version="0.1.0")

# Pydantic Schemas
class SimilarMoviesRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User-provided script fragment or idea.")
    model: Literal["mpnet", "minilm"] = Field(
        "minilm",
        description="Embedding model used for the backend matrix.",
    )
    k: int = Field(5, ge=1, le=50, description="Number of similar movies to return.")

class SimilarMovie(BaseModel):
    title: str
    affinity: float

class SimilarMoviesResponse(BaseModel):
    results: list[SimilarMovie]

def _allowed_origins() -> list[str]:
    """
    Resolve CORS origins from env var `API_CORS_ORIGINS` (comma-separated)
    or fall back to local frontend defaults.
    """
    raw = os.getenv("API_CORS_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

# Allow CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionaries to store our loaded artifacts
MATRICES: dict[str, np.ndarray] = {}
TITLES_MAP: dict[str, list[str]] = {}
MODELS: dict[str, Any] = {}
MODEL_NAMES = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
MODEL_LOAD_LOCKS = {model_id: threading.Lock() for model_id in MODEL_NAMES}


def _ensure_resources_loaded(model_id: Literal["mpnet", "minilm"]) -> None:
    """
    Lazily load embeddings and model artifacts for the requested model id.
    """
    # Defer heavyweight imports so the web process can bind a port quickly on Render.
    from sentence_transformers import SentenceTransformer
    from src.embeddings import load_embeddings

    if model_id in MATRICES and model_id in MODELS and model_id in TITLES_MAP:
        return

    with MODEL_LOAD_LOCKS[model_id]:
        # Double check after acquiring the lock.
        if model_id in MATRICES and model_id in MODELS and model_id in TITLES_MAP:
            return

        matrix, titles = load_embeddings(model_id=model_id)
        MATRICES[model_id] = matrix.astype(np.float32)
        TITLES_MAP[model_id] = titles
        MODELS[model_id] = SentenceTransformer(MODEL_NAMES[model_id])

@app.on_event("startup")
def _load_resources() -> None:
    """
    Optionally preload models declared in API_PRELOAD_MODELS.
    Example: API_PRELOAD_MODELS=minilm,mpnet
    """
    raw = os.getenv("API_PRELOAD_MODELS", "").strip()
    preload_models = [model_id.strip() for model_id in raw.split(",") if model_id.strip()]

    for model_id in preload_models:
        if model_id not in MODEL_NAMES:
            print(f"Warning: unknown model in API_PRELOAD_MODELS: '{model_id}'")
            continue
        try:
            print(f"Preloading resources for {model_id}...")
            _ensure_resources_loaded(model_id)
        except FileNotFoundError:
            print(
                f"Warning: embeddings for {model_id} were not found in data/processed/. "
                "Requests for this model will fail until files are available."
            )
        except Exception as exc:
            print(f"Warning: failed preloading {model_id}: {exc}")


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "loaded_models": sorted(MODELS.keys()),
    }

@app.post("/api/similar-movies", response_model=SimilarMoviesResponse)
def similar_movies(payload: SimilarMoviesRequest) -> SimilarMoviesResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    req_model = payload.model
    try:
        _ensure_resources_loaded(req_model)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Embeddings for model '{req_model}' were not found. "
                f"Generate them with `python -m src.embeddings {req_model}` and redeploy."
            ),
        ) from None
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Backend failed to load model '{req_model}': {exc}",
        ) from None

    # Route request to correct matrix and transformer model
    matrix = MATRICES[req_model]
    model = MODELS[req_model]
    titles = TITLES_MAP[req_model]

    # Vectorize payload
    vec = model.encode([payload.text], convert_to_numpy=True).astype(np.float32).ravel()

    # Cosine similarity without sklearn dependency at request time.
    denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)) + 1e-8
    sims = (matrix @ vec) / denom
    order = np.argsort(sims)[::-1][: payload.k]

    def clean_title(raw_title: str) -> str:
        # e.g., "Gateway IMDb_012345" -> "Gateway IMDb", or "Alien_123" -> "Alien"
        if "_" in raw_title:
            raw_title = raw_title.rsplit("_", 1)[0]
        # Sometimes scraped data literally includes " IMDb"
        if raw_title.endswith(" IMDb"):
            raw_title = raw_title[:-5]
        return raw_title

    results = [
        SimilarMovie(title=clean_title(titles[i]), affinity=float(sims[i])) for i in order
    ]
    return SimilarMoviesResponse(results=results)
