from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings import load_embeddings


class SimilarMoviesRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User-provided script fragment or idea.")
    model: Literal["mpnet", "minilm"] = Field(
        "mpnet",
        description="Embedding model used for the backend matrix.",
    )
    k: int = Field(5, ge=1, le=50, description="Number of similar movies to return.")


class SimilarMovie(BaseModel):
    title: str
    affinity: float


class SimilarMoviesResponse(BaseModel):
    results: list[SimilarMovie]


app = FastAPI(title="Hollywood Mirror API", version="0.1.0")


@app.on_event("startup")
def _load_resources() -> None:
    """
    Load embeddings matrix and titles once at startup.
    """
    global MATRIX, TITLES, BACKEND_MODEL_ID, MODEL

    MATRIX, TITLES = load_embeddings()
    MATRIX = MATRIX.astype(np.float32)

    # Infer backend model dim to choose default model id
    dim = MATRIX.shape[1]
    if dim == 768:
        BACKEND_MODEL_ID = "mpnet"
        model_name = "sentence-transformers/all-mpnet-base-v2"
    elif dim == 384:
        BACKEND_MODEL_ID = "minilm"
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        raise RuntimeError(f"Unexpected embedding dimension {dim}")

    MODEL = SentenceTransformer(model_name)


@app.post("/api/similar-movies", response_model=SimilarMoviesResponse)
def similar_movies(payload: SimilarMoviesRequest) -> SimilarMoviesResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    if payload.model != BACKEND_MODEL_ID:
        # For now we only support the model used to create MATRIX.
        raise HTTPException(
            status_code=400,
            detail=f"Backend embeddings were built with '{BACKEND_MODEL_ID}', "
            f"but request asked for '{payload.model}'. Regenerate embeddings or adjust frontend.",
        )

    vec = MODEL.encode([payload.text], convert_to_numpy=True).astype(np.float32)
    sims = cosine_similarity(MATRIX, vec).ravel()
    order = np.argsort(sims)[::-1][: payload.k]

    results = [
        SimilarMovie(title=TITLES[i], affinity=float(sims[i])) for i in order
    ]
    return SimilarMoviesResponse(results=results)

