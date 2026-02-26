from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings import load_embeddings

# Pydantic Schemas
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

# App configuration
app = FastAPI(title="Hollywood Mirror API", version="0.1.0")

# Allow CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionaries to store our loaded artifacts
MATRICES = {}
TITLES_MAP = {}
MODELS = {}

@app.on_event("startup")
def _load_resources() -> None:
    """
    Load embedding matrices, titles, and Sentence Transformers models for both supported model types.
    """
    for model_id, model_name in [("mpnet", "sentence-transformers/all-mpnet-base-v2"), 
                                 ("minilm", "sentence-transformers/all-MiniLM-L6-v2")]:
        try:
            print(f"Loading matrix and titles for {model_id}...")
            matrix, titles = load_embeddings(model_id=model_id)
            MATRICES[model_id] = matrix.astype(np.float32)
            TITLES_MAP[model_id] = titles
            
            print(f"Loading SentenceTransformer '{model_name}' into RAM...")
            MODELS[model_id] = SentenceTransformer(model_name)
            
        except FileNotFoundError:
            print(f"Warning: Embeddings for {model_id} not found in data/processed/. Search for this model will fail.")

@app.post("/api/similar-movies", response_model=SimilarMoviesResponse)
def similar_movies(payload: SimilarMoviesRequest) -> SimilarMoviesResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    req_model = payload.model
    if req_model not in MATRICES or req_model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"The requested model '{req_model}' is not currently loaded in the backend. "
            f"Please run `python -m src.embeddings {req_model}` first and restart the server.",
        )

    # Route request to correct matrix and transformer model
    matrix = MATRICES[req_model]
    model = MODELS[req_model]
    titles = TITLES_MAP[req_model]

    # Vectorize payload
    vec = model.encode([payload.text], convert_to_numpy=True).astype(np.float32)

    # Cosine Similarity
    sims = cosine_similarity(matrix, vec).ravel()
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

