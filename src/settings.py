from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

ModelId = Literal["mpnet", "minilm"]
SUPPORTED_MODELS: tuple[ModelId, ...] = ("mpnet", "minilm")
DEFAULT_MODEL: ModelId = "minilm"
DEFAULT_MAX_RESULTS = 50
DEFAULT_WARMUP_MODEL: ModelId = DEFAULT_MODEL


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _resolve_cors_origins() -> tuple[str, ...]:
    raw = os.getenv("API_CORS_ORIGINS", "").strip()
    if raw:
        return _split_csv(raw)
    return ("*",)


def _resolve_preload_models() -> tuple[ModelId, ...]:
    raw = os.getenv("API_PRELOAD_MODELS", "").strip()
    if not raw:
        return ()
    models: list[ModelId] = []
    for candidate in _split_csv(raw):
        if candidate in SUPPORTED_MODELS:
            models.append(cast(ModelId, candidate))
        else:
            print(f"Warning: unknown model in API_PRELOAD_MODELS: '{candidate}'")
    return tuple(models)


def _resolve_processed_dir() -> Path:
    configured = os.getenv("DATA_PROCESSED_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "data" / "processed").resolve()


def _resolve_query_cache_size() -> int:
    raw = os.getenv("API_QUERY_CACHE_SIZE", "").strip()
    if not raw:
        return 128
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"Warning: invalid API_QUERY_CACHE_SIZE '{raw}', falling back to 128")
        return 128


@dataclass(frozen=True, slots=True)
class Settings:
    cors_origins: tuple[str, ...]
    preload_models: tuple[ModelId, ...]
    processed_dir: Path
    query_cache_size: int
    default_model: ModelId = DEFAULT_MODEL
    warmup_model: ModelId = DEFAULT_WARMUP_MODEL
    max_results: int = DEFAULT_MAX_RESULTS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        cors_origins=_resolve_cors_origins(),
        preload_models=_resolve_preload_models(),
        processed_dir=_resolve_processed_dir(),
        query_cache_size=_resolve_query_cache_size(),
    )
