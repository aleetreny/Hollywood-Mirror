from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from src.embeddings import MODEL_CONFIG, has_embeddings, load_embeddings
from src.settings import ModelId, SUPPORTED_MODELS, get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(slots=True)
class LoadedResources:
    encoder: SentenceTransformer
    matrix: np.ndarray
    titles: tuple[str, ...]


def clean_title(raw_title: str) -> str:
    if "_" in raw_title:
        raw_title = raw_title.rsplit("_", 1)[0]
    if raw_title.endswith(" IMDb"):
        raw_title = raw_title[:-5]
    return raw_title


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    return normalized / np.clip(norms, 1e-8, None)


class QueryVectorCache:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._items: OrderedDict[tuple[ModelId, str], np.ndarray] = OrderedDict()
        self._lock = Lock()

    def get(self, model_id: ModelId, text: str) -> np.ndarray | None:
        key = (model_id, text)
        with self._lock:
            cached = self._items.get(key)
            if cached is None:
                return None
            self._items.move_to_end(key)
            return cached.copy()

    def put(self, model_id: ModelId, text: str, vector: np.ndarray) -> None:
        if self._max_size <= 0:
            return
        key = (model_id, text)
        with self._lock:
            self._items[key] = vector.copy()
            self._items.move_to_end(key)
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)


_SETTINGS = get_settings()
_MODEL_NAMES = {model_id: MODEL_CONFIG[model_id][0] for model_id in SUPPORTED_MODELS}
_MODEL_DIMENSIONS = {"minilm": 384, "mpnet": 768}
_MODEL_LOCKS = {model_id: Lock() for model_id in SUPPORTED_MODELS}
_RESOURCES: dict[ModelId, LoadedResources] = {}
_QUERY_CACHE = QueryVectorCache(max_size=_SETTINGS.query_cache_size)


@lru_cache(maxsize=1)
def available_models() -> tuple[ModelId, ...]:
    models: list[ModelId] = []
    for model_id in SUPPORTED_MODELS:
        if not has_embeddings(
            processed_dir=_SETTINGS.processed_dir,
            model_id=model_id,
        ):
            continue
        models.append(model_id)
    return tuple(models)


def model_dimensions() -> dict[ModelId, int]:
    return _MODEL_DIMENSIONS.copy()


def loaded_models() -> tuple[ModelId, ...]:
    return tuple(model_id for model_id in SUPPORTED_MODELS if model_id in _RESOURCES)


def _load_encoder(model_id: ModelId) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAMES[model_id])


def get_resources(model_id: ModelId) -> LoadedResources:
    cached = _RESOURCES.get(model_id)
    if cached is not None:
        return cached

    with _MODEL_LOCKS[model_id]:
        cached = _RESOURCES.get(model_id)
        if cached is not None:
            return cached

        matrix, titles = load_embeddings(processed_dir=_SETTINGS.processed_dir, model_id=model_id)
        resources = LoadedResources(
            encoder=_load_encoder(model_id),
            matrix=_normalize_matrix(matrix),
            titles=tuple(titles),
        )
        _RESOURCES[model_id] = resources
        return resources


def preload_configured_models() -> None:
    for model_id in _SETTINGS.preload_models:
        try:
            print(f"Preloading resources for {model_id}...")
            get_resources(model_id)
        except FileNotFoundError:
            print(
                f"Warning: embeddings for {model_id} were not found in "
                f"{_SETTINGS.processed_dir}. Requests for this model will fail "
                "until files are available."
            )
        except Exception as exc:
            print(f"Warning: failed preloading {model_id}: {exc}")


def warmup_model(model_id: ModelId) -> bool:
    already_loaded = model_id in _RESOURCES
    get_resources(model_id)
    return not already_loaded


def _normalize_query_text(text: str) -> str:
    return " ".join(text.split())


def encode_query(text: str, model_id: ModelId) -> np.ndarray:
    normalized_text = _normalize_query_text(text)
    cached = _QUERY_CACHE.get(model_id, normalized_text)
    if cached is not None:
        return cached

    resources = get_resources(model_id)
    vector = resources.encoder.encode(
        [normalized_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    encoded = np.asarray(vector[0], dtype=np.float32)
    _QUERY_CACHE.put(model_id, normalized_text, encoded)
    return encoded


def search_similar_movies(text: str, model_id: ModelId, limit: int) -> list[dict[str, float | str]]:
    resources = get_resources(model_id)
    query = encode_query(text, model_id=model_id)
    scores = resources.matrix @ query

    top_k = min(limit, scores.shape[0])
    if top_k <= 0:
        return []

    candidate_indexes = np.argpartition(scores, -top_k)[-top_k:]
    ordered_indexes = candidate_indexes[np.argsort(scores[candidate_indexes])[::-1]]

    return [
        {
            "title": clean_title(resources.titles[index]),
            "affinity": float(scores[index]),
        }
        for index in ordered_indexes
    ]
