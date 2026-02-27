"""
Word-based chunking (respecting model limits), Sentence Transformers vectorization,
and mean pooling per movie. Reusable from Quarto and the web app.
Output: matrix [N, dim] stored in .npy plus titles in .txt (same order).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Available models: (huggingface_name, chunk_size_words)
# - minilm: 256 tokens max, 384 dims, faster. Chunks ~200 words.
# - mpnet:  384 tokens max, 768 dims, higher quality. Chunks ~300 words, works on CPU.
MODEL_CONFIG = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 200),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", 300),
}
DEFAULT_MODEL_ID = "mpnet"
OVERLAP_FRAC = 0.10  # 10% overlap between chunks


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap_frac: float = OVERLAP_FRAC,
) -> list[str]:
    """
    Split text into ~chunk_size word chunks with overlap overlap_frac
    (for example, 0.1 = 10%). chunk_size must fit model limits.
    """
    if not text or not str(text).strip():
        return []
    words = str(text).split()
    if len(words) <= chunk_size:
        return [" ".join(words)] if words else []
    step = max(1, int(chunk_size * (1 - overlap_frac)))
    chunks = []
    for start in range(0, len(words), step):
        block = words[start : start + chunk_size]
        if not block:
            break
        chunks.append(" ".join(block))
        if start + chunk_size >= len(words):
            break
    return chunks


def compute_movie_embedding(
    text: str,
    model: SentenceTransformer,
    chunk_size: int = 300,
    overlap_frac: float = OVERLAP_FRAC,
) -> np.ndarray:
    """
    Chunk, encode, and mean-pool a single movie text.
    Output embedding dimension is model-dependent (384 or 768).
    """
    chunks = chunk_text(text, chunk_size=chunk_size, overlap_frac=overlap_frac)
    if not chunks:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim, dtype=np.float32)
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return np.mean(embeddings, axis=0).astype(np.float32)


def build_embedding_matrix(
    df: pd.DataFrame,
    model: Optional[SentenceTransformer] = None,
    chunk_size: int = 300,
    text_column: str = "cleaned_text",
    title_column: str = "movie_title",
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[np.ndarray, list[str]]:
    """
    For each DataFrame row, perform chunking + encode + mean pooling.
    Returns (matrix, titles) where matrix has shape (N, dim) and titles share row order.
    """
    if model is None:
        name, cs = MODEL_CONFIG.get(model_id, MODEL_CONFIG[DEFAULT_MODEL_ID])
        model = SentenceTransformer(name)
        chunk_size = cs
    titles = []
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding movies"):
        text = row[text_column]
        title = row[title_column]
        vec = compute_movie_embedding(text, model, chunk_size=chunk_size)
        titles.append(str(title))
        rows.append(vec)
    matrix = np.stack(rows, axis=0)
    return matrix, titles


def run(
    processed_dir: Optional[Path] = None,
    input_name: str = "movies_cleaned",
    output_name: Optional[str] = None,
    force: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[np.ndarray, list[str]]:
    """
    Read cleaned DataFrame (parquet or csv), compute embedding matrix,
    and save matrix.npy + titles.txt. If files exist and force=False, load them.

    - model_id: "mpnet" (default, 768 dims, 300-word chunks) or "minilm" (384 dims, 200-word chunks).
    - output_name:
        - If None (default), use "movie_embeddings_{model_id}".
        - If provided, use the explicit prefix.

    Returns (matrix, titles).
    """
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = processed_dir or repo_root / "data" / "processed"
    base_in = processed_dir / input_name

    # Model-specific default output name.
    output_name = output_name or f"movie_embeddings_{model_id}"
    base_out = processed_dir / output_name
    path_npy = base_out.with_suffix(".npy")
    path_titles = base_out.with_suffix(".txt")

    if not force and path_npy.exists() and path_titles.exists():
        matrix = np.load(path_npy, allow_pickle=False)
        titles = path_titles.read_text(encoding="utf-8").strip().split("\n")
        if len(titles) == matrix.shape[0]:
            return matrix, titles

    if base_in.with_suffix(".parquet").exists():
        df = pd.read_parquet(base_in.with_suffix(".parquet"))
    elif base_in.with_suffix(".csv").exists():
        df = pd.read_csv(base_in.with_suffix(".csv"), encoding="utf-8")
    else:
        raise FileNotFoundError(f"Could not find {base_in}.parquet or {base_in}.csv")

    model_name, chunk_size = MODEL_CONFIG.get(model_id, MODEL_CONFIG[DEFAULT_MODEL_ID])
    model = SentenceTransformer(model_name)
    matrix, titles = build_embedding_matrix(df, model=model, chunk_size=chunk_size)

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(path_npy, matrix)
    path_titles.write_text("\n".join(titles), encoding="utf-8")

    return matrix, titles


def load_embeddings(
    processed_dir: Optional[Path] = None,
    model_id: str = DEFAULT_MODEL_ID,
    output_name: Optional[str] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Load persisted matrix and titles (for Quarto or API/web usage).

    - model_id: "mpnet" or "minilm". Default matches generation defaults.
    - output_name:
        - If None (default), look for "movie_embeddings_{model_id}".
        - For backward compatibility, fall back to "movie_embeddings".
    """
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = processed_dir or repo_root / "data" / "processed"

    # Preferred base name.
    base_name = output_name or f"movie_embeddings_{model_id}"
    base = processed_dir / base_name
    path_npy = base.with_suffix(".npy")
    path_titles = base.with_suffix(".txt")

    # Backward compatibility with legacy base name (without model suffix).
    if not path_npy.exists() or not path_titles.exists():
        legacy_base = processed_dir / "movie_embeddings"
        legacy_npy = legacy_base.with_suffix(".npy")
        legacy_txt = legacy_base.with_suffix(".txt")
        if legacy_npy.exists() and legacy_txt.exists():
            path_npy = legacy_npy
            path_titles = legacy_txt
        else:
            raise FileNotFoundError(
                f"Run first: python -m src.embeddings "
                f"(could not find {base}.* or movie_embeddings.*)"
            )
    matrix = np.load(path_npy, allow_pickle=False)
    titles = path_titles.read_text(encoding="utf-8").strip().split("\n")
    return matrix, titles


if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
    if model_id not in MODEL_CONFIG:
        print("Usage: python -m src.embeddings [mpnet|minilm]  (default: mpnet)")
        sys.exit(1)
    matrix, titles = run(model_id=model_id)
    print(
        f"Done: matrix {matrix.shape} ({model_id}) and {len(titles)} titles -> "
        f"data/processed/movie_embeddings_{model_id}.npy/.txt"
    )
