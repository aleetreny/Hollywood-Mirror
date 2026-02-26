"""
Chunking por palabras (respetando límite del modelo), vectorización con Sentence Transformers
y mean pooling por película. Función reutilizable por Quarto y por la app Streamlit.
Output: matriz [N, dim] guardada en .npy + títulos en .txt (mismo orden).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Modelos disponibles: (nombre_huggingface, chunk_size_en_palabras)
# - minilm: 256 tokens max, 384 dims, rápido. Chunks ~200 palabras.
# - mpnet:  384 tokens max, 768 dims, mejor calidad. Chunks ~300 palabras. Corre bien en CPU.
MODEL_CONFIG = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 200),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", 300),
}
DEFAULT_MODEL_ID = "mpnet"
OVERLAP_FRAC = 0.10  # 10% overlap entre chunks


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap_frac: float = OVERLAP_FRAC,
) -> list[str]:
    """
    Divide un texto en bloques de ~chunk_size palabras con solapamiento overlap_frac
    (ej. 0.1 = 10%). chunk_size debe caber en el modelo (minilm: 200, mpnet: 300).
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
    Chunking + encode + mean pooling para un solo texto (una película).
    La dimensión del vector la da el modelo (384 o 768).
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
    Para cada fila del DataFrame: chunking + encode + mean pooling.
    Devuelve (matrix, titles) con matrix de shape (N, dim) y titles en el mismo orden.
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
    Lee el DataFrame limpio (parquet o csv), calcula la matriz de embeddings
    y guarda matrix.npy + titles en .txt. Si los archivos ya existen y force=False, los carga.

    - model_id: "mpnet" (por defecto, 768 dims, chunks 300 palabras) o "minilm" (384 dims, 200 palabras).
    - output_name:
        - Si es None (por defecto), se usa "movie_embeddings_{model_id}".
        - Si se pasa un nombre explícito, se usa ese prefijo.

    Devuelve (matrix, titles).
    """
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = processed_dir or repo_root / "data" / "processed"
    base_in = processed_dir / input_name

    # Nombre de salida dependiente del modelo si no se especifica
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
        raise FileNotFoundError(f"No se encontró {base_in}.parquet ni {base_in}.csv")

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
    Carga la matriz y los títulos guardados (para uso en Quarto o en la API/web).

    - model_id: "mpnet" o "minilm". Por defecto, el mismo que se usa al generar embeddings.
    - output_name:
        - Si es None (por defecto), se busca "movie_embeddings_{model_id}".
        - Para compatibilidad hacia atrás, si no existe se intenta "movie_embeddings".
    """
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = processed_dir or repo_root / "data" / "processed"

    # Nombre base preferido
    base_name = output_name or f"movie_embeddings_{model_id}"
    base = processed_dir / base_name
    path_npy = base.with_suffix(".npy")
    path_titles = base.with_suffix(".txt")

    # Compatibilidad con nombres antiguos sin sufijo de modelo
    if not path_npy.exists() or not path_titles.exists():
        legacy_base = processed_dir / "movie_embeddings"
        legacy_npy = legacy_base.with_suffix(".npy")
        legacy_txt = legacy_base.with_suffix(".txt")
        if legacy_npy.exists() and legacy_txt.exists():
            path_npy = legacy_npy
            path_titles = legacy_txt
        else:
            raise FileNotFoundError(
                f"Ejecuta antes: python -m src.embeddings "
                f"(no se encontró {base}.* ni movie_embeddings.*)"
            )
    matrix = np.load(path_npy, allow_pickle=False)
    titles = path_titles.read_text(encoding="utf-8").strip().split("\n")
    return matrix, titles


if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
    if model_id not in MODEL_CONFIG:
        print(f"Uso: python -m src.embeddings [mpnet|minilm]  (por defecto: mpnet)")
        sys.exit(1)
    matrix, titles = run(model_id=model_id)
    print(
        f"Listo: matriz {matrix.shape} ({model_id}) y {len(titles)} títulos → "
        f"data/processed/movie_embeddings_{model_id}.npy/.txt"
    )
