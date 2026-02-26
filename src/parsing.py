"""
Extracción y limpieza de texto desde los JSON en data/raw/.
Output: DataFrame con columnas [movie_title, cleaned_text].

Solo se extraen bloques con head_type 'heading' o 'speaker/title';
se ignoran 'transition'.
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


# Tipos de bloque que nos interesan (transición se ignora)
KEEP_HEAD_TYPES = {"heading", "speaker/title"}


def _clean_fragment(text: str) -> str:
    """Limpia un fragmento de texto: normaliza espacios y caracteres innecesarios."""
    if not text or not isinstance(text, str):
        return ""
    # Normalizar espacios (incl. tabs, múltiples espacios)
    text = re.sub(r"\s+", " ", text)
    # Eliminar caracteres de control
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
    return text.strip()


def extract_text_from_file(path: Path) -> tuple[str, str]:
    """
    Lee un JSON de película y devuelve (movie_title, cleaned_text).

    - movie_title: nombre del archivo sin extensión ni ID (ej. "About Time_2194499" -> "About Time").
    - cleaned_text: concatenación de todos los `text` de bloques con head_type
      en KEEP_HEAD_TYPES, limpios y separados por espacio.
    """
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    fragments: list[str] = []
    for scene in data:
        if not isinstance(scene, list):
            continue
        for block in scene:
            if not isinstance(block, dict):
                continue
            if block.get("head_type") not in KEEP_HEAD_TYPES:
                continue
            t = block.get("text")
            if t:
                fragments.append(_clean_fragment(t))

    # Título: quitar .json; opcionalmente quitar sufijo _IMDBID para nombre "limpio"
    stem = path.stem
    # Formato típico: "Movie Title_1234567" -> dejamos el stem completo para identificar
    movie_title = stem

    cleaned_text = " ".join(f for f in fragments if f)
    return movie_title, cleaned_text


def parse_raw_to_dataframe(
    raw_dir: Path,
    *,
    pattern: str = "*.json",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Itera sobre todos los JSON en raw_dir, extrae y limpia el texto,
    y devuelve un DataFrame con columnas [movie_title, cleaned_text].
    """
    paths = sorted(raw_dir.glob(pattern))
    rows = []
    for path in tqdm(paths, desc="Parsing JSONs"):
        try:
            title, text = extract_text_from_file(path)
            rows.append({"movie_title": title, "cleaned_text": text})
        except Exception as e:
            tqdm.write(f"Error en {path.name}: {e}")
            continue

    return pd.DataFrame(rows)


def run(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    output_name: str = "movies_cleaned",
) -> pd.DataFrame:
    """
    Ejecuta el pipeline de parsing: lee data/raw/, escribe en data/processed/.

    - raw_dir: carpeta con JSON (por defecto data/raw respecto al repo).
    - processed_dir: carpeta de salida (por defecto data/processed).
    - output_name: nombre base del archivo (se guarda .parquet y .csv).

    Devuelve el DataFrame generado.
    """
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = raw_dir or repo_root / "data" / "raw"
    processed_dir = processed_dir or repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = parse_raw_to_dataframe(raw_dir)
    base = processed_dir / output_name
    df.to_parquet(f"{base}.parquet", index=False)
    df.to_csv(f"{base}.csv", index=False, encoding="utf-8")
    return df


if __name__ == "__main__":
    df = run()
    print(f"Listo: {len(df)} películas → data/processed/movies_cleaned.parquet y .csv")
