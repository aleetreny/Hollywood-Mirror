"""
Extract and clean text from screenplay JSON files in data/raw/.
Output: DataFrame with columns [movie_title, cleaned_text].

Only blocks with head_type 'heading' or 'speaker/title' are kept;
'transition' blocks are ignored.
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


# Block types we keep ('transition' is excluded).
KEEP_HEAD_TYPES = {"heading", "speaker/title"}


def _clean_fragment(text: str) -> str:
    """Clean a text fragment by normalizing whitespace and control characters."""
    if not text or not isinstance(text, str):
        return ""
    # Normalize whitespace (tabs, repeated spaces, etc.).
    text = re.sub(r"\s+", " ", text)
    # Drop control characters except newlines and tabs.
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
    return text.strip()


def extract_text_from_file(path: Path, encoding: str = "utf-8") -> tuple[str, str]:
    """
    Read one movie JSON and return (movie_title, cleaned_text).

    - movie_title: filename stem (for example, "About Time_2194499").
    - cleaned_text: all `text` values from blocks with head_type in
      KEEP_HEAD_TYPES, cleaned and joined by spaces.
    """
    raw = path.read_text(encoding=encoding)
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

    # Keep filename stem so title+imdbid mapping stays stable.
    stem = path.stem
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
    Iterate over all JSON files in raw_dir, extract and clean text,
    and return a DataFrame with columns [movie_title, cleaned_text].
    """
    paths = sorted(raw_dir.glob(pattern))
    rows = []
    for path in tqdm(paths, desc="Parsing JSONs"):
        try:
            title, text = extract_text_from_file(path, encoding=encoding)
            rows.append({"movie_title": title, "cleaned_text": text})
        except Exception as e:
            tqdm.write(f"Error in {path.name}: {e}")
            continue

    return pd.DataFrame(rows)


def run(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    output_name: str = "movies_cleaned",
) -> pd.DataFrame:
    """
    Run parsing pipeline: read data/raw/, write to data/processed/.

    - raw_dir: directory with JSON files (default: repo/data/raw).
    - processed_dir: output directory (default: repo/data/processed).
    - output_name: output basename (.parquet and .csv are written).

    Returns the generated DataFrame.
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
    print(f"Done: {len(df)} movies -> data/processed/movies_cleaned.parquet and .csv")
