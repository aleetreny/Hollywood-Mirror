"""
precompute.py
Runs once. Computes UMAP 2D + 3D, merges all metadata and NLP metrics,
and saves the final DataFrame to data/processed/galaxia_precalc.parquet.
After running this, quarto render is nearly instant.
"""

import re
import numpy as np
import pandas as pd
import umap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROC = REPO / "data" / "processed"
META = REPO / "data" / "metadata"
OUT  = PROC / "galaxia_precalc.parquet"

print("Loading embeddings...")
matrix     = np.load(PROC / "movie_embeddings_mpnet.npy")
titles_raw = (PROC / "movie_embeddings_mpnet.txt").read_text().strip().split("\n")

def clean_title(raw):
    if "_" in raw:
        raw = raw.rsplit("_", 1)[0]
    if raw.endswith(" IMDb"):
        raw = raw[:-5]
    return raw

def extract_imdbid(raw):
    if "_" in raw:
        return int(raw.rsplit("_", 1)[1])
    return -1

titles   = [clean_title(t) for t in titles_raw]
imdb_ids = [extract_imdbid(t) for t in titles_raw]
N, D = matrix.shape
print(f"  {N} movies, {D} dims")

print("Loading metadata and metrics...")
meta_raw = pd.read_csv(META / "movie_meta_data.csv")
metrics  = pd.read_csv(PROC / "movie_metrics.csv")

df = pd.DataFrame({"title": titles, "title_raw": titles_raw, "imdbid": imdb_ids, "emb_idx": range(N)})
df = df.merge(meta_raw, on="imdbid", how="left", suffixes=("", "_meta"))
metrics_indexed = metrics.set_index("movie_title")
for col in ["word_count", "lexical_diversity", "sentiment", "subjectivity", "avg_sentence_length"]:
    df[col] = df["title_raw"].map(metrics_indexed[col])

assert len(df) == N

def parse_dollars(val):
    if pd.isna(val): return np.nan
    m = re.search(r"\$([\d,]+)", str(val))
    return int(m.group(1).replace(",", "")) if m else np.nan

df["budget_usd"]   = df["budget"].apply(parse_dollars)
df["opening_usd"]  = df["opening weekend"].apply(parse_dollars)
df["imdb_rating"]  = df["imdb user rating"].replace(-1, np.nan)
df["meta_score"]   = df["metascore"].replace(-1, np.nan)
df["year_clean"]   = df["year"].replace(-1, np.nan)
df["primary_genre"] = df["genres"].apply(
    lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown"
)

# Top genres for display
top_genres = df["primary_genre"].value_counts().head(10).index.tolist()
df["genre_display"] = df["primary_genre"].apply(lambda g: g if g in top_genres else "Other")

print("Computing UMAP 2D (this takes ~90s)...")
r2 = umap.UMAP(n_neighbors=25, min_dist=0.1, n_components=2, metric="cosine", random_state=42)
c2 = r2.fit_transform(matrix)
df["umap_x"] = c2[:, 0]
df["umap_y"] = c2[:, 1]

print("Computing UMAP 3D (this takes ~90s)...")
r3 = umap.UMAP(n_neighbors=25, min_dist=0.1, n_components=3, metric="cosine", random_state=42)
c3 = r3.fit_transform(matrix)
df["umap_3d_x"] = c3[:, 0]
df["umap_3d_y"] = c3[:, 1]
df["umap_3d_z"] = c3[:, 2]

# Save only the columns we need (exclude huge text columns)
keep_cols = [
    "title", "imdbid", "emb_idx",
    "year_clean", "primary_genre", "genre_display", "genres",
    "imdb_rating", "meta_score", "budget_usd", "opening_usd",
    "awards", "directors", "countries",
    "word_count", "lexical_diversity", "sentiment", "subjectivity", "avg_sentence_length",
    "umap_x", "umap_y", "umap_3d_x", "umap_3d_y", "umap_3d_z",
]
keep_cols = [c for c in keep_cols if c in df.columns]

print(f"Saving parquet to {OUT} ...")
df[keep_cols].to_parquet(OUT, index=False)
print("Done! File size:", OUT.stat().st_size // 1024, "KB")
print(f"Top genres: {top_genres}")
