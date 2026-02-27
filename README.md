# Hollywood Mirror

Hollywood Mirror is a cinematic analysis project built on NLP and semantic embeddings. It includes a Python data pipeline, a FastAPI backend, a React/Vite frontend, and a Quarto report.

## What is included

1. Scientific analysis in Quarto (`analysis/galaxia.qmd`) with UMAP projections, clustering, and editorial interpretation.
2. A semantic search web app where users submit free text and get Top-K similar movies.
3. A reproducible pipeline for script parsing, NLP metrics extraction, and embedding generation.

## Repository structure

```text
Hollywood Mirror/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                        # Local screenplay JSON files (not committed)
│   ├── metadata/                   # Local metadata CSV files (not committed)
│   └── processed/                  # Mixed: committed embedding files + local generated artifacts
├── src/
│   ├── parsing.py                  # JSON -> movies_cleaned.{parquet,csv}
│   ├── extract_metrics.py          # NLP metrics -> movie_metrics.csv
│   ├── embeddings.py               # mpnet/minilm embeddings -> .npy + .txt
│   ├── precompute.py               # merge + UMAP -> galaxia_precalc.parquet
│   └── api.py                      # endpoint POST /api/similar-movies
├── analysis/
│   ├── galaxia.qmd
│   ├── custom.scss
│   └── _quarto.yml
└── frontend/
    ├── package.json
    ├── .env.example
    └── src/
```

## Requirements

- Python 3.10 or newer.
- Node.js 20 or newer.
- Quarto CLI, only required to render the report.

Base Python setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data source

Primary dataset:

- Kaggle `gufukuro/movie-scripts-corpus`:
  https://www.kaggle.com/datasets/gufukuro/movie-scripts-corpus?resource=download

Git tracking in this repository is intentionally selective:

- `data/raw/` is ignored (`data/raw/` in `.gitignore`) and is not committed.
- `data/metadata/` is effectively local-only because `*.csv` is ignored globally.
- `data/processed/` is partially committed.
  - Committed now: `movie_embeddings_mpnet.npy`, `movie_embeddings_mpnet.txt`,
    `movie_embeddings_minilm.npy`, `movie_embeddings_minilm.txt`.
  - Local-only by ignore rules: generated `*.csv` and `*.parquet` files such as
    `movies_cleaned.csv`, `movies_cleaned.parquet`, `movie_metrics.csv`, and
    `galaxia_precalc.parquet`.

From the Kaggle dataset, this project uses the `movie_metadata` subset, especially:

- `movie_meta_data.csv` (required by current pipeline)
- `screenplay_awards.csv` (optional, useful for extra analysis)

Example using `kagglehub`:

```python
# pip install kagglehub[pandas-datasets] pandas
import kagglehub
from kagglehub import KaggleDatasetAdapter

meta = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "gufukuro/movie-scripts-corpus",
    "movie_metadata/movie_meta_data.csv",
)

awards = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "gufukuro/movie-scripts-corpus",
    "movie_metadata/screenplay_awards.csv",
)

meta.to_csv("data/metadata/movie_meta_data.csv", index=False)
awards.to_csv("data/metadata/screenplay_awards.csv", index=False)
```

Also place screenplay JSON files from the same Kaggle dataset under `data/raw/`.

## Data pipeline

```bash
# 1) Parse screenplay JSON files
python -m src.parsing

# 2) Compute NLP metrics
python -m src.extract_metrics

# 3) Build embeddings (both models supported by the API)
python -m src.embeddings mpnet
python -m src.embeddings minilm

# 4) Build precomputed dataset for Quarto
python -m src.precompute
```

Expected artifacts in `data/processed/`:

- `movies_cleaned.parquet` and `movies_cleaned.csv`
- `movie_metrics.csv`
- `movie_embeddings_mpnet.npy` and `movie_embeddings_mpnet.txt`
- `movie_embeddings_minilm.npy` and `movie_embeddings_minilm.txt`
- `galaxia_precalc.parquet`

Commit status note:

- Committed: the two embedding pairs (`*.npy` + `*.txt`).
- Ignored by `.gitignore`: generated `*.csv` and `*.parquet` artifacts.

## Backend API

```bash
uvicorn src.api:app --reload --port 8000
```

Main endpoint:

- `POST /api/similar-movies`
- Body JSON:
  - `text`: string
  - `model`: `mpnet` or `minilm`
  - `k`: integer (1-50)

## Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Environment variable:

- `VITE_API_BASE_URL`, default is `http://localhost:8000`.

Useful commands:

- `npm run lint`
- `npm run build`

## Quarto report

```bash
cd analysis
quarto render galaxia.qmd
```

HTML output is written to `analysis/_site/`.
