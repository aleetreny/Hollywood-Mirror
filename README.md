# Hollywood Mirror

Hollywood Mirror is now organized as a web-first monorepo:

- `frontend/` contains the React + Vite client.
- `src/api.py` exposes the semantic search API.
- `data/processed/` stores the committed embedding artifacts required by the web app.
- `analysis/` contains the Quarto report and is intentionally separate from the web deploy path.


## Quick Start

### 1. Install the full local environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm --prefix frontend ci
```

### 2. Run the API

```bash
python -m uvicorn src.api:app --reload --port 8000
```

The API starts on `http://localhost:8000`.

### 3. Run the frontend

```bash
cp frontend/.env.example frontend/.env
npm --prefix frontend run dev
```

The frontend starts on `http://localhost:3000`.

## Web Architecture

The web application only depends on:

- `src/`
- `frontend/`
- `data/processed/movie_embeddings_*.npy`
- `data/processed/movie_embeddings_*.txt`

Everything related to Quarto, exploratory analysis, raw screenplay data, and local
metadata is excluded from the web deployment path.

### API behavior

Main routes:

- `GET /healthz`
- `GET /api/capabilities`
- `POST /api/similar-movies`
- `POST /api/warmup`

Search models currently supported:

- `minilm`
- `mpnet`

The API normalizes embedding matrices once, caches repeated query vectors in memory,
and exposes a dedicated warmup route for the default model.
It also defaults to public CORS (`*`) so a static frontend can call it from Vercel,
Cloudflare, Netlify, or Hugging Face without extra credentials setup.

Cold-start strategy for first-time visitors:

- the frontend calls `POST /api/warmup` as soon as the page opens
- the default model (`minilm`) is already baked into the Hugging Face Docker image
- the backend container preloads `minilm` on startup in Docker deployments
- the search button stays disabled while the initial warmup is still running

## Python Dependencies

The repository uses:

- `requirements.txt` for the full local environment
- `requirements-web.txt` for the production API runtime only

## Frontend Build

Useful commands:

```bash
npm --prefix frontend run build
npm --prefix frontend run check
```

The frontend defaults to:

- local API: `http://localhost:8000`
- production API: same origin, unless `VITE_API_BASE_URL` is explicitly set
- for Vercel + Hugging Face, set `VITE_API_BASE_URL` to the public Space URL

## Data Pipeline

The data pipeline remains available but is not required to run the web app.

Expected local-only inputs:

- `data/raw/`
- `data/metadata/movie_meta_data.csv`

Pipeline entry points:

```bash
python -m src.parsing
python -m src.extract_metrics
python -m src.embeddings mpnet
python -m src.embeddings minilm
python -m src.precompute
```

## Quarto Report

The report is intentionally out of the web deployment path.

```bash
cd analysis
quarto render galaxia.qmd
```

## Deployment Notes

### Backend container

The Docker image is trimmed for the API runtime:

- installs only `requirements-web.txt`
- downloads and bakes the default `minilm` model during image build
- copies only `src/` and `data/processed/`
- excludes `frontend/`, `analysis/`, raw data, and generated report artifacts

### Static frontend providers

The repository includes:

- `vercel.json`

It keeps the Vercel build rooted on `frontend/dist` even when the repo root is used,
and proxies `/api/*` requests to the Hugging Face Space backend.

### Hugging Face Spaces

`upload_hf.py` now ignores analysis artifacts, frontend assets, raw data, and generated
CSV/Parquet files so uploads stay focused on the API runtime.