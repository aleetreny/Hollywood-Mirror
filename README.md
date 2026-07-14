# Hollywood Mirror

Hollywood Mirror is a private semantic movie search app that runs entirely in the browser.

- `frontend/` contains the React + Vite application.
- `frontend/src/search/` contains the Web Worker that runs MiniLM locally with Transformers.js.
- `data/processed/` stores the committed embedding artifacts used by the web app and analysis.
- `src/` contains the offline data-processing and embedding pipeline.
- `analysis/` contains the Quarto report and is intentionally separate from the web deployment.

The production app has no API server, container, warm-up service, or Hugging Face Space dependency. User queries stay on the device.

## Run the web app locally

Requirements:

- Node.js 20 or newer.

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

The development server runs at `http://localhost:3000`.

Useful checks:

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run build
npm --prefix frontend run check
```

The first browser visit downloads the quantized MiniLM model and the static movie index. Those assets are cached by the browser for later visits.

## Production deployment

The only production deployment is the Vercel frontend at:

- `https://hollywood-mirror.vercel.app`

Vercel project settings:

- Root directory: `frontend`
- Framework: Vite
- Install command: `npm ci`
- Build command: `npm run build`
- Output directory: `dist`
- Environment variables: none required

No rewrite, proxy, serverless function, Python runtime, or external inference service is needed.

## Offline data pipeline

The Python environment is only required to regenerate data and analysis artifacts.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected local-only inputs include:

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

## Quarto report

The report remains independent from the deployed web app.

```bash
cd analysis
quarto render galaxia.qmd
```
