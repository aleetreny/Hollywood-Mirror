# Hollywood Mirror Frontend

React + Vite frontend for semantic similarity search against the Hollywood Mirror API.

## Requirements

- Node.js 20 or newer.

## Setup

1. Install dependencies:

```bash
npm install
```

2. Create environment variables:

```bash
cp .env.example .env
```

3. Update `VITE_API_BASE_URL` in `.env` if your backend is not running at `http://localhost:8000`.
   If you omit it, local development defaults to `http://localhost:8000` and
   production defaults to same-origin API requests.
   For a Vercel frontend talking to Hugging Face Spaces, point it at the public Space URL.

## Development

```bash
npm run dev
```

Local server runs at `http://localhost:3000`.

## Quality and build

- `npm run typecheck`: TypeScript type-check.
- `npm run lint`: alias for `typecheck`.
- `npm run build`: production build with Vite.
- `npm run preview`: local preview of the production build.
- `npm run clean`: remove `dist/`.
- `npm run check`: type-check plus production build.

On page load, the app triggers a background backend warmup request and keeps the
search action disabled until that initial warmup finishes, so the first real search
is less likely to pay the full cold-start cost.
