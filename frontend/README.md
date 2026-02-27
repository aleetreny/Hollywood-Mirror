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

## Development

```bash
npm run dev
```

Local server runs at `http://localhost:3000`.

## Quality and build

- `npm run lint`: TypeScript type-check.
- `npm run build`: production build with Vite.
- `npm run preview`: local preview of the production build.
- `npm run clean`: remove `dist/`.
