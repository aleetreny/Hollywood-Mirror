# Hollywood Mirror Frontend

React + Vite application for private semantic movie search in the browser.

The app uses a Web Worker and Transformers.js to run the quantized MiniLM model locally. It compares the query embedding with the committed movie embedding matrix and does not call a backend API.

## Requirements

- Node.js 20 or newer.

## Development

```bash
npm ci
npm run dev
```

The local server runs at `http://localhost:3000`.

No `.env` file or API URL is required.

## Quality and build

- `npm run typecheck`: TypeScript type-check.
- `npm run lint`: alias for `typecheck`.
- `npm run build`: production build with Vite.
- `npm run preview`: local preview of the production build.
- `npm run clean`: remove `dist/`.
- `npm run check`: type-check plus production build.

## Production

The frontend is deployed directly to Vercel with `frontend/` as the project root and `dist/` as the output directory.

The first visit downloads the model and static search index. Subsequent visits use the browser cache, and all query text remains on the user's device.
