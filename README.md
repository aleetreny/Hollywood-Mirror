# Hollywood Mirror

Sistema de análisis de datos cinematográficos con **NLP** y **embeddings** que mapea el estilo narrativo de ~2.600 películas a partir de guiones anotados.

## Productos

1. **Exploración científica (Quarto)** — Reducción de dimensionalidad (UMAP), visualización de la “Galaxia del Cine” y análisis de clusters sobre el espacio latente de películas.
2. **Producto de software (Web App)** — Frontend React/Vite + API FastAPI: el usuario pega un texto y obtiene el Top‑K de películas más similares según embeddings.

## Base de datos

**`data/raw/`** — Base de datos del proyecto: **un JSON por película** (~2.600 archivos). Cada JSON contiene escenas con bloques (`head_type`, `text`, `head_text`). El pipeline lee directamente esta carpeta.

## Estructura del repositorio

```
Hollywood Mirror/
├── context.md              # Especificación del proyecto (en español)
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # Base de datos: 1 JSON por película (escenas → bloques)
│   └── processed/          # DataFrame limpio + embeddings (.parquet, .npy, .txt)  [NO en git]
├── src/                    # Backend / pipeline en Python
│   ├── parsing.py          # Extracción y limpieza desde JSON → DataFrame [movie_title, cleaned_text]
│   ├── embeddings.py       # Chunking, vectorización (Sentence Transformers), mean pooling
│   ├── api.py              # API FastAPI: /api/similar-movies (Top‑K películas similares)
│   └── utils.py            # Utilidades varias
├── analysis/               # Documento Quarto (UMAP, Plotly, clusters, conclusiones)
└── frontend/               # Web app React/Vite generada con Google AI Studio
    ├── package.json
    └── src/…               # Componentes de UI y cliente HTTP (Vite)
```

## Pipeline de datos

1. **Parsing:** Extraer `text` de bloques con `head_type` `heading` o `speaker`/`title`; ignorar `transition`. Salida: DataFrame `[movie_title, cleaned_text]`.
2. **Embeddings:** Chunking (tamaño según modelo) → modelo Sentence Transformers (`all-mpnet-base-v2` por defecto, 768 dims; alternativa `all-MiniLM-L6-v2`, 384 dims) → mean pooling por película → matriz `movie_embeddings.npy` + títulos en `movie_embeddings.txt`. Si ya existen y `force=False`, no se regeneran.

## Uso rápido

```bash
# Entorno (ejemplo con conda; también puedes usar venv)
conda create -n hollywood python=3.11 -y
conda activate hollywood
pip install -r requirements.txt

# 1) Limpieza de datos (JSON → DataFrame):
python -m src.parsing

# 2) Embeddings
# mpnet (768 dims, más calidad; requiere máquina algo potente):
python -m src.embeddings mpnet
# o bien, alternativa más ligera:
python -m src.embeddings minilm

# 3) API backend (FastAPI)
uvicorn src.api:app --reload --port 8000

# 4) Web frontend (Vite) – desde la carpeta frontend/
cd frontend
npm install
echo 'VITE_API_BASE_URL=http://localhost:8000' > .env
npm run dev
```

Para el análisis Quarto:

```bash
cd analysis
quarto render galaxia.qmd
```

## Requerimientos técnicos

- **Progreso:** `tqdm` en procesamiento masivo.
- **Persistencia:** Guardar estados intermedios; no regenerar embeddings en cada ejecución.
- **Modularidad:** La generación de embeddings es una función independiente usada por Quarto, la API y la web.
