# Hollywood Mirror

Sistema de análisis de datos cinematográficos con **NLP** y **embeddings** que mapea el estilo narrativo de ~2.600 películas a partir de guiones anotados.

## Productos

1. **Exploración científica (Quarto)** — Reducción de dimensionalidad (UMAP), visualización de la “Galaxia del Cine” y análisis de clusters.
2. **Producto de software (Streamlit)** — App web: el usuario pega un fragmento de texto y obtiene el Top 5 de películas más similares y su posición en la galaxia.

## Base de datos

**`data/raw/`** — Base de datos del proyecto: **un JSON por película** (~2.600 archivos). Cada JSON contiene escenas con bloques (`head_type`, `text`, `head_text`). El pipeline lee directamente esta carpeta.

## Estructura del repositorio

```
Hollywood Mirror/
├── context.md              # Especificación del proyecto
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # Base de datos: 1 JSON por película (escenas → bloques)
│   └── processed/          # DataFrame [movie_title, cleaned_text], matriz .npy/.parquet
├── src/
│   ├── parsing.py          # Extracción y limpieza desde JSON
│   ├── embeddings.py       # Chunking, vectorización, mean pooling (reutilizable)
│   └── utils.py            # Progreso (tqdm), persistencia
├── analysis/               # Documento Quarto (UMAP, Plotly, clusters)
├── app/                    # Aplicación Streamlit
└── artifacts/              # Estados intermedios (evitar regenerar embeddings)
```

## Pipeline de datos

1. **Parsing:** Extraer `text` de bloques con `head_type` `heading` o `speaker`/`title`; ignorar `transition`. Salida: DataFrame `[movie_title, cleaned_text]`.
2. **Embeddings:** Chunking ~300 palabras (modelo por defecto `all-mpnet-base-v2`, 768 dims; overlap 10 %) → mean pooling por película → matriz en `movie_embeddings.npy` + `movie_embeddings.txt`. Opción `minilm` para 384 dims y chunks 200 palabras. Si ya existen, no se regeneran.

## Uso rápido

```bash
# Entorno
python -m venv .venv
source .venv/bin/activate   # o .venv\Scripts\activate en Windows
pip install -r requirements.txt

# Limpieza de datos (JSON → DataFrame): ~2 min para 2.600 películas
python -m src.parsing

# Embeddings (mpnet por defecto: 768 dims, chunks 300 palabras). ~30–60 min en CPU
python -m src.embeddings
# Alternativa más rápida (minilm, 384 dims): python -m src.embeddings minilm

# App Streamlit (cuando esté implementada)
streamlit run app/app.py

# Quarto: renderizar desde analysis/ (instalar Quarto por separado)
quarto render analysis/
```

## Requerimientos técnicos

- **Progreso:** `tqdm` en procesamiento masivo.
- **Persistencia:** Guardar estados intermedios; no regenerar embeddings en cada ejecución.
- **Modularidad:** La generación de embeddings es una función independiente usada por Quarto y por la app.
