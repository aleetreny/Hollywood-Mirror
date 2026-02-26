# Proyecto: Hollywood Mirror — Análisis Geométrico y Buscador Semántico de Cine

## 1. Visión General del Proyecto
**Hollywood Mirror** es un sistema de análisis de datos cinematográficos que utiliza **Natural Language Processing (NLP)** y **Embeddings** para mapear el estilo narrativo de 2,600 películas. El proyecto se divide en dos productos diferenciados:
1.  **Exploración Científica (Quarto):** Un análisis de reducción de dimensionalidad para visualizar la "Galaxia del Cine".
2.  **Producto de Software (Streamlit):** Una aplicación web que identifica a qué película se asemeja un fragmento de texto introducido por el usuario.

## 2. Especificación de los Datos (Dataset)
Utilizamos la base de datos en **`data/raw/`**: un JSON por película (origen: `rule_based_annotations`). Cada archivo tiene una estructura de "Escenas" que contienen "Bloques".

### Estructura de un bloque JSON:
- `head_type`: Clasificación del fragmento.
    - **`heading`**: Descripciones de acción y escena. (Relevante para el estilo visual/ritmo).
    - **`speaker/title`**: Diálogos de personajes. (Relevante para la voz y tono).
    - **`transition`**: Instrucciones técnicas tipo "FADE IN". (**Ignorar** para el modelo).
- `text`: El contenido textual crudo.
- `head_text`: Metadatos sobre la localización (INT/EXT), personajes y momento del día.

## 3. Pipeline de Procesamiento (Data Engineering)
El agente de programación debe implementar el siguiente flujo de trabajo:

### A. Extracción y Limpieza (Parsing)
- Iterar sobre los 2,600 archivos JSON en `data/raw/`.
- Extraer solo los campos `text` asociados a `heading` y `speaker/title`.
- Concatenar el texto por película, eliminando caracteres especiales innecesarios.
- **Output:** Un DataFrame de Pandas con columnas `[movie_title, cleaned_text]`.

### B. Estrategia de Embeddings (Chunking & Pooling)
Dado que los guiones exceden el límite de tokens de los modelos Transformer:
1.  **Chunking:** Dividir cada guion en bloques de ~300 palabras (modelo por defecto: `all-mpnet-base-v2`, max 384 tokens) con solapamiento del 10%.
2.  **Vectorización:** Procesar cada bloque con `all-mpnet-base-v2` (768 dims; alternativa: `all-MiniLM-L6-v2`, 384 dims).
3.  **Mean Pooling:** Calcular el vector promedio de todos los bloques de una película para obtener su **Vector Identidad**.
4.  **Almacenamiento:** Guardar la matriz `[N, 768]` (o 384 si se usa MiniLM) en `.npy` + títulos en `.txt`.

## 4. Vía 1: Análisis Geométrico (Documento Quarto)
El objetivo es el estudio de la estructura del espacio latente:
- **Reducción de Dimensiones:** Aplicar **UMAP** (Uniform Manifold Approximation and Projection) para reducir las 384 dimensiones a 2D o 3D.
- **Visualización Interactiva:** Crear un scatter plot con **Plotly** donde cada punto sea una película.
- **Análisis de Clusters:** Identificar cómo se agrupan los géneros de forma natural basándose en el texto, no en etiquetas previas.
- **Métricas:** Calcular distancias de similitud entre películas famosas (ej: comparar la cercanía de *Inception* con el resto de la filmografía de Nolan).

## 5. Vía 2: Aplicación Web (Streamlit)
El objetivo es crear una herramienta interactiva:
- **Input:** Un `st.text_area` donde el usuario pega un guion propio o una idea.
- **Inferencia:**
    1.  Generar el embedding del texto del usuario en tiempo real.
    2.  Calcular la **Similitud del Coseno** contra la matriz de 2,600 películas.
- **Output:** - Top 5 de películas más similares.
    - Porcentaje de afinidad estilística.
    - Visualización del punto del usuario dentro de la "Galaxia de Películas".

## 6. Requerimientos Técnicos para el Agente
- **Progreso:** Utilizar `tqdm` para mostrar el progreso durante el procesamiento masivo.
- **Persistencia:** Guardar estados intermedios. No se deben regenerar los embeddings cada vez que se ejecute la app.
- **Modularidad:** El código de generación de embeddings debe ser una función independiente reutilizable tanto por el documento Quarto como por la App.