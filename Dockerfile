# Usa una imagen base de Python oficial
FROM python:3.10-slim

# Crea el usuario no root obligatorio para Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user

# Cambia al nuevo usuario
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Crea el directorio de trabajo
WORKDIR /app

# Copia los archivos de requerimientos con los permisos correctos
COPY --chown=user ./requirements.txt requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Pre-descarga de modelos de IA en el arranque (en la caché del usuario) para evitar Timeouts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copia el resto del código del proyecto (archivos, etc.)
COPY --chown=user . /app

# Expone el puerto por defecto de HuggingFace Spaces (7860)
ENV PORT=7860
EXPOSE 7860

# Comando para ejecutar la app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
