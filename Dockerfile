FROM python:3.10-slim

RUN useradd -m -u 1000 user

USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME="/home/user/.cache/huggingface"
ENV SENTENCE_TRANSFORMERS_HOME="/home/user/.cache/sentence-transformers"
ENV API_PRELOAD_MODELS="minilm"

WORKDIR /app

COPY --chown=user ./requirements-web.txt requirements-web.txt

RUN pip install --no-cache-dir --upgrade -r requirements-web.txt
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Default model cached in image: minilm")
PY

COPY --chown=user ./src ./src
COPY --chown=user ./data/processed ./data/processed

ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
