# Manual AI RAG — Cloud Run image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HOME=/app/models/hf \
    OMP_NUM_THREADS=2 \
    ONNXRUNTIME_INTRA_OP_NUM_THREADS=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip uv

# Install deps first (better layer cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Bake BM25 + MiniLM into the image so cold starts never download models
COPY scripts/bake_models.py /tmp/bake_models.py
RUN mkdir -p /app/models && .venv/bin/python /tmp/bake_models.py && rm /tmp/bake_models.py

# App code
COPY agent ./agent
COPY vectorDB ./vectorDB
COPY govt_manual_ui ./govt_manual_ui
COPY scripts ./scripts
COPY main.py ./

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
