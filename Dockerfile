# Manual AI RAG — Cloud Run image
# Model weights are baked in a throwaway stage so torch/optimum never ship.
FROM python:3.12-slim AS modelbake

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HOME=/tmp/hf \
    OMP_NUM_THREADS=2 \
    ONNXRUNTIME_INTRA_OP_NUM_THREADS=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY scripts/bake_models.py scripts/bake_onnx_rerankers.py /tmp/
RUN mkdir -p /app/models && python /tmp/bake_models.py
# optimum+torch live in /tmp/opt and are not copied into the runtime image
RUN uv pip install --python /app/.venv/bin/python --target /tmp/opt 'optimum[onnxruntime]' sentencepiece
ENV PYTHONPATH=/tmp/opt
RUN python /tmp/bake_onnx_rerankers.py

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

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --from=modelbake /app/models /app/models

COPY agent ./agent
COPY app ./app
COPY vectorDB ./vectorDB
COPY govt_manual_ui ./govt_manual_ui
COPY scripts ./scripts
COPY main.py ./

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
