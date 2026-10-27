"""Shared dense embedder: OpenRouter Qwen3-Embedding-8B.

All ingestion scripts and the live search path must use the *same*
embedding model and output dimensionality, otherwise vectors written to
Qdrant are not comparable with query vectors.

Configuration (environment):
    OPENROUTER_API_KEY              required
    OPENROUTER_EMBEDDING_MODEL      default: qwen/qwen3-embedding-8b
    OPENROUTER_EMBEDDING_DIMENSIONS default: 1536
"""

import os
import random
import time

import httpx

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_DIMENSIONS = 1536

# Qwen3 embedding models are instruction-aware on the query side. Document
# chunks are embedded without an instruction; search queries use this one
# consistently everywhere (hybrid search + semantic cache).
QUERY_INSTRUCTION = (
    "Given a search query about Uttar Pradesh government documents, "
    "retrieve relevant passages that answer the query."
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenRouterEmbeddings:
    """Minimal LangChain-style embedder backed by the OpenRouter API.

    Exposes the same interface the codebase already uses:
        embed_documents(list[str]) -> list[list[float]]
        embed_query(str)           -> list[float]
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 32,
        timeout: float = 20.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file."
            )
        self.model = model or os.getenv("OPENROUTER_EMBEDDING_MODEL", DEFAULT_MODEL)
        self.dimensions = int(
            dimensions
            or os.getenv("OPENROUTER_EMBEDDING_DIMENSIONS", DEFAULT_DIMENSIONS)
        )
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        # One pooled client for the process. httpx.Client is safe to share across threads.
        self._http = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    # ── Internal request with retry ──────────────────────────────────────
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._http.post(OPENROUTER_EMBEDDINGS_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                vectors = [item["embedding"] for item in items]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"OpenRouter returned {len(vectors)} embeddings "
                        f"for {len(texts)} inputs."
                    )
                for vector in vectors:
                    if len(vector) != self.dimensions:
                        raise RuntimeError(
                            f"OpenRouter returned a {len(vector)}-dim vector; "
                            f"expected {self.dimensions}."
                        )
                return vectors
            except httpx.HTTPStatusError as e:
                detail = e.response.text[:500]
                last_error = RuntimeError(f"OpenRouter HTTP {e.response.status_code}: {detail}")
                if e.response.status_code not in _RETRYABLE_STATUS:
                    raise last_error from e
            except (httpx.HTTPError, TimeoutError, RuntimeError) as e:
                last_error = e

            if attempt < self.max_retries - 1:
                # Jitter so concurrent retries don't hit OpenRouter in lockstep
                delay = min(2.0, 0.5 * (2**attempt)) + random.uniform(0, 0.5)
                print(
                    f"      ⚠️  Embedding request failed ({last_error}); "
                    f"retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        raise RuntimeError(
            f"OpenRouter embedding failed after {self.max_retries} attempts: {last_error}"
        )

    # ── LangChain-style interface ────────────────────────────────────────
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        safe_texts = [text if text and text.strip() else " " for text in texts]
        sub_batches = [
            safe_texts[i : i + self.batch_size]
            for i in range(0, len(safe_texts), self.batch_size)
        ]
        if len(sub_batches) == 1:
            return self._embed_batch(sub_batches[0])

        # Run sub-batches concurrently (order-preserving) for throughput.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(4, len(sub_batches))) as pool:
            results = list(pool.map(self._embed_batch, sub_batches))
        return [vector for batch in results for vector in batch]

    def embed_query(self, text: str) -> list[float]:
        instructed = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"
        return self._embed_batch([instructed])[0]


def get_dense_embedder() -> OpenRouterEmbeddings:
    return OpenRouterEmbeddings()
