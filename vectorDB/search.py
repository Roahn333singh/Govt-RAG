import os
import time

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PayloadSchemaType,
    Prefetch,
    SparseVector,
    VectorParams,
)

from app.request_context import note_timing

load_dotenv(override=True)

_tracer = trace.get_tracer(__name__)

# Config
QDRANT_URL = os.getenv(
    "CLUSTER_ENDPOINT",
    "http://localhost:6333",
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = "my-collection"
CACHE_COLLECTION_NAME = "semantic-query-cache"

# Rerank / retrieve pool sizes — balanced for recall vs Cloud Run memory
PREFETCH_LIMIT = 40
RRF_CANDIDATES = int(os.getenv("RRF_CANDIDATES", "28"))
# Cross-encoder cost grows with input length; chunks rarely need more than this to rank
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "1200"))
DEFAULT_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"

# Init Client and Embedder
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=int(os.getenv("QDRANT_TIMEOUT_SECONDS", "15")),
    check_compatibility=False,
)


reranker = None
dense_embedder = None
sparse_embedder = None
_services_ready = False


def init_services():
    global reranker, sparse_embedder, dense_embedder, _services_ready

    # Import inside the function to avoid macOS top-level proxy freeze during Uvicorn startup
    print("Loading AI libraries (this might take a moment)...")
    from fastembed import SparseTextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    try:
        from vectorDB.embeddings import get_dense_embedder
    except ImportError:
        from embeddings import get_dense_embedder

    print("Checking Qdrant collections...")
    if not qdrant_client.collection_exists(CACHE_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=CACHE_COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        qdrant_client.create_payload_index(
            collection_name=CACHE_COLLECTION_NAME,
            field_name="language",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    print("Loading models (dense, sparse, and reranker)...")
    if dense_embedder is None:
        dense_embedder = get_dense_embedder()
    if sparse_embedder is None:
        sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    if reranker is None:
        model = os.getenv("RERANKER_MODEL") or DEFAULT_RERANKER
        reranker = TextCrossEncoder(model_name=model)
    _services_ready = True
    print("Services ready!")


def services_ready() -> bool:
    return bool(
        _services_ready and dense_embedder and sparse_embedder and reranker
    )


def run_hybrid_search(
    question: str,
    top_k: int = 6,
    dense_vector: list[float] | None = None,
):
    """
    Two-stage retrieval:
      Stage 1 — Hybrid Search (Dense + Sparse RRF)
      Stage 2 — Cross-Encoder Re-ranking

    Pass dense_vector to reuse an embedding already computed (e.g. cache lookup).
    """
    print(f"\nQuerying: '{question[:80]}'...")

    if dense_vector is not None:
        dense_query_vector = dense_vector
    else:
        started = time.perf_counter()
        with _tracer.start_as_current_span("embed.query") as span:
            span.set_attribute(
                "model",
                os.getenv("OPENROUTER_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"),
            )
            span.set_attribute(
                "dims", int(os.getenv("OPENROUTER_EMBEDDING_DIMENSIONS", "1536"))
            )
            try:
                dense_query_vector = dense_embedder.embed_query(question)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                raise
        note_timing("embed_ms", (time.perf_counter() - started) * 1000)
    sparse_query_vector = list(sparse_embedder.embed([question]))[0]

    started = time.perf_counter()
    with _tracer.start_as_current_span("qdrant.hybrid_search") as span:
        span.set_attribute("collection", COLLECTION_NAME)
        span.set_attribute("limit", RRF_CANDIDATES)
        try:
            candidates = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=dense_query_vector,
                        using="abstract-dense-vector",
                        limit=PREFETCH_LIMIT,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_query_vector.indices,
                            values=sparse_query_vector.values,
                        ),
                        using="sparse-vector",
                        limit=PREFETCH_LIMIT,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=RRF_CANDIDATES,
            ).points
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise
        span.set_attribute("num_results", len(candidates))
    note_timing("qdrant_ms", (time.perf_counter() - started) * 1000)

    if not candidates:
        return []

    started = time.perf_counter()
    with _tracer.start_as_current_span("rerank") as span:
        span.set_attribute("model", os.getenv("RERANKER_MODEL") or DEFAULT_RERANKER)
        span.set_attribute("num_docs", len(candidates))
        span.set_attribute("top_k", top_k)
        try:
            doc_texts = [(r.payload.get("text") or "")[:RERANK_MAX_CHARS] for r in candidates]
            scores = list(reranker.rerank(question, doc_texts))
            scored_pairs = sorted(
                zip(scores, candidates), key=lambda x: x[0], reverse=True
            )
            reranked = []
            for score, candidate in scored_pairs[:top_k]:
                candidate.score = score
                reranked.append(candidate)
            print(
                f"✅ Re-ranked {len(candidates)} candidates → returning top {len(reranked)}"
            )
            return reranked
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            print(f"⚠️  Re-ranking failed ({e}), falling back to RRF order")
            return candidates[:top_k]
        finally:
            note_timing("rerank_ms", (time.perf_counter() - started) * 1000)
