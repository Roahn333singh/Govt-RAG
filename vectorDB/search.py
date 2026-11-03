import asyncio
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
# One cross-encoder pass over the fused pool, not one pass per query variant.
RERANK_POOL = int(os.getenv("RERANK_POOL", "20"))
RRF_K = 60
# Cross-encoder sequence length. MiniLM's native max is 512; longer inputs
# (jina's default) are what blew the 8 GiB Cloud Run limit.
RERANK_MAX_TOKENS = int(os.getenv("RERANK_MAX_TOKENS", "512"))
DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"

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
# One rerank per instance. Other requests wait instead of stacking ONNX sessions.
_rerank_slots = asyncio.Semaphore(1)


def init_services():
    global reranker, sparse_embedder, dense_embedder, _services_ready

    # Import inside the function to avoid macOS top-level proxy freeze during Uvicorn startup
    print("Loading AI libraries (this might take a moment)...")
    from fastembed import SparseTextEmbedding

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

    from vectorDB.rerankers import load_reranker, selected_name

    reranker_name = selected_name()
    print(f"Loading models (dense, sparse, and reranker={reranker_name})...")
    if dense_embedder is None:
        dense_embedder = get_dense_embedder()
    if sparse_embedder is None:
        sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    if reranker is None:
        reranker = load_reranker(reranker_name)
    _services_ready = True
    print("Services ready!")


def services_ready() -> bool:
    return bool(
        _services_ready and dense_embedder and sparse_embedder and reranker
    )


def _cap_rerank_length(encoder) -> None:
    """Bound the ONNX sequence so a long-context model cannot be selected by accident."""
    tok = getattr(getattr(encoder, "model", None), "tokenizer", None)
    if tok is not None and hasattr(tok, "enable_truncation"):
        tok.enable_truncation(max_length=RERANK_MAX_TOKENS)


def tokenizer_of(obj):
    """FastEmbed keeps the tokenizer on encoder.model; ONNX backends keep theirs on the wrapper."""
    if obj is None:
        return None
    direct = getattr(obj, "tokenizer", None)
    if direct is not None and hasattr(direct, "encode"):
        return direct
    nested = getattr(getattr(obj, "model", None), "tokenizer", None)
    if nested is not None and hasattr(nested, "encode"):
        return nested
    deeper = getattr(getattr(getattr(obj, "model", None), "model", None), "tokenizer", None)
    if deeper is not None and hasattr(deeper, "encode"):
        return deeper
    return None


def truncate_text(text: str, tokenizer=None) -> str:
    """Keep about RERANK_MAX_TOKENS tokens. Char cap is the fallback."""
    text = text or ""
    if tokenizer is not None:
        try:
            enc = tokenizer.encode(text, add_special_tokens=False)
            ids = list(getattr(enc, "ids", enc))
            if len(ids) <= RERANK_MAX_TOKENS:
                return text
            return tokenizer.decode(ids[:RERANK_MAX_TOKENS])
        except Exception:
            pass
    return text[: RERANK_MAX_TOKENS * 4]


def truncate_for_rerank(text: str) -> str:
    return truncate_text(text, tokenizer_of(reranker))


def _hit_key(hit) -> tuple:
    payload = hit.payload or {}
    return (
        payload.get("document-id"),
        payload.get("page"),
        (payload.get("text") or "")[:120],
    )


def rrf_merge(batches: list, limit: int = RERANK_POOL) -> list:
    """Fuse ranked lists from query variants. Higher reciprocal rank wins."""
    scores: dict[tuple, float] = {}
    chosen: dict[tuple, object] = {}
    for batch in batches:
        for rank, hit in enumerate(batch or [], start=1):
            key = _hit_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in chosen:
                chosen[key] = hit
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    from vectorDB.rerankers import stamp_rrf

    chosen_hits = []
    for key in ordered[:limit]:
        hit = chosen[key]
        stamp_rrf(hit, scores[key])
        chosen_hits.append(hit)
    return chosen_hits


def run_hybrid_search(
    question: str,
    limit: int = RRF_CANDIDATES,
    dense_vector: list[float] | None = None,
):
    """
    Hybrid search only (dense + sparse, fused with RRF inside Qdrant).

    Cross-encoder rerank is a separate once-per-request step. Pass dense_vector
    to reuse an embedding already computed for the cache lookup.
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
        span.set_attribute("limit", limit)
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
                limit=limit,
            ).points
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise
        span.set_attribute("num_results", len(candidates))
    note_timing("qdrant_ms", (time.perf_counter() - started) * 1000)

    return candidates


def rerank_hits(question: str, candidates: list, top_k: int):
    """Score one fused pool. Documents are truncated before the cross-encoder."""
    if not candidates:
        return []
    if reranker is None:
        raise RuntimeError("Reranker is not loaded. Refusing to fall back to another backend.")
    started = time.perf_counter()
    with _tracer.start_as_current_span("rerank") as span:
        span.set_attribute("model", getattr(reranker, "model_id", DEFAULT_RERANKER))
        span.set_attribute("reranker.name", getattr(reranker, "name", "minilm"))
        span.set_attribute("num_docs", len(candidates))
        span.set_attribute("top_k", top_k)
        span.set_attribute("max_length", RERANK_MAX_TOKENS)
        try:
            pairs = reranker.rerank(question, candidates, top_k)
            reranked = []
            for candidate, score in pairs:
                candidate.score = float(score)
                reranked.append(candidate)
            span.set_attribute("batch_size", int(getattr(reranker, "batch_size", len(candidates))))
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
            elapsed_ms = (time.perf_counter() - started) * 1000
            span.set_attribute(
                "batch_size", int(getattr(reranker, "batch_size", len(candidates)))
            )
            span.set_attribute("ms", round(elapsed_ms, 2))
            note_timing("rerank_ms", elapsed_ms)


async def rerank_once(question: str, candidates: list, top_k: int):
    """One rerank at a time on this instance. Callers wait on the semaphore."""
    from app.telemetry import run_traced_in_threadpool

    async with _rerank_slots:
        return await run_traced_in_threadpool(rerank_hits, question, candidates, top_k)
