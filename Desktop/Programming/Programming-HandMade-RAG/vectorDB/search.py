import os

from dotenv import load_dotenv
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

load_dotenv(override=True)

# Config
QDRANT_URL = os.getenv(
    "CLUSTER_ENDPOINT",
    "https://d63776f5-a8d8-4f7f-baed-49b914dfe0a1.us-east-1-1.aws.cloud.qdrant.io",
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my-collection"
CACHE_COLLECTION_NAME = "semantic-query-cache"


# Init Client and Embedder
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)


reranker = None
dense_embedder = None
sparse_embedder = None
reranker = None

def init_services():
    global reranker, sparse_embedder, dense_embedder
    
    # Import inside the function to avoid macOS top-level proxy freeze during Uvicorn startup
    print("Loading AI libraries (this might take a moment)...")
    from fastembed import SparseTextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    print("Checking Qdrant collections...")
    if not qdrant_client.collection_exists(CACHE_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=CACHE_COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        # Tell Qdrant to build a turbo-fast index for our language filter!
        qdrant_client.create_payload_index(
            collection_name=CACHE_COLLECTION_NAME,
            field_name="language",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    
    print("Loading models (dense, sparse, and reranker)...")
    if dense_embedder is None:
        dense_embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            output_dimensionality=1536,
        )
    if sparse_embedder is None:
        sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    if reranker is None:
        reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    print("Services ready!")

# reranker is now initialized in init_services() to prevent import hanging


def run_hybrid_search(question: str, top_k: int = 6):
    """
    Two-stage retrieval:
      Stage 1 — Hybrid Search (Dense + Sparse RRF): casts a wide net over the corpus.
      Stage 2 — Cross-Encoder Re-ranking: precisely scores every candidate by reading
                 the query and each chunk together, then returns only the best top_k.
    """
    print(f"\nQuerying: '{question}'...")

    # ── Stage 1: Hybrid Search ─────────────────────────────────────────────────
    dense_query_vector = dense_embedder.embed_query(question)
    sparse_query_vector = list(sparse_embedder.embed([question]))[0]

    # Fetch more candidates than we need so the reranker has room to work
    candidates = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_query_vector,
                using="abstract-dense-vector",
                limit=40,  # was 25 — fetch more from each branch
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_query_vector.indices,
                    values=sparse_query_vector.values,
                ),
                using="sparse-vector",
                limit=40,  # was 25
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=25,  # was 15 — give reranker a wider pool to work with
    ).points

    if not candidates:
        return []

    # ── Stage 2: Cross-Encoder Re-ranking ─────────────────────────────────────
    # The cross-encoder reads (query, chunk) pairs together — much more precise
    # than embedding similarity alone, at the cost of slightly more compute.
    try:
        doc_texts = [r.payload["text"] for r in candidates]
        # rerank() returns one float score per document, in the same order as input.
        # Higher score = more relevant. Zip, sort descending, take top_k.
        scores = list(reranker.rerank(question, doc_texts))
        scored_pairs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        reranked = []
        for s, candidate in scored_pairs[:top_k]:
            candidate.score = s
            reranked.append(candidate)
            
        print(
            f"✅ Re-ranked {len(candidates)} candidates → returning top {len(reranked)}"
        )
        return reranked
    except Exception as e:
        print(f"⚠️  Re-ranking failed ({e}), falling back to RRF order")
        return candidates[:top_k]


# if __name__ == "__main__":
#     # Let's see if the database knows the answer to this!
#     print(run_hybrid_search("What are the objectives of the Computer Centre?"))
