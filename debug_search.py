import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

load_dotenv()
from vectorDB import search
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

search.init_services()
dense_embedder = search.dense_embedder
sparse_embedder = search.sparse_embedder
reranker = search.reranker
qdrant_client = search.qdrant_client
COLLECTION_NAME = search.COLLECTION_NAME

QUERY = sys.argv[1] if len(sys.argv) > 1 else "What is the URL for the Pragati portal?"
TOP_RRF = int(os.getenv("DEBUG_TOP_RRF", "15"))
TOP_CE = int(os.getenv("DEBUG_TOP_CE", "10"))
TEXT_CHARS = int(os.getenv("DEBUG_TEXT_CHARS", "400"))


def _print_hit(i: int, score_label: str, score: float, point) -> None:
    pl = point.payload or {}
    text = (pl.get("text") or "").replace("\n", " ")
    if len(text) > TEXT_CHARS:
        text = text[:TEXT_CHARS] + "…"

    print(f"  #{i:02d} [{score_label} {score:.4f}]")
    print(f"      point_id     : {point.id}")
    print(f"      document-id  : {pl.get('document-id')}")
    print(f"      source       : {pl.get('source')}")
    print(f"      page         : {pl.get('page')}")
    print(f"      category     : {pl.get('category')} | {pl.get('category_clean')}")
    print(f"      data_source  : {pl.get('data_source')}")
    print(f"      language     : {pl.get('language')}")
    print(f"      text         : {text!r}")
    print()


def debug_search(question: str) -> None:
    print("=" * 70)
    print(f"  QUERY: {question}")
    print("=" * 70)

    dense_vec = dense_embedder.embed_query(question)
    sparse_obj = list(sparse_embedder.embed([question]))[0]

    print(f"\n[STAGE 1] Raw RRF candidates (before reranking) — top {TOP_RRF}:\n")
    candidates = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="abstract-dense-vector", limit=40),
            Prefetch(
                query=SparseVector(
                    indices=sparse_obj.indices.tolist(),
                    values=sparse_obj.values.tolist(),
                ),
                using="sparse-vector",
                limit=40,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=TOP_RRF,
        with_payload=True,
    ).points

    if not candidates:
        print("  NO RESULTS — collection may be empty or doc was not ingested!")
        return

    for i, c in enumerate(candidates, 1):
        _print_hit(i, "RRF", c.score, c)

    print(f"\n[STAGE 2] After cross-encoder reranking (top {TOP_CE}):\n")
    scores = list(reranker.rerank(question, [c.payload["text"] for c in candidates]))
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:TOP_CE]
    for i, (s, c) in enumerate(ranked, 1):
        _print_hit(i, "CE", s, c)

    print("TIP: Stage 2 is what the chat app roughly feeds the LLM.")
    print("     Copy document-id / source / point_id to filter or delete in Qdrant.")
    print("     Dashboard filter example:  document-id:Nivida Path.pdf")


if __name__ == "__main__":
    debug_search(QUERY)
