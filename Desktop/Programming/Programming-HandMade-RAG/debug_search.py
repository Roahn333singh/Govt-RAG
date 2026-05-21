import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()
from vectorDB.search import dense_embedder, sparse_embedder, reranker, qdrant_client, COLLECTION_NAME
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

QUERY = sys.argv[1] if len(sys.argv) > 1 else "What is the URL for the Pragati portal?"

def debug_search(question):
    print("=" * 70)
    print(f"  QUERY: {question}")
    print("=" * 70)

    dense_vec  = dense_embedder.embed_query(question)
    sparse_obj = list(sparse_embedder.embed([question]))[0]

    print("\n[STAGE 1] Raw RRF candidates (before reranking):\n")
    candidates = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="abstract-dense-vector", limit=25),
            Prefetch(query=SparseVector(indices=sparse_obj.indices.tolist(), values=sparse_obj.values.tolist()), using="sparse-vector", limit=25),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=15,
    ).points

    if not candidates:
        print("  NO RESULTS — collection may be empty or doc was not ingested!"); return

    for i, c in enumerate(candidates):
        text = c.payload.get("text","")[:100].replace("\n"," ")
        doc  = c.payload.get("document-id","?")
        hit  = "★" if any(k in text.lower() for k in ["pragati","upsdc","fmisc","igrs"]) else " "
        print(f"  {hit} #{i+1:02d} [RRF {c.score:.4f}] {doc}")
        print(f"        {text!r}\n")

    print("\n[STAGE 2] After cross-encoder reranking (top 6):\n")
    scores = list(reranker.rerank(question, [c.payload["text"] for c in candidates]))
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:6]
    for i, (s, c) in enumerate(ranked):
        text = c.payload.get("text","")[:100].replace("\n"," ")
        doc  = c.payload.get("document-id","?")
        hit  = "★" if any(k in text.lower() for k in ["pragati","upsdc","fmisc","igrs"]) else " "
        print(f"  {hit} #{i+1} [CE {s:+.4f}] {doc}")
        print(f"        {text!r}\n")

    all_text = " ".join(c.payload.get("text","") for _,c in ranked).lower()
    print("VERDICT:", "✅ 'pragati' found in top-6" if "pragati" in all_text else "❌ 'pragati' NOT in top-6 — retrieval miss")

debug_search(QUERY)
