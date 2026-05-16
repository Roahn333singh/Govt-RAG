import os
import time
import uuid

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList, PointStruct, SparseVector

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv(
    "CLUSTER_ENDPOINT",
    "https://d63776f5-a8d8-4f7f-baed-49b914dfe0a1.us-east-1-1.aws.cloud.qdrant.io",
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = "my-collection"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DENSE_VECTOR_NAME = "abstract-dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"

# ── Clients ───────────────────────────────────────────────────────────────────
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

dense_embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=GOOGLE_API_KEY,
    output_dimensionality=1536,
)

sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

doc_path = DATA_DIR


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return loader.load()


def is_extractable(file_path: str, min_avg_chars: int = 80) -> bool:
    """Return True if the PDF has enough real extractable text (not a scanned image)."""
    try:
        docs = load_documents(file_path)
        if not docs:
            return False
        avg_chars = sum(len(d.page_content) for d in docs) / len(docs)
        return avg_chars >= min_avg_chars
    except Exception:
        return False


def split_document(docs, chunk_size=900, chunk_overlap=150):
    """
    900 / 150 keeps table rows and paragraphs together.
    The old default (500/50) was splitting tables mid-row.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_points(chunks, document_id):
    """
    Embed every chunk and return Qdrant PointStructs.

    Before embedding we prepend a context header:
        [Document: WaterSectorPortals.pdf]
        <chunk text>

    This makes every chunk's dense vector aware of which document it
    belongs to, dramatically improving retrieval for sparse chunks like
    table rows that have no surrounding context on their own.

    The original text (no header) is what gets stored in the payload
    so the LLM still receives clean, readable context.
    """
    original_texts = [chunk.page_content for chunk in chunks]
    enriched_texts = [f"[Document: {document_id}]\n{t}" for t in original_texts]

    # ── Dense embedding — batched with rate-limit retry ───────────────────
    #
    # NOTE:
    #   Some Gemini embedding integrations can return only 1 vector for a
    #   multi-text embed_documents(batch) call. If that happens, we fall back
    #   to per-chunk embedding for that batch to avoid silent data loss.
    dense_vecs = []
    batch_size = 10
    for i in range(0, len(enriched_texts), batch_size):
        batch = enriched_texts[i : i + batch_size]
        try:
            batch_dense_vecs = dense_embedder.embed_documents(batch)
            if len(batch_dense_vecs) != len(batch):
                print(
                    "    ⚠️  Dense batch embedding returned "
                    f"{len(batch_dense_vecs)}/{len(batch)} vectors. "
                    "Falling back to per-chunk embedding for this batch."
                )
                batch_dense_vecs = [
                    dense_embedder.embed_documents([text])[0] for text in batch
                ]
            dense_vecs.extend(batch_dense_vecs)
            time.sleep(1)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("    ⚠️  Rate limit — sleeping 60 s then retrying...")
                time.sleep(60)
                batch_dense_vecs = dense_embedder.embed_documents(batch)
                if len(batch_dense_vecs) != len(batch):
                    print(
                        "    ⚠️  Dense batch embedding returned "
                        f"{len(batch_dense_vecs)}/{len(batch)} vectors. "
                        "Falling back to per-chunk embedding for this batch."
                    )
                    batch_dense_vecs = [
                        dense_embedder.embed_documents([text])[0] for text in batch
                    ]
                dense_vecs.extend(batch_dense_vecs)
            else:
                raise

    # ── Sparse BM25 — on original text for clean keyword matching ─────────
    sparse_vecs = list(sparse_embedder.embed(original_texts))

    if len(dense_vecs) != len(chunks) or len(sparse_vecs) != len(chunks):
        raise RuntimeError(
            "Embedding/chunk size mismatch for "
            f"{document_id}: chunks={len(chunks)}, "
            f"dense={len(dense_vecs)}, sparse={len(sparse_vecs)}"
        )

    # ── Assemble points ────────────────────────────────────────────────────
    points = []
    for i, (chunk, dense_vec, sparse_vec) in enumerate(
        zip(chunks, dense_vecs, sparse_vecs)
    ):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload={
                    "document-id": document_id,
                    "text": chunk.page_content,  # original text for LLM
                    "page": chunk.metadata.get("page", i),
                    "source": chunk.metadata.get("source", ""),
                    "category": "data_folder",
                    "category_clean": "Data Folder",
                    "data_source": "DataFolder",
                    "file_type": "pdf",
                },
            )
        )
    return points


def delete_document_chunks(document_id: str) -> int:
    """
    Remove all Qdrant points for a given document_id.
    Always call this before re-ingesting a file to avoid duplicate chunks.
    """
    all_points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=2000,
        with_payload=True,
        with_vectors=False,
    )
    ids_to_delete = [
        p.id for p in all_points if p.payload.get("document-id") == document_id
    ]
    if ids_to_delete:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=ids_to_delete),
        )
        print(f"    🗑️  Deleted {len(ids_to_delete)} stale chunk(s) for '{document_id}'")
    return len(ids_to_delete)


def upload_documents(doc_path, force_reingest=False):
    """
    Ingest every PDF / TXT in doc_path.

    force_reingest=True  →  deletes existing chunks before re-uploading
                            (use this when fixing a broken ingestion)
    """
    files = [
        f
        for f in os.listdir(doc_path)
        if os.path.isfile(os.path.join(doc_path, f)) and f.endswith((".pdf", ".txt"))
    ]
    skipped = []

    for filename in files:
        file_path = os.path.join(doc_path, filename)
        print(f"\n{'=' * 55}")
        print(f"📄  {filename}")

        if not is_extractable(file_path):
            print("    ⚠️  Skipping — no extractable text (scanned PDF?)")
            skipped.append(filename)
            continue

        if force_reingest:
            delete_document_chunks(filename)

        try:
            docs = load_documents(file_path)
            chunks = split_document(docs)
            print(f"    {len(docs)} page(s) → {len(chunks)} chunks")
            points = build_points(chunks, filename)
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"    ✅  Uploaded {len(points)} points")
        except Exception as e:
            print(f"    ❌  Failed: {e}")
            skipped.append(filename)
            continue

        print("    ⏳  Waiting 15 s before next file...")
        time.sleep(15)

    print(f"\n{'=' * 55}")
    print(f"Done.  {len(files) - len(skipped)}/{len(files)} files ingested.")
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for f in skipped:
            print(f"  - {f}")


if __name__ == "__main__":
    upload_documents(doc_path, force_reingest=True)
