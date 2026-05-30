import os
import sys
import time
import uuid
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
try:
    from langchain_community.document_loaders import Docx2txtLoader
except ImportError:
    Docx2txtLoader = None

from fastembed import SparseTextEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import LangDetectException, detect
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from langchain_core.documents import Document
try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    import pytesseract
except ImportError:
    pass

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("CLUSTER_ENDPOINT", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = "my-collection"

DENSE_VECTOR_NAME = "abstract-dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"

ANSARI_ROOT = os.path.join(os.path.dirname(__file__), "..", "ansari AI")
SKIP_LOG = os.path.join(os.path.dirname(__file__), "..", "ansari_skipped_files.log")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# ── Metadata Extractor ─────────────────────────────────────────────────────────
def get_metadata_from_path(filepath: str, root_dir: str) -> dict:
    rel_path = os.path.relpath(filepath, root_dir)
    parts = rel_path.split(os.sep)
    filename = parts[-1]

    if len(parts) == 1:
        category = "General"
        sub_folder = ""
    else:
        category = parts[0]
        sub_folder = parts[1] if len(parts) >= 3 else ""

    return {
        "category": category,
        "category_clean": category,
        "sub_folder": sub_folder,
        "data_source": "ansari_ai_folder",
        "file_type": os.path.splitext(filename)[1].lstrip(".").lower(),
        "ingestion_date": str(date.today()),
        "document-id": filename,
        "source": filepath,
    }

# ── Language Detection ─────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    try:
        if len(text.strip()) < 20:
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"

# ── Document Loader ────────────────────────────────────────────────────────────
def load_document(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(filepath).load()
    elif ext == ".txt":
        return TextLoader(filepath, encoding="utf-8").load()
    elif ext == ".docx":
        if Docx2txtLoader is None:
            raise ImportError("Install docx2txt: pip install docx2txt")
        return Docx2txtLoader(filepath).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_text_with_ocr(filepath: str):
    print(f"    [OCR Fallback] Getting page count...")
    try:
        info = pdfinfo_from_path(filepath)
        num_pages = info["Pages"]
    except Exception as e:
        print(f"    [OCR Error] Could not get PDF info: {e}")
        return []
        
    extracted_docs = []
    print(f"    [OCR Fallback] Processing {num_pages} pages...")
    for i in range(1, num_pages + 1):
        try:
            print(f"      - OCR page {i}/{num_pages}", end="\r")
            images = convert_from_path(filepath, first_page=i, last_page=i, dpi=200)
            if images:
                text = pytesseract.image_to_string(images[0], lang='hin+eng')
                if text.strip():
                    extracted_docs.append(Document(page_content=text, metadata={"page": i - 1, "source": filepath}))
                images[0].close()
        except Exception as e:
            print(f"\n      - OCR Error on page {i}: {e}")
    print()  # newline after progress
    return extracted_docs

def is_krutidev_gibberish(text: str) -> bool:
    """Detect if the text is legacy non-Unicode (Kruti Dev) gibberish."""
    kd_markers = ['¼', '½', 'ç', '¶', 'ñ', 'gS', 'ds', 'esa', 'dh', 'vkSj', 'djus', 'fofufnZ"V', 'x;s', 'tk;s']
    count = sum(1 for marker in kd_markers if marker in text)
    # If we find 2 or more distinct Kruti Dev markers in the text, it's highly likely gibberish
    return count >= 2

def is_text_extractable(docs) -> bool:
    total_text = "".join(d.page_content for d in docs).strip()
    if len(total_text) <= 50:
        return False
        
    # Check if the extracted text is actually Kruti Dev gibberish
    if is_krutidev_gibberish(total_text[:2000]):
        print("    [OCR Trigger] Detected Kruti Dev gibberish encoding. Forcing OCR...")
        return False
        
    return True

# ── Chunker ────────────────────────────────────────────────────────────────────
def split_document(docs, chunk_size=900, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)

# ── Point Builder ──────────────────────────────────────────────────────────────
def build_points(chunks, file_metadata: dict, dense_embedder, sparse_embedder) -> list:
    original_texts = [chunk.page_content for chunk in chunks]
    enriched_texts = [
        f"[Document: {file_metadata['document-id']} | Category: {file_metadata['category_clean']}]\n{text}"
        for text in original_texts
    ]

    dense_vecs = []
    batch_size = 10
    for i in range(0, len(enriched_texts), batch_size):
        batch = enriched_texts[i : i + batch_size]
        try:
            batch_dense_vecs = dense_embedder.embed_documents(batch)
            if len(batch_dense_vecs) != len(batch):
                print(f"      ⚠️  Dense batch embedding returned {len(batch_dense_vecs)}/{len(batch)} vectors. Falling back to per-chunk embedding.")
                batch_dense_vecs = [dense_embedder.embed_documents([text])[0] for text in batch]
            dense_vecs.extend(batch_dense_vecs)
            time.sleep(1) # Small sleep to avoid typical rate limits
        except Exception as e:
            error_str = str(e)
            if any(err in error_str for err in ["429", "RESOURCE_EXHAUSTED", "503", "500", "UNAVAILABLE"]):
                print(f"      ⚠️  API Error ({error_str.split()[0]}) — sleeping 60 s then retrying...")
                time.sleep(60)
                print("      🔄 Retrying batch...")
                try:
                    batch_dense_vecs = dense_embedder.embed_documents(batch)
                    if len(batch_dense_vecs) != len(batch):
                        batch_dense_vecs = [dense_embedder.embed_documents([text])[0] for text in batch]
                    dense_vecs.extend(batch_dense_vecs)
                except Exception as inner_e:
                    print(f"      ❌ Second attempt failed: {inner_e}")
                    raise inner_e
            else:
                raise

    sparse_vecs = list(sparse_embedder.embed(original_texts))

    if len(dense_vecs) != len(chunks) or len(sparse_vecs) != len(chunks):
        raise RuntimeError(f"Embedding mismatch: chunks={len(chunks)}, dense={len(dense_vecs)}, sparse={len(sparse_vecs)}")

    points = []
    lang = detect_language(original_texts[0]) if original_texts else "unknown"

    for i, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
        payload = {
            **file_metadata,
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", i),
            "language": lang,
        }
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
            },
            payload=payload,
        )
        points.append(point)
    return points

# ── Qdrant Uploader ────────────────────────────────────────────────────────────
def upload_to_qdrant(client, points: list, batch_size=50):
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

# ── Clear Collection ───────────────────────────────────────────────────────────
def clear_qdrant_collection(client):
    print("\n" + "=" * 60)
    print(f"  🗑️  Deleting ALL existing points from '{COLLECTION_NAME}'")
    print("=" * 60)
    
    if not client.collection_exists(COLLECTION_NAME):
        print(f"  🛠️  Collection '{COLLECTION_NAME}' not found. It will be created.")
        return

    # Delete points in batches using scroll
    while True:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break
        ids = [p.id for p in points]
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=ids),
        )
        print(f"  🗑️  Deleted batch of {len(ids)} points...")

    print("  ✅ Collection cleared completely.\n")

# ── Ingest Ansari folder ───────────────────────────────────────────────────────
def ingest_ansari_folder(qdrant_client, dense_embedder, sparse_embedder):
    print("\n" + "=" * 60)
    print(f"  📂 Ingesting folder: {ANSARI_ROOT}")
    print("=" * 60)

    all_files = []
    
    # Optional: Allow passing a single file path as argument for targeted re-ingestion
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        if os.path.exists(target_file):
            all_files.append(target_file)
            print(f"  🎯 Targeted ingestion for single file: {target_file}")
        else:
            print(f"  ❌ Targeted file not found: {target_file}")
            return
    else:
        for root, _, filenames in os.walk(ANSARI_ROOT):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_EXTENSIONS and not fname.startswith("."):
                    all_files.append(os.path.join(root, fname))

    total = len(all_files)
    if total == 0:
        print(f"  ⚠️  No supported files found in ansari AI folder.\n")
        return

    print(f"  Found {total} supported files to process.\n")

    uploaded = 0
    skipped = 0
    errors = 0

    with open(SKIP_LOG, "a", encoding="utf-8") as skip_log:
        for idx, filepath in enumerate(all_files, 1):
            fname = os.path.basename(filepath)
            print(f"\n  [{idx}/{total}] Processing: {fname}")

            try:
                print("    - Loading document...")
                docs = load_document(filepath)
                
                ext = os.path.splitext(filepath)[1].lower()
                used_ocr = False

                if not is_text_extractable(docs):
                    if ext == ".pdf":
                        print(f"    - No text found. Attempting OCR...")
                        docs = extract_text_with_ocr(filepath)
                        used_ocr = True
                        if not docs:
                            print(f"    ⏭  Skipped (image-only / empty PDF, OCR yielded no text)")
                            skip_log.write(f"[SKIP_IMAGE] {filepath}\n")
                            skipped += 1
                            continue
                    else:
                        print(f"    ⏭  Skipped (empty file)")
                        skip_log.write(f"[SKIP_EMPTY] {filepath}\n")
                        skipped += 1
                        continue

                print("    - Splitting document into chunks...")
                chunks = split_document(docs)
                if not chunks:
                    print(f"    ⏭  Skipped (0 chunks after splitting)")
                    skip_log.write(f"[SKIP_EMPTY] {filepath}\n")
                    skipped += 1
                    continue

                print(f"    - Generated {len(chunks)} chunks. Building embeddings...")
                file_metadata = get_metadata_from_path(filepath, ANSARI_ROOT)
                points = build_points(chunks, file_metadata, dense_embedder, sparse_embedder)
                
                print(f"    - Uploading to Qdrant...")
                upload_to_qdrant(qdrant_client, points)

                uploaded += 1
                lang = detect_language(chunks[0].page_content)
                print(f"    ✅ Successfully ingested {len(chunks)} chunks | category='{file_metadata['category_clean']}' | lang='{lang}'")

            except Exception as e:
                print(f"    ❌ Error: {e}")
                skip_log.write(f"[ERROR] {filepath}: {e}\n")
                errors += 1

    print(f"\n  ── Summary ─────────────────────────────────────────")
    print(f"  ✅ Uploaded : {uploaded}")
    print(f"  ⏭  Skipped  : {skipped}")
    print(f"  ❌ Errors   : {errors}")
    print(f"  ────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    if not os.path.exists(ANSARI_ROOT):
        print(f"\n❌ ERROR: ansari AI folder not found at:\n   {ANSARI_ROOT}")
        exit(1)

    print("\n🔌 Connecting to Qdrant and loading embedders...")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120.0)
    dense_embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=1536,
    )
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("✅ Embedders ready!\n")

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"🛠️  Collection '{COLLECTION_NAME}' not found. Creating it...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=1536, distance=Distance.COSINE)
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )

    # 1. Clear existing database points only if running a full ingestion
    if len(sys.argv) <= 1:
        clear_qdrant_collection(qdrant_client)
    else:
        print("⏭️  Skipping collection clear because this is a targeted single-file ingestion.")

    # 2. Ingest ansari AI folder completely
    ingest_ansari_folder(qdrant_client, dense_embedder, sparse_embedder)

    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"📊 Total vectors now in '{COLLECTION_NAME}': {info.points_count:,}\n")
    print("✅ Ansari AI ingestion complete!")
