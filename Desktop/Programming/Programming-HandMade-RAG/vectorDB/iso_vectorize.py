import os
import re
import time
import uuid
from datetime import date

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
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = "my-collection"

DENSE_VECTOR_NAME = "abstract-dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"

ISO_ROOT = os.path.join(os.path.dirname(__file__), "..", "ISO_Ai_helpdesk_Document")
SKIP_LOG = os.path.join(os.path.dirname(__file__), "..", "skipped_files.log")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".JPG", ".TIF"}

# Skip files with more than this many chunks to avoid rate-limit blocking.
# These will be logged to skipped_files.log to be processed later.
MAX_CHUNKS_PER_FILE = 200

# ── Category Name Cleaner ──────────────────────────────────────────────────────
CATEGORY_MAP = {
    "1.establishment": "Establishment",
    "2.revenue": "Revenue",
    "3.design": "Design",
    "4.it": "IT",
    "5.flood": "Flood",
    "6.canals": "Canals",
    "7.dams": "Dams",
    "8.barrages": "Barrages",
    "9.acts and manuals": "Acts and Manuals",
    "10. accounts": "Accounts",
    "11. tenders and contract": "Tenders and Contract",
    "12.miscellaneous": "Miscellaneous",
    "13. drains": "Drains",
    "survey": "Survey",
    "cdd drg to iso 23-04-2025": "Engineering Drawings",
    "cdd drg. to iso  22-04-2025": "Engineering Drawings",
}


def clean_category_name(folder_name: str) -> str:
    """Convert '1.Establishment' → 'Establishment'"""
    key = folder_name.strip().lower()
    if key in CATEGORY_MAP:
        return CATEGORY_MAP[key]
    # Fallback: strip leading numbers and punctuation
    cleaned = re.sub(r"^\d+[\.\s]+", "", folder_name).strip()
    return cleaned if cleaned else folder_name


def is_engineering_drawing_folder(category: str) -> bool:
    return "cdd drg" in category.lower()


# ── Metadata Extractor ─────────────────────────────────────────────────────────
def get_metadata_from_path(filepath: str, iso_root: str) -> dict:
    """
    Extracts structured metadata from a file's path relative to ISO_ROOT.

    Handles two cases:
      1. Root-level file:   ISO_Root/filename.pdf
         → category = "General", sub_folder = "", project = ""

      2. Category file:     ISO_Root/6.Canals/HSR CANAL PART 1.pdf
         → category = "6.Canals", sub_folder = "", project = ""

      3. Deep file:         ISO_Root/CDD Drg/Bansagar Canal/file.pdf
         → category = "CDD Drg...", sub_folder = "Bansagar Canal", project = ""
    """
    rel_path = os.path.relpath(filepath, iso_root)
    parts = rel_path.split(os.sep)
    filename = parts[-1]

    # File is directly in the ISO root (no category folder)
    if len(parts) == 1:
        category = "root"
        sub_folder = ""
        project = ""
    else:
        category = parts[0]
        # sub_folder = 2nd level dir (only if file is 3+ levels deep)
        sub_folder = parts[1] if len(parts) >= 3 else ""
        # project    = 3rd level dir (only if file is 4+ levels deep)
        project = parts[2] if len(parts) >= 4 else ""

    # Map "root" to a clean readable name
    if category == "root":
        category_clean = "General"
    else:
        category_clean = clean_category_name(category)

    is_drawing = is_engineering_drawing_folder(category)

    return {
        "category": category,
        "category_clean": category_clean,
        "sub_folder": sub_folder,
        "project": project,
        "data_source": "ISO_Ai_helpdesk_Document",
        "file_type": os.path.splitext(filename)[1].lstrip(".").lower(),
        "is_engineering_drawing": is_drawing,
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


def is_text_extractable(docs) -> bool:
    """Returns True only if at least one page has meaningful text."""
    total_text = "".join(d.page_content for d in docs).strip()
    return len(total_text) > 50


# ── Chunker ────────────────────────────────────────────────────────────────────
def split_document(docs, chunk_size=900, chunk_overlap=150):
    """
    900 / 150 keeps table rows and multi-line paragraphs together.
    The old default (500/50) was splitting structured content mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# ── Point Builder ──────────────────────────────────────────────────────────────
def build_points(chunks, file_metadata: dict, dense_embedder, sparse_embedder) -> list:
    """
    Embed every chunk and return Qdrant PointStructs.

    Context enrichment: we prepend the document name AND category to the
    text BEFORE embedding. A chunk like "max file size 500 KB" now embeds
    as "[Document: IGRSFAQ.pdf | Category: IT] max file size 500 KB",
    which dramatically improves retrieval for short or context-poor chunks.

    The original text is stored in the payload so the LLM sees clean content.
    """
    original_texts = [chunk.page_content for chunk in chunks]

    # Context-enriched text used only for the dense embedding
    enriched_texts = [
        f"[Document: {file_metadata['document-id']} | Category: {file_metadata['category_clean']}]\n{text}"
        for text in original_texts
    ]

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
                    "      ⚠️  Dense batch embedding returned "
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
                print("      ⚠️  Rate limit — sleeping 60 s then retrying...")
                time.sleep(60)
                print("      🔄 Retrying batch...")
                batch_dense_vecs = dense_embedder.embed_documents(batch)
                if len(batch_dense_vecs) != len(batch):
                    print(
                        "      ⚠️  Dense batch embedding returned "
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
            f"{file_metadata['document-id']}: chunks={len(chunks)}, "
            f"dense={len(dense_vecs)}, sparse={len(sparse_vecs)}"
        )

    points = []

    # Detect language from first chunk
    lang = detect_language(original_texts[0]) if original_texts else "unknown"

    for i, (chunk, dense_vec, sparse_vec) in enumerate(
        zip(chunks, dense_vecs, sparse_vecs)
    ):
        payload = {
            **file_metadata,
            "text": chunk.page_content,  # original text for the LLM
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


# ── Qdrant Uploader (batched) ──────────────────────────────────────────────────
def upload_to_qdrant(client, points: list, batch_size=50):
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)


# ── Single-Folder Ingestion ────────────────────────────────────────────────────
def ingest_folder(
    folder_path: str,
    iso_root: str,
    qdrant_client,
    dense_embedder,
    sparse_embedder,
    root_only: bool = False,
):
    """Ingests all supported files from a single folder (and its subfolders, unless root_only=True)."""

    folder_name = os.path.basename(folder_path)
    print(f"\n{'=' * 60}")
    print(
        f"  📂 Ingesting folder: {folder_name if not root_only else 'Root Level Files'}"
    )
    print(f"{'=' * 60}")

    # Discover supported files in this folder only
    all_files = []
    if root_only:
        for fname in os.listdir(folder_path):
            full_path = os.path.join(folder_path, fname)
            if os.path.isfile(full_path):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_EXTENSIONS and not fname.startswith("."):
                    all_files.append(full_path)
    else:
        for root, _, filenames in os.walk(folder_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_EXTENSIONS and not fname.startswith("."):
                    all_files.append(os.path.join(root, fname))

    total = len(all_files)
    if total == 0:
        print(f"  ⚠️  No supported files found in this folder.\n")
        return

    print(f"  Found {total} supported files.\n")

    skipped_image = 0
    skipped_error = 0
    uploaded = 0

    with open(SKIP_LOG, "a", encoding="utf-8") as skip_log:
        for idx, filepath in enumerate(all_files, 1):
            fname = os.path.basename(filepath)
            print(f"  [{idx}/{total}] {fname}")

            try:
                docs = load_document(filepath)

                if not is_text_extractable(docs):
                    print(f"    ⏭  Skipped (image-only / empty PDF)")
                    skip_log.write(f"[SKIP] {filepath}\n")
                    skipped_image += 1
                    continue

                chunks = split_document(docs)
                if not chunks:
                    print(f"    ⏭  Skipped (0 chunks after splitting)")
                    continue

                if len(chunks) > MAX_CHUNKS_PER_FILE:
                    print(
                        f"    ⏭  Skipped (Too large: {len(chunks)} chunks). Will cause rate limits. Logged for later."
                    )
                    skip_log.write(f"[SKIP_LARGE] {filepath} ({len(chunks)} chunks)\n")
                    continue

                file_metadata = get_metadata_from_path(filepath, iso_root)
                points = build_points(
                    chunks, file_metadata, dense_embedder, sparse_embedder
                )
                upload_to_qdrant(qdrant_client, points)

                uploaded += 1
                lang = detect_language(chunks[0].page_content)
                print(
                    f"    ✅ {len(chunks)} chunks | category='{file_metadata['category_clean']}' | lang='{lang}'"
                )

            except Exception as e:
                print(f"    ❌ Error: {e}")
                skip_log.write(f"[ERROR] {filepath}: {e}\n")
                skipped_error += 1

            if idx < total:
                print(f"    ⏳ 15s rate-limit pause (Avoiding 429 Quota Errors)...")
                time.sleep(15)

    print(f"\n  ── Folder Summary ──────────────────────────────")
    print(f"  ✅ Uploaded : {uploaded}")
    print(f"  ⏭  Skipped  : {skipped_image} (image-only)")
    print(f"  ❌ Errors   : {skipped_error}")
    print(f"  ────────────────────────────────────────────────\n")


# ── Interactive Folder Menu ────────────────────────────────────────────────────
def show_menu(iso_root: str):
    """Lists all top-level category folders + a 'root files' option for the user to pick."""

    # Get all top-level directories
    entries = sorted(os.listdir(iso_root))
    folders = [
        e
        for e in entries
        if os.path.isdir(os.path.join(iso_root, e)) and not e.startswith(".")
    ]
    root_pdfs = [
        e
        for e in entries
        if os.path.isfile(os.path.join(iso_root, e))
        and os.path.splitext(e)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    options = []

    # Option 0: Root-level files
    if root_pdfs:
        options.append(
            (
                "__ROOT__",
                f"Root level files ({len(root_pdfs)} files) — GOs, Manuals, Rules",
            )
        )

    # One option per category folder
    for folder in folders:
        folder_path = os.path.join(iso_root, folder)
        count = sum(
            1
            for _, _, fnames in os.walk(folder_path)
            for f in fnames
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            and not f.startswith(".")
        )
        clean = clean_category_name(folder)
        options.append((folder, f"{clean:<30} ({count} supported files)  [{folder}]"))

    print(f"\n{'=' * 65}")
    print(f"  🗂️  ISO_Ai_helpdesk_Document — Folder Ingestion Menu")
    print(f"{'=' * 65}")

    for i, (_, label) in enumerate(options):
        print(f"  [{i + 1:>2}]  {label}")

    print(f"\n  [ 0]  Exit")
    print(f"{'=' * 65}\n")

    return options


if __name__ == "__main__":
    if not os.path.exists(ISO_ROOT):
        print(f"\n❌ ERROR: ISO folder not found at:\n   {ISO_ROOT}")
        print("Make sure the Google Drive symlink is active and synced.\n")
        exit(1)

    # Init clients once (reused across sessions)
    print("\n🔌 Connecting to Qdrant and loading embedders...")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
    dense_embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=1536,
    )
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("✅ Ready!\n")

    # Ensure collection exists before getting info
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(
            f"🛠️  Collection '{COLLECTION_NAME}' not found. Creating it with proper vector configs..."
        )
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=1536, distance=Distance.COSINE)
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )

    # Show current cluster state
    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"📊 Current vectors in '{COLLECTION_NAME}': {info.points_count:,}")

    while True:
        options = show_menu(ISO_ROOT)
        try:
            choice = int(input("Enter folder number to ingest (0 to exit): ").strip())
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Exiting.\n")
            break

        if choice == 0:
            print("\n👋 Exiting.\n")
            break

        if choice < 1 or choice > len(options):
            print("  ⚠️  Invalid choice. Try again.\n")
            continue

        folder_key, label = options[choice - 1]

        if folder_key == "__ROOT__":
            target_path = ISO_ROOT
            ingest_folder(
                target_path,
                ISO_ROOT,
                qdrant_client,
                dense_embedder,
                sparse_embedder,
                root_only=True,
            )
        else:
            target_path = os.path.join(ISO_ROOT, folder_key)
            ingest_folder(
                target_path,
                ISO_ROOT,
                qdrant_client,
                dense_embedder,
                sparse_embedder,
                root_only=False,
            )

        # Show updated cluster count after each folder
        info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"📊 Total vectors now in '{COLLECTION_NAME}': {info.points_count:,}\n")
        print("✅ Go test your bot! Come back and pick the next folder when ready.")
        input("\nPress Enter to return to the menu...")
