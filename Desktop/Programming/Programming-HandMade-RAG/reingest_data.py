"""
Full Re-ingestion Script
-------------------------
Phase A  →  Fixes the 8 data/ folder PDFs (all had only 1 chunk).
Phase B  →  Re-ingests the entire ISO_Ai_helpdesk_Document folder.

Both phases use the improved settings:
  - Chunk size 900 / overlap 150   (keeps tables and paragraphs intact)
  - Context-enriched embeddings    (document name + category prepended)
  - Batched embedding with retry   (handles Gemini free-tier rate limits)

Every skipped file or folder is recorded in skipped_files.log with:
  - Timestamp
  - Phase (A / B)
  - Reason code  (SCANNED | TOO_LARGE | ERROR | NO_FILES | EMPTY)
  - Full path
  - Details (e.g. chunk count, error message)

Usage:
    uv run python reingest_data.py                   # both phases (wipes first)
    uv run python reingest_data.py --fresh --data    # wipe all + re-ingest data/ only
    uv run python reingest_data.py --iso             # add ISO on top of existing data/
    uv run python reingest_data.py --summary         # print the skip log

Recommended first-run workflow:
    Step 1:  uv run python reingest_data.py --fresh --data
    Step 2:  uv run python debug_search.py  (verify retrieval)
    Step 3:  uv run python reingest_data.py --iso  (once happy with Step 2)
"""

import os
import sys
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

from fastembed import SparseTextEmbedding
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

from vectorDB.iso_vectorize import (
    GOOGLE_API_KEY,
    ISO_ROOT,
    QDRANT_API_KEY,
    QDRANT_URL,
    ingest_folder,
)
from vectorDB.vectorize import (
    COLLECTION_NAME,
    DATA_DIR,
    build_points,
    delete_document_chunks,
    load_documents,
    split_document,
)

# ── Shared clients ────────────────────────────────────────────────────────────
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

dense_embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=GOOGLE_API_KEY,
    output_dimensionality=1536,
)

sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

# The 8 data/ folder PDFs confirmed to have only 1 chunk
DATA_PDFS = [
    "WaterSectorPortals.pdf",
    "IGRSFAQ.pdf",
    "ComputerCentre.pdf",
    "CMISFAQ.pdf",
    "FMISC.pdf",
    "UPSWIC.pdf",
    "ISO.pdf",
    "Allportaldetail.pdf",
]

SKIP_LOG_FILE = os.path.join(os.path.dirname(__file__), "skipped_files.log")


# ─────────────────────────────────────────────────────────────────────────────
# SkipLogger
# ─────────────────────────────────────────────────────────────────────────────


class SkipLogger:
    """
    Records every skipped file or folder during ingestion.

    Reason codes:
        SCANNED    — PDF has no extractable text (image-based / scanned)
        TOO_LARGE  — Chunk count exceeds MAX_CHUNKS_PER_FILE limit
        ERROR      — Unexpected exception during processing
        NO_FILES   — Folder contains zero supported files
        EMPTY      — Document produced 0 chunks after splitting
        NOT_FOUND  — File does not exist on disk
    """

    REASON_DESC = {
        "SCANNED": "No extractable text — likely a scanned image PDF",
        "TOO_LARGE": "Too many chunks — would exhaust Gemini API quota",
        "ERROR": "Processing error",
        "NO_FILES": "Folder has no supported files (.pdf / .txt / .docx)",
        "EMPTY": "Produced 0 chunks after splitting",
        "NOT_FOUND": "File not found on disk",
    }

    def __init__(self, log_path: str = SKIP_LOG_FILE):
        self.log_path = log_path
        self.entries: list[dict] = []
        # Write a session-start banner so entries from different runs are separated
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n{'=' * 72}\n"
                f"SESSION  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'=' * 72}\n"
            )

    def log(self, path: str, reason: str, phase: str, details: str = "") -> None:
        """Record one skipped item — writes to file AND keeps in memory."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = dict(ts=ts, phase=phase, reason=reason, path=path, details=details)
        self.entries.append(entry)

        line = f"[{ts}] [Phase {phase}] [{reason:<9}] {path}"
        if details:
            line += f"  ← {details}"

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def summary(self) -> None:
        """Print a grouped, human-readable summary to stdout."""
        print(f"\n{'=' * 60}")
        if not self.entries:
            print("  SKIP SUMMARY — nothing was skipped ✅")
            print(f"{'=' * 60}")
            return

        print(f"  SKIP SUMMARY — {len(self.entries)} item(s) skipped")
        print(f"  Full log → {self.log_path}")
        print(f"{'=' * 60}")

        by_reason = {}
        for e in self.entries:
            by_reason.setdefault(e["reason"], []).append(e)

        for reason, items in sorted(by_reason.items()):
            desc = self.REASON_DESC.get(reason, reason)
            print(f"\n  [{reason}] {desc}  ({len(items)})")
            for e in items:
                name = os.path.relpath(e["path"], os.path.dirname(__file__))
                details = f"  ← {e['details']}" if e["details"] else ""
                print(f"      • {name}{details}")

        print(f"\n{'=' * 60}")

    @classmethod
    def print_log_file(cls, log_path: str = SKIP_LOG_FILE) -> None:
        """Print the raw contents of the skip log — useful for --summary flag."""
        if not os.path.exists(log_path):
            print("No skip log found yet.")
            return
        with open(log_path, encoding="utf-8") as f:
            print(f.read())


# ─────────────────────────────────────────────────────────────────────────────
# PHASE A — data/ folder
# ─────────────────────────────────────────────────────────────────────────────


def run_phase_a(logger: SkipLogger) -> None:
    print("\n" + "=" * 60)
    print("  PHASE A — Re-ingesting data/ folder PDFs")
    print("=" * 60)

    # ── A1: Diagnose ──────────────────────────────────────────────────────
    print("\n[A1] Text extractability check:\n")
    print(f"  {'FILE':<35}  {'PAGES':>5}  {'CHARS':>8}  STATUS")
    print(f"  {'-' * 35}  {'-' * 5}  {'-' * 8}  {'-' * 16}")

    extractable = []
    for filename in DATA_PDFS:
        file_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(file_path):
            print(f"  {filename:<35}  FILE NOT FOUND")
            logger.log(file_path, "NOT_FOUND", "A")
            continue

        try:
            docs = PyPDFLoader(file_path).load()
            total = sum(len(d.page_content) for d in docs)
            avg = total // max(len(docs), 1)
            ok = avg >= 80
            flag = "✅ extractable" if ok else "❌ likely scanned"
            print(f"  {filename:<35}  {len(docs):>5}  {total:>8,}  {flag}")

            if ok:
                extractable.append(filename)
            else:
                logger.log(
                    file_path,
                    "SCANNED",
                    "A",
                    f"{len(docs)} pages, {avg} avg chars/page",
                )
        except Exception as e:
            print(f"  {filename:<35}  ERROR: {e}")
            logger.log(file_path, "ERROR", "A", str(e))

    if not extractable:
        print("\n❌ No extractable PDFs found in data/ folder.")
        return

    print(f"\n  → {len(extractable)}/{len(DATA_PDFS)} files will be re-ingested.")

    # ── A2: Delete stale chunks (only needed for --data standalone run) ──
    # When both phases run, delete_all_points() already wiped everything.
    print(f"\n[A2] Deleting stale chunks...\n")
    for filename in extractable:
        delete_document_chunks(filename)

    # ── A3: Re-ingest ─────────────────────────────────────────────────────
    print(f"\n[A3] Re-ingesting...\n")
    for filename in extractable:
        file_path = os.path.join(DATA_DIR, filename)
        print(f"\n{'=' * 55}")
        print(f"📄  {filename}")
        try:
            docs = load_documents(file_path)
            chunks = split_document(docs)
            if not chunks:
                print("    ⚠️  0 chunks produced — skipping")
                logger.log(file_path, "EMPTY", "A")
                continue
            print(f"    {len(docs)} page(s) → {len(chunks)} chunks")
            points = build_points(chunks, filename)
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"    ✅  Uploaded {len(points)} points")
        except Exception as e:
            print(f"    ❌  Failed: {e}")
            logger.log(file_path, "ERROR", "A", str(e))
            continue

        if filename != extractable[-1]:
            print("    ⏳  Waiting 15 s...")
            time.sleep(15)

    # ── A4: Verify ────────────────────────────────────────────────────────
    print(f"\n[A4] Verifying chunk counts...\n")
    all_chunks, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=2000,
        with_payload=True,
        with_vectors=False,
    )
    counts = Counter(r.payload.get("document-id") for r in all_chunks)
    print(f"  {'FILE':<35}  {'CHUNKS':>6}  STATUS")
    print(f"  {'-' * 35}  {'-' * 6}  {'-' * 12}")
    for filename in DATA_PDFS:
        n = counts.get(filename, 0)
        flag = "✅" if n > 1 else ("⚠️  still 1" if n == 1 else "❌ 0")
        print(f"  {filename:<35}  {n:>6}  {flag}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B — ISO_Ai_helpdesk_Document folder
# ─────────────────────────────────────────────────────────────────────────────


def run_phase_b(logger: SkipLogger, already_clean: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  PHASE B — Re-ingesting ISO_Ai_helpdesk_Document/")
    print("=" * 60)

    # ── B1: Delete existing ISO chunks ───────────────────────────────────
    # Skipped when the full run already called delete_all_points() upfront.
    if already_clean:
        print("\n[B1] Skipping deletion — collection was already wiped.")
    else:
        print("\n[B1] Deleting all existing ISO folder chunks from Qdrant...\n")
        all_points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5000,
            with_payload=True,
            with_vectors=False,
        )
        data_pdf_names = set(DATA_PDFS)
        iso_ids = [
            p.id
            for p in all_points
            if p.payload.get("data_source") == "ISO_Ai_helpdesk_Document"
            or (
                p.payload.get("document-id") not in data_pdf_names
                and p.payload.get("category") != "data_folder"
            )
        ]
        if iso_ids:
            qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=iso_ids),
            )
            print(f"    🗑️  Deleted {len(iso_ids)} existing ISO chunks")
        else:
            print("    ℹ️  No existing ISO chunks found")

    # ── B2: Re-ingest all subfolders ──────────────────────────────────────
    print(f"\n[B2] Re-ingesting ISO folder contents...\n")

    entries = sorted(os.listdir(ISO_ROOT))
    subfolders = [
        e
        for e in entries
        if os.path.isdir(os.path.join(ISO_ROOT, e)) and not e.startswith(".")
    ]
    has_root_files = any(
        os.path.isfile(os.path.join(ISO_ROOT, e))
        and e.endswith((".pdf", ".txt", ".docx"))
        for e in entries
    )

    print(
        f"  {len(subfolders)} category folders + {'root files' if has_root_files else 'no root files'}\n"
    )

    # Root-level files
    if has_root_files:
        _ingest_folder_logged(ISO_ROOT, root_only=True, logger=logger)

    # Category subfolders
    for folder_name in subfolders:
        folder_path = os.path.join(ISO_ROOT, folder_name)

        # Log folders that have zero supported files
        file_count = sum(
            1
            for _, _, fnames in os.walk(folder_path)
            for fn in fnames
            if fn.endswith((".pdf", ".txt", ".docx")) and not fn.startswith(".")
        )
        if file_count == 0:
            print(f"\n  ⚠️  Skipping empty folder: {folder_name}")
            logger.log(
                folder_path, "NO_FILES", "B", "no .pdf / .txt / .docx files found"
            )
            continue

        _ingest_folder_logged(folder_path, root_only=False, logger=logger)

    # ── B3: Verify ────────────────────────────────────────────────────────
    print(f"\n[B3] Verifying ISO re-ingestion...\n")
    all_chunks, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=5000,
        with_payload=True,
        with_vectors=False,
    )
    iso_chunks = [
        r
        for r in all_chunks
        if r.payload.get("data_source") == "ISO_Ai_helpdesk_Document"
    ]
    counts = Counter(r.payload.get("category_clean") for r in iso_chunks)
    print(f"  Total ISO chunks in Qdrant: {len(iso_chunks)}\n")
    print(f"  {'CATEGORY':<35}  {'CHUNKS':>6}")
    print(f"  {'-' * 35}  {'-' * 6}")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {str(cat):<35}  {n:>6}")


def _ingest_folder_logged(
    folder_path: str, root_only: bool, logger: SkipLogger
) -> None:
    """
    Thin wrapper around iso_vectorize.ingest_folder that intercepts its
    existing SKIP_LOG writes and mirrors them into our SkipLogger too.

    Strategy: after ingest_folder runs, we read any NEW lines appended
    to iso_vectorize's skipped_files.log and parse them into the logger.
    """
    from vectorDB.iso_vectorize import SKIP_LOG

    # Record where the skip log ends BEFORE this folder runs
    prior_size = os.path.getsize(SKIP_LOG) if os.path.exists(SKIP_LOG) else 0

    ingest_folder(
        folder_path,
        ISO_ROOT,
        qdrant_client,
        dense_embedder,
        sparse_embedder,
        root_only=root_only,
    )

    # Read any new lines appended by ingest_folder and mirror into logger
    if os.path.exists(SKIP_LOG):
        with open(SKIP_LOG, encoding="utf-8") as f:
            f.seek(prior_size)
            new_lines = f.read().splitlines()

        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[SKIP_LARGE]"):
                path = line.split("]", 1)[1].strip().split(" (")[0]
                details = line.split("(")[1].rstrip(")") if "(" in line else ""
                logger.log(path, "TOO_LARGE", "B", details)
            elif line.startswith("[SKIP]"):
                path = line.split("]", 1)[1].strip()
                logger.log(path, "SCANNED", "B")
            elif line.startswith("[ERROR]"):
                rest = line.split("]", 1)[1].strip()
                path, _, details = rest.partition(": ")
                logger.log(path, "ERROR", "B", details)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def delete_all_points() -> None:
    """
    Wipe every point from the collection so both phases start from a
    completely clean slate. Called automatically when running both phases.
    """
    print("\n" + "=" * 60)
    print("  PRE-FLIGHT: Deleting ALL existing points from Qdrant")
    print("=" * 60)

    all_points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=False,
        with_vectors=False,
    )
    if not all_points:
        print("  ℹ️  Collection is already empty — nothing to delete.")
        return

    ids = [p.id for p in all_points]
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=ids),
    )
    print(f"  🗑️  Deleted {len(ids)} points — collection is now empty.\n")


def main() -> None:
    args = sys.argv[1:]

    if "--summary" in args:
        SkipLogger.print_log_file()
        return

    run_data = "--iso" not in args
    run_iso = "--data" not in args

    print("=" * 60)
    print("  FULL RE-INGESTION")
    print(
        f"  Phases: {'A (data/)  ' if run_data else ''}{'B (ISO/)' if run_iso else ''}"
    )
    print("=" * 60)

    logger = SkipLogger()

    # When both phases run together, wipe everything first for a clean slate.
    # Individual --data or --iso runs do their own targeted deletions instead.
    full_run = run_data and run_iso
    fresh = "--fresh" in args

    # Wipe the collection when:
    #   a) both phases run together (always start clean), OR
    #   b) --fresh flag is explicitly passed (e.g. --fresh --data)
    if full_run or fresh:
        delete_all_points()

    if run_data:
        run_phase_a(logger)

    if run_iso:
        # already_clean=True tells Phase B to skip its own B1 deletion step
        # because the collection was already wiped above.
        run_phase_b(logger, already_clean=(full_run or fresh))

    # Always print the skip summary at the end
    logger.summary()

    print("\n" + "=" * 60)
    print("✅  Re-ingestion complete!")
    print("\nNext steps:")
    print('  uv run python debug_search.py "What is the URL for the Pragati portal?"')
    print('  uv run python debug_search.py "How to recover IGRS password?"')
    print("  uv run python reingest_data.py --summary   ← view full skip log")
    print("=" * 60)


if __name__ == "__main__":
    main()
