import os
import sys
import time
import uuid
import re
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import LangDetectException, detect
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
)

load_dotenv(override=True)

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("CLUSTER_ENDPOINT", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = "my-collection"

DENSE_VECTOR_NAME = "abstract-dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"

VITTPATH_DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vittpath2021 docs"))

# ── Language Detection ─────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    try:
        if len(text.strip()) < 20:
            return "hi"  # default to Hindi since most of the documents are in Hindi
        return detect(text)
    except LangDetectException:
        return "hi"

# ── Detailed Splitter ──────────────────────────────────────────────────────────
def split_text_detailed(text: str) -> list:
    lines = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for line in lines:
        if line.startswith("#"):
            continue
        # Filter out very short lines like "or", "and", or just page numbers like "(94)"
        val = line.strip()
        val_clean = val.strip("*").strip("(").strip(")").strip()
        if len(val_clean) < 10 and (val_clean.lower() == "or" or val_clean.lower() == "and" or val_clean.isdigit()):
            continue
            
        if line.startswith("*") or line.startswith("-") or re.match(r"^\(\w\)", line) or re.match(r"^\d+\.", line):
            chunks.append(line)
        else:
            sentences = re.split(r"(?<=[।.!?])\s+", line)
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current_chunk) + len(sentence) < 250:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk)
    return chunks

# ── Parser ─────────────────────────────────────────────────────────────────────
def parse_vittpath_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Get overall title (first H1)
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else os.path.basename(filepath)

    # Split by "Page X of Y" or "Content from Page Index Reference X" headers
    page_splits = re.split(
        r"\n(?:##\s+)?((?:Page\s+\d+\s+of\s+\d+|Content from Page Index Reference\s+\d+))\s*\n",
        content,
        flags=re.IGNORECASE
    )
    
    pages = []
    if len(page_splits) == 1:
        pages.append({
            "header": "General Content",
            "content": content,
            "page_num": 1
        })
    else:
        # First section of text before the first page marker is Page 1 content (or Introduction)
        first_body = page_splits[0].strip()
        pages.append({
            "header": "Page 1 Content",
            "content": first_body,
            "page_num": 1
        })
            
        for i in range(1, len(page_splits), 2):
            header = page_splits[i].strip()
            page_num_match = re.search(r"\d+", header)
            val = int(page_num_match.group()) if page_num_match else 1
            
            # If the marker is "Page Index Reference X", X corresponds to 0-indexed page,
            # so the 1-indexed page is X + 1. If it is "Page X of Y", the body after it
            # corresponds to page X + 1. So page_num is always val + 1.
            page_num = val + 1
            
            body = page_splits[i+1].strip() if i+1 < len(page_splits) else ""
            pages.append({
                "header": f"Page {page_num} Content",
                "content": body,
                "page_num": page_num
            })

    # Q&A split patterns: Match bold questions like **1. Q: ...** or **1. प्रश्न: ...** or **Q1:** etc.
    q_patterns = [
        r"\n\s*(?:\*\*\d+\.\s*Q\s*:\s*\*?|\*\*Q\d+\s*:\s*\*?|\d+\.\s*Q\s*:\s*|\*\*प्रश्न\s*\d*\s*:\s*\*?|\*\*Question\s*\d*\s*:\s*\*?)",
        r"\n\s*(?:\*\*प्रश्न\s*\d+:\*\*|\*\*Question\s*\d+:\*\*|\*\*Question:\*\*)"
    ]
    q_regex = "|".join(q_patterns)

    ans_patterns = [
        r"(?:\*\*A\s*:\s*\*?|\*\*Answer\s*:\s*\*?|\*\*उत्तर\s*:\s*\*?)",
        r"(?:\*\*उत्तर\s*\d+:\*\*|\*\*Answer\s*\d+:\*\*)"
    ]
    ans_regex = "|".join(ans_patterns)

    parsed_pages = []

    for page in pages:
        body = page["content"]
        page_num = page["page_num"]
        header = page["header"]

        # Search for QA divider
        qa_divider_idx = -1
        qa_headers = [
            "#### संभावी प्रश्नोत्तर",
            "## संभावी प्रश्नोत्तर",
            "### संभावी प्रश्नोत्तर",
            "संभावी प्रश्नोत्तर",
            "#### Contextual Q&A for RAG",
            "## Contextual Q&A for RAG"
        ]
        
        for qah in qa_headers:
            idx = body.find(qah)
            if idx != -1:
                # Find start of line containing the header to avoid leaving a trailing "## "
                line_start = body.rfind("\n", 0, idx)
                if line_start != -1:
                    qa_divider_idx = line_start
                else:
                    qa_divider_idx = idx
                break

        if qa_divider_idx != -1:
            main_text = body[:qa_divider_idx].strip()
            qa_text = body[qa_divider_idx:].strip()
        else:
            main_text = body.strip()
            qa_text = ""

        # Clean trailing page number markers or markdown symbols
        main_text_cleaned = re.sub(r"\(?\d+\)?\s*$", "", main_text).strip()

        # Parse Q&As first
        page_qas = []
        if qa_text:
            qa_text_adjusted = "\n" + qa_text
            matches = list(re.finditer(q_regex, qa_text_adjusted, re.IGNORECASE))
            
            for j in range(len(matches)):
                start_idx = matches[j].start()
                end_idx = matches[j+1].start() if j + 1 < len(matches) else len(qa_text_adjusted)
                
                qa_block = qa_text_adjusted[start_idx:end_idx].strip()
                ans_match = re.search(ans_regex, qa_block, re.IGNORECASE)
                if ans_match:
                    q_full = qa_block[:ans_match.start()].strip()
                    # Strip bold markers and numbering from the question text
                    q_clean = re.sub(r"^(?:\*\*\d+\.\s*Q\s*:\s*\*?|\*\*Q\d+\s*:\s*\*?|\d+\.\s*Q\s*:\s*|\*\*प्रश्न\s*\d*\s*:\s*\*?|\*\*Question\s*\d*\s*:\s*\*?)\s*", "", q_full, flags=re.IGNORECASE).strip()
                    q_clean = q_clean.strip("*").strip()
                    
                    a_full = qa_block[ans_match.end():].strip()
                    a_clean = re.sub(r"\(?\d+\)?\s*$", "", a_full).strip()
                    a_clean = a_clean.strip("*").strip()
                    
                    page_qas.append({
                        "question": q_clean,
                        "answer": a_clean
                    })
                else:
                    page_qas.append({
                        "question": qa_block,
                        "answer": ""
                    })

        # Split main_text if it exceeds a maximum size (e.g., 8000 characters)
        max_chars = 8000
        if len(main_text_cleaned) <= max_chars:
            parsed_pages.append({
                "title": title,
                "header": header,
                "page_num": page_num,
                "main_text": main_text_cleaned,
                "qas": page_qas
            })
        else:
            # Split main_text_cleaned by paragraphs to create virtual pages
            paragraphs = main_text_cleaned.split("\n\n")
            current_text = ""
            part_idx = 1
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current_text) + len(para) + 2 > max_chars:
                    if current_text:
                        parsed_pages.append({
                            "title": title,
                            "header": f"{header} (Part {part_idx})",
                            "page_num": page_num,
                            "main_text": current_text.strip(),
                            "qas": []
                        })
                        part_idx += 1
                        current_text = para
                    else:
                        current_text = para
                else:
                    if current_text:
                        current_text += "\n\n" + para
                    else:
                        current_text = para
                        
            if current_text:
                parsed_pages.append({
                    "title": title,
                    "header": f"{header} (Part {part_idx})",
                    "page_num": page_num,
                    "main_text": current_text.strip(),
                    "qas": page_qas  # Attach all Q&As to the last part
                })
            else:
                if parsed_pages:
                    parsed_pages[-1]["qas"] = page_qas
                else:
                    parsed_pages.append({
                        "title": title,
                        "header": header,
                        "page_num": page_num,
                        "main_text": "",
                        "qas": page_qas
                    })

    return parsed_pages

def build_points_for_file(filepath: str, parsed_pages: list, dense_embedder, sparse_embedder) -> list:
    filename = os.path.basename(filepath)
    points = []
    
    # Lists to batch embed at once
    child_texts_to_embed = []
    payloads = []

    for page in parsed_pages:
        title = page["title"]
        header = page["header"]
        page_num = page["page_num"]
        main_text = page["main_text"]
        qas = page["qas"]

        # Pre-construct parent context
        parent_context = f"# {title}\n## {header}\n\n{main_text}"
        
        # Detected language of the page content
        lang = detect_language(main_text if main_text else title)

        # 1. Process Page Main Content
        if main_text.strip():
            chunks = split_text_detailed(main_text)
            for chunk_idx, chunk_text in enumerate(chunks):
                child_text = chunk_text.strip()
                
                # Enriched text for the RAG prompt / cross-encoder
                payload_text = f"[Source: {title} | Page {page_num}]\n{child_text}\n\n[Full Context]:\n{parent_context}"
                
                metadata = {
                    "category": "General",
                    "category_clean": "General",
                    "sub_folder": "",
                    "data_source": "vittpath_markdown",
                    "file_type": "md",
                    "ingestion_date": str(date.today()),
                    "document-id": "vittpath 2021.pdf",
                    "source": filepath,
                    "page": page_num,
                    "language": lang,
                    "text": payload_text
                }
                
                # Enriched text for the embedding model (adds topic context)
                embed_text = f"Title: {title}\nTopic: {header}\nContent: {child_text}"
                child_texts_to_embed.append(embed_text)
                payloads.append(metadata)

        # 2. Process Page Q&As
        for qa in qas:
            q = qa["question"]
            a = qa["answer"]
            if not q:
                continue

            # Child text is Question + Answer to allow keyword search on both
            child_text = f"प्रश्न: {q}\nउत्तर: {a}"
            
            # Enriched text for the RAG prompt / cross-encoder
            payload_text = f"[Q&A - Source: {title} | Page {page_num}]\n**प्रश्न:** {q}\n**उत्तर:** {a}\n\n[Full Context]:\n{parent_context}"
            
            metadata = {
                "category": "General",
                "category_clean": "General",
                "sub_folder": "",
                "data_source": "vittpath_markdown",
                "file_type": "md",
                "ingestion_date": str(date.today()),
                "document-id": "vittpath 2021.pdf",
                "source": filepath,
                "page": page_num,
                "language": lang,
                "text": payload_text
            }
            
            # Enriched text for the embedding model (adds topic context)
            embed_text = f"Title: {title}\nTopic: {header}\nQ&A:\n{child_text}"
            child_texts_to_embed.append(embed_text)
            payloads.append(metadata)

    if not child_texts_to_embed:
        return []

    print(f"    - Generating embeddings for {len(child_texts_to_embed)} child chunks...")

    # Embed dense vectors in batches
    dense_vecs = []
    batch_size = 10
    for idx in range(0, len(child_texts_to_embed), batch_size):
        batch = child_texts_to_embed[idx : idx + batch_size]
        
        # Exponential backoff for rate limiting
        retries = 3
        delay = 2
        for r in range(retries):
            try:
                batch_embeddings = dense_embedder.embed_documents(batch)
                if len(batch_embeddings) != len(batch):
                    batch_embeddings = [
                        dense_embedder.embed_documents([text])[0] for text in batch
                    ]
                dense_vecs.extend(batch_embeddings)
                time.sleep(1)
                break
            except Exception as e:
                print(f"      ⚠️ Embedding batch failed (attempt {r+1}/{retries}): {e}")
                if r == retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2

    # Embed sparse vectors
    sparse_vecs = list(sparse_embedder.embed(child_texts_to_embed))

    if len(dense_vecs) != len(child_texts_to_embed) or len(sparse_vecs) != len(child_texts_to_embed):
        raise RuntimeError(f"Embedding mismatch in {filename}: expected={len(child_texts_to_embed)}, dense={len(dense_vecs)}, sparse={len(sparse_vecs)}")

    for i in range(len(child_texts_to_embed)):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                DENSE_VECTOR_NAME: dense_vecs[i],
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse_vecs[i].indices.tolist(),
                    values=sparse_vecs[i].values.tolist(),
                ),
            },
            payload=payloads[i]
        )
        points.append(point)

    return points

def main():
    if not os.path.exists(VITTPATH_DOCS_DIR):
        print(f"\n❌ ERROR: Vittpath docs directory not found at: {VITTPATH_DOCS_DIR}")
        sys.exit(1)

    print("\n🔌 Connecting to Qdrant Cloud and loading embedders...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120.0)
    dense_embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=1536,
    )
    sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("✅ Embedders ready!\n")

    # Check if a single target file is specified as command line argument
    md_files = []
    if len(sys.argv) > 1:
        target_arg = sys.argv[1]
        # Resolve path
        if os.path.isabs(target_arg):
            target_path = target_arg
        else:
            target_path = os.path.abspath(os.path.join(VITTPATH_DOCS_DIR, target_arg))
            
        if os.path.exists(target_path):
            md_files.append(target_path)
            print(f"🎯 Targeted ingestion for single file: {target_path}")
        else:
            # Try matching basename
            matched = False
            for f in os.listdir(VITTPATH_DOCS_DIR):
                if f == target_arg or f == os.path.basename(target_arg):
                    md_files.append(os.path.join(VITTPATH_DOCS_DIR, f))
                    print(f"🎯 Targeted ingestion matched: {md_files[-1]}")
                    matched = True
                    break
            if not matched:
                print(f"❌ Error: Targeted file not found: {target_arg}")
                sys.exit(1)
    else:
        # Get list of markdown files
        md_files = [os.path.join(VITTPATH_DOCS_DIR, f) for f in os.listdir(VITTPATH_DOCS_DIR) if f.endswith(".md")]
        md_files.sort()

    print(f"Found {len(md_files)} markdown documents to process.")

    total_chunks = 0
    start_time = time.time()

    for idx, filepath in enumerate(md_files, 1):
        fname = os.path.basename(filepath)
        print(f"\n[{idx}/{len(md_files)}] Processing file: {fname}")
        
        try:
            # 1. Parse pages and Q&As
            parsed_pages = parse_vittpath_file(filepath)
            
            # 2. Build points (embedding occurs here)
            points = build_points_for_file(filepath, parsed_pages, dense_embedder, sparse_embedder)
            
            if not points:
                print(f"    ⏭  Skipped: No content parsed in {fname}")
                continue
                
            # 3. Upload in batches to Qdrant
            print(f"    - Uploading {len(points)} vectors to Qdrant collection '{COLLECTION_NAME}' in batches of 50...")
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch_points = points[i : i + batch_size]
                retries = 5
                delay = 2
                for r in range(retries):
                    try:
                        client.upsert(collection_name=COLLECTION_NAME, points=batch_points)
                        break
                    except Exception as e:
                        print(f"      ⚠️ Upload batch {i//batch_size + 1} failed (attempt {r+1}/{retries}): {e}")
                        if r == retries - 1:
                            raise e
                        time.sleep(delay)
                        delay *= 2
            
            total_chunks += len(points)
            print(f"    ✅ Successfully ingested {len(points)} vectors for {fname}")
            
        except Exception as e:
            print(f"    ❌ Error processing {fname}: {e}")

    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("  🚀 VITTPATH MARKDOWN INGESTION RUN COMPLETE")
    print("=" * 60)
    print(f"  Total Chunks/Vectors Uploaded: {total_chunks:,}")
    print(f"  Time Elapsed: {duration/60:.2f} minutes")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
