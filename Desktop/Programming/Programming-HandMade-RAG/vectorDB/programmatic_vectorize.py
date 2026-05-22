import os
import sys
import time
import uuid
import re
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

# ── Krutidev to Unicode Converter ──────────────────────────────────────────────
class KrutidevToUnicode:
    CHARS_KD = [
        "ñ", "Q+Z", "sas", "aa", ")Z", "ZZ", "‘", "’", "“", "”",
        "å", "ƒ", "„", "…", "†", "‡", "ˆ", "‰", "Š", "‹",
        "¶+", "d+", "[+k", "[+", "x+", "T+", "t+", "M+", "<+", "Q+", ";+", "j+", "u+",
        "Ùk", "Ù", "ä", "–", "—", "é", "™", "=kk", "f=k",
        "à", "á", "â", "ã", "ºz", "º", "í", "{k", "{", "=", "«",
        "Nî", "Vî", "Bî", "Mî", "<î", "|", "K", "}",
        "J", "Vª", "Mª", "<ªª", "Nª", "Ø", "Ý", "nzZ", "æ", "ç", "Á", "xz", "#", ":",
        "v‚", "vks", "vkS", "vk", "v", "b±", "Ã", "bZ", "b", "m", "Å", ",s", ",", "_",
        "ô", "d", "Dk", "D", "[k", "[", "x", "Xk", "X", "Ä", "?k", "?", "³",
        "pkS", "p", "Pk", "P", "N", "t", "Tk", "T", ">", "÷", "¥",
        "ê", "ë", "V", "B", "ì", "ï", "M+", "<+", "M", "<", ".k", ".",
        "r", "Rk", "R", "Fk", "F", ")", "n", "/k", "èk", "/", "Ë", "è", "u", "Uk", "U",
        "i", "Ik", "I", "Q", "¶", "c", "Ck", "C", "Hk", "H", "e", "Ek", "E",
        ";", "¸", "j", "y", "Yk", "Y", "G", "o", "Ok", "O",
        "'k", "'", "\"k", "\"", "l", "Lk", "L", "g",
        "È", "z",
        "Ì", "Í", "Î", "Ï", "Ñ", "Ò", "Ó", "Ô", "Ö", "Ø", "Ù", "Ük", "Ü",
        "‚", "ks", "kS", "k", "h", "q", "w", "`", "s", "S",
        "a", "¡", "%", "W", "•", "·", "∙", "·", "~j", "~", "\\", "+", " ः",
        "^", "*", "Þ", "ß", "(", "¼", "½", "¿", "À", "¾", "A", "-", "&", "&", "Œ", "]", "~ ", "@"
    ]

    CHARS_UNICODE = [
        "॰", "QZ+", "sa", "a", "र्द्ध", "Z", "\"", "\"", "'", "'",
        "०", "१", "२", "३", "४", "५", "६", "७", "८", "९",
        "फ़्", "क़", "ख़", "ख़्", "ग़", "ッズ", "ज़", "ड़", "ढ़", "फ़", "य़", "ऱ", "ऩ",
        "त्त", "त्त्", "क्त", "दृ", "कृ", "न्न", "न्न्", "=k", "f=",
        "ह्न", "ह्य", "हृ", "ह्म", "ह्र", "ह्", "द्द", "क्ष", "क्ष्", "त्र", "त्र्",
        "छ्य", "ट्य", "ठ्य", "ड्य", "ढ्य", "द्य", "ज्ञ", "द्व",
        "श्र", "ट्र", "ड्र", "ढ्र", "छ्र", "क्र", "फ्र", "र्द्र", "द्र", "प्र", "प्र", "ग्र", "रु", "रू",
        "ऑ", "ओ", "औ", "आ", "अ", "ईं", "ई", "ई", "इ", "उ", "ऊ", "ऐ", "ए", "ऋ",
        "क्क", "क", "क", "क्", "ख", "ख्", "ग", "ग", "ग्", "घ", "घ", "घ्", "ङ",
        "चै", "च", "च", "च्", "छ", "ज", "ज", "ज्", "झ", "झ्", "ञ",
        "ट्ट", "ट्ठ", "ट", "ठ", "ड्ड", "ड्ढ", "ड़", "ढ़", "ड", "ढ", "ण", "ण्",
        "त", "त", "त्", "थ", "थ्", "द्ध", "द", "ध", "ध", "ध्", "ध्", "ध्", "न", "न", "न्",
        "प", "प", "प्", "फ", "फ्", "ब", "ब", "ब्", "भ", "भ्", "म", "म", "म्",
        "य", "य्", "र", "ल", "ल", "ल्", "ळ", "व", "व", "व्",
        "श", "श्", "ष", "ष्", "स", "स", "स्", "ह",
        "ीं", "्र",
        "द्द", "ट्ट", "ट्ठ", "ड्ड", "कृ", "भ", "्य", "ड्ढ", "झ्", "क्र", "त्त्", "श", "श्",
        "ॉ", "ो", "ौ", "ा", "ी", "ु", "ू", "ृ", "े", "ै",
        "ं", "ँ", "ः", "ॅ", "ऽ", "ऽ", "ऽ", "ऽ", "्र", "्", "?", "़", ":",
        "‘", "’", "“", "”", ";", "(", ")", "{", "}", "=", "।", ".", "-", "µ", "॰", ",", "् ", "/"
    ]

    @staticmethod
    def convert_word(processPart: str) -> str:
        if not processPart:
            return ""

        for k, u in zip(KrutidevToUnicode.CHARS_KD, KrutidevToUnicode.CHARS_UNICODE):
            processPart = processPart.replace(k, u)

        # Code for Replacing five Special glyphs
        processPart = processPart.replace('±', "Zं")
        processPart = processPart.replace('Æ', "र्f")

        # f -> ि
        position_of_i = processPart.find('f')
        while position_of_i > -1:
            if position_of_i + 1 < len(processPart):
                charecter_next_to_i = processPart[position_of_i + 1]
                charecter_to_be_replaced = "f" + charecter_next_to_i
                processPart = processPart.replace(charecter_to_be_replaced, charecter_next_to_i + "ि")
            position_of_i = processPart.find('f', position_of_i + 1)

        processPart = processPart.replace('Ç', "fa")
        processPart = processPart.replace('É', "र्fa")

        # fa -> िं
        position_of_i = processPart.find('fa')
        while position_of_i > -1:
            if position_of_i + 2 < len(processPart):
                charecter_next_to_ip2 = processPart[position_of_i + 2]
                charecter_to_be_replaced = "fa" + charecter_next_to_ip2
                processPart = processPart.replace(charecter_to_be_replaced, charecter_next_to_ip2 + "िं")
            position_of_i = processPart.find('fa', position_of_i + 1)

        processPart = processPart.replace('Ê', "ीZ")

        # Eliminate 'chhotee ee kee maatraa' on half-letters
        position_of_wrong_ee = processPart.find("ि्")
        while position_of_wrong_ee > -1:
            if position_of_wrong_ee + 2 < len(processPart):
                consonent_next_to_wrong_ee = processPart[position_of_wrong_ee + 2]
                charecter_to_be_replaced = "ि्" + consonent_next_to_wrong_ee
                processPart = processPart.replace(charecter_to_be_replaced, "्" + consonent_next_to_wrong_ee + "ि")
            position_of_wrong_ee = processPart.find("ि्", position_of_wrong_ee + 2)

        # Eliminate reph "Z"
        set_of_matras = "अ आ इ ई उ ऊ ए ऐ ओ औ ा ि ी ु ू ृ े ै ो ौ ं : ँ ॅ"
        position_of_R = processPart.find("Z")
        while position_of_R > -1:
            probable_position_of_half_r = position_of_R - 1
            if probable_position_of_half_r >= 0:
                charecter_at_probable_position_of_half_r = processPart[probable_position_of_half_r]
                while probable_position_of_half_r >= 0 and charecter_at_probable_position_of_half_r in set_of_matras:
                    probable_position_of_half_r -= 1
                    if probable_position_of_half_r >= 0:
                        charecter_at_probable_position_of_half_r = processPart[probable_position_of_half_r]

                charecter_to_be_replaced = processPart[probable_position_of_half_r : position_of_R]
                new_replacement_string = "र्" + charecter_to_be_replaced
                charecter_to_be_replaced = charecter_to_be_replaced + "Z"
                processPart = processPart.replace(charecter_to_be_replaced, new_replacement_string)
            position_of_R = processPart.find("Z")

        return processPart

# Curated English whitelist
ENGLISH_WORDS_WHITELIST = {
    "bid", "bids", "capacity", "available", "value", "number", "years", "prescribed", "completion", 
    "work", "question", "updated", "current", "price", "level", "negotiation", "negotiations", 
    "tender", "tenders", "limited", "single", "lowest", "highest", "cost", "costing", "estimated", 
    "rate", "rates", "clause", "clauses", "manual", "procurement", "financial", "rules", "order", 
    "orders", "government", "state", "department", "arbitration", "portal", "website", "login", 
    "user", "password", "pdf", "docx", "excel", "sheet", "page", "date", "gst", "pwd", "nit", 
    "emd", "fdr", "bg", "boq", "for", "against", "three", "two", "one", "costing", "amount", 
    "equal", "normally", "there", "should", "no", "negotiation", "negotiations", "rare", "exception", 
    "only", "directly", "affected", "bidder", "can", "represent", "in", "this", "regard", "who", 
    "has", "participated", "the", "and", "of", "to", "is", "on", "with", "as", "at", "by", "an", 
    "be", "that", "from", "it", "not", "or", "are", "which", "shall", "will", "should", "would", 
    "may", "any", "all", "other", "such", "where", "etc", "normally", "be", "no", "rare", 
    "exception", "only", "a", "who", "has", "participated", "in", "this", "regard", "is", "it", 
    "possible", "sometimes", "group", "quote", "same", "against", "procuring", "entity", "reject", 
    "proposal", "however", "would", "advisable", "fix", "normative", "abnormally", "low", "one", 
    "combination", "other", "elements", "assessed", "reasonable", "unconditional", "complete", 
    "evaluation", "determination", "responsiveness", "committee", "scrutiny", "found", "such", 
    "aspects", "fully", "taken", "care", "contractor", "re-bidding", "costs", "firstly", "actual", 
    "retendering", "secondly", "delay", "execution", "inadequate", "competition", "non-availability", 
    "suitable", "quotations", "enlisted", "bidders", "urgent", "unsolicited", "offers", "limited", 
    "enquire", "ignored", "ministries", "similar", "completed", "works", "less", "than", "estimated",
    "suppression", "means", "company", "does", "submit", "final", "consideration", "determined",
    "solely", "basis", "even", "when", "submitted", "anti-competitive", "practices", "otherwise",
    "expected", "compete", "secretly", "conspire"
}

# Only strip standard visual punctuation wrapping words (not part of Kruti Dev letter mappings)
PUNCT_TO_STRIP = '()[]{}<>,.?!:;\'"`“”‘’A'

def convert_punctuation(punc: str) -> str:
    mapping = {
        ',': 'ए',
        ';': 'य',
        "'": 'श',
        '"': 'ष',
        ']': ',',
        '[': 'ख्',
        'A': '।',
        '-': '.',
        '}': 'द्व',
        '{': 'क्ष्',
        '|': 'द्य',
        '<': 'ढ',
        '>': 'झ',
        '~': '्',
        '@': '/',
        '\\': '?',
    }
    res = []
    for char in punc:
        res.append(mapping.get(char, char))
    return "".join(res)

def convert_token(token: str) -> str:
    # Find leading punctuation
    leading = ""
    for char in token:
        if char in PUNCT_TO_STRIP or char.isspace():
            leading += char
        else:
            break
            
    # Find trailing punctuation
    trailing = ""
    for char in reversed(token[len(leading):]):
        if char in PUNCT_TO_STRIP or char.isspace():
            trailing = char + trailing
        else:
            break
            
    # Extract inner word
    inner_word = token[len(leading) : len(token) - len(trailing)]
    if not inner_word:
        return convert_punctuation(token)
        
    # If the token contains no letters (e.g. punctuation, numbers), keep it as-is
    if not re.search(r'[a-zA-Z]', inner_word):
        return leading + inner_word + trailing
        
    # Check if inner word is English
    word_clean = re.sub(r'[^a-zA-Z]', '', inner_word)
    is_english = False
    if word_clean:
        is_all_caps_single = len(word_clean) == 1 and word_clean.isupper()
        is_whitelist_word = word_clean.lower() in ENGLISH_WORDS_WHITELIST
        if is_whitelist_word or is_all_caps_single:
            is_english = True
            
    if is_english:
        return leading + inner_word + trailing
    else:
        return convert_punctuation(leading) + KrutidevToUnicode.convert_word(inner_word) + convert_punctuation(trailing)

def convert_text(text: str) -> str:
    tokens = re.split(r'(\s+)', text)
    return "".join(convert_token(token) for token in tokens)

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
    return count >= 2

def is_text_extractable(docs) -> bool:
    total_text = "".join(d.page_content for d in docs).strip()
    if len(total_text) <= 50:
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
    from google import genai
    from google.genai import types
    client = genai.Client()
    
    batch_size = 50
    for i in range(0, len(enriched_texts), batch_size):
        batch = enriched_texts[i : i + batch_size]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
        config = types.EmbedContentConfig(output_dimensionality=1536)
        try:
            res = client.models.embed_content(
                model="models/gemini-embedding-2",
                contents=contents,
                config=config,
            )
            batch_dense_vecs = [list(e.values) for e in res.embeddings]
            dense_vecs.extend(batch_dense_vecs)
            time.sleep(0.5)
        except Exception as e:
            error_str = str(e)
            if any(err in error_str for err in ["429", "RESOURCE_EXHAUSTED", "503", "500", "UNAVAILABLE"]):
                print(f"      ⚠️  API Error ({error_str.split()[0]}) — sleeping 60 s then retrying...")
                time.sleep(60)
                print("      🔄 Retrying batch...")
                try:
                    res = client.models.embed_content(
                        model="models/gemini-embedding-2",
                        contents=contents,
                        config=config,
                    )
                    batch_dense_vecs = [list(e.values) for e in res.embeddings]
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
    print(f"  🗑  Deleting ALL existing points from '{COLLECTION_NAME}'")
    print("=" * 60)
    
    if not client.collection_exists(COLLECTION_NAME):
        print(f"  🛠  Collection '{COLLECTION_NAME}' not found. It will be created.")
        return

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
        print(f"  🗑  Deleted batch of {len(ids)} points...")

    print("  ✅ Collection cleared completely.\n")

# ── Ingest Ansari folder ───────────────────────────────────────────────────────
def ingest_ansari_folder(qdrant_client, dense_embedder, sparse_embedder):
    print("\n" + "=" * 60)
    print(f"  📂 Ingesting folder programmatically: {ANSARI_ROOT}")
    print("=" * 60)

    all_files = []
    
    # Allow passing a single file path as argument for targeted re-ingestion
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
        print(f"  ⚠  No supported files found in ansari AI folder.\n")
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
                converted_programmatically = False

                # Programmatic font conversion check for Kruti Dev PDFs
                if ext == ".pdf" and docs:
                    # Sample text from first few pages to detect legacy encoding
                    sample_text = "".join(d.page_content for d in docs[:5])[:2000]
                    if is_krutidev_gibberish(sample_text):
                        print("    ⚡ Detected Kruti Dev gibberish encoding. Running programmatic font conversion...")
                        for doc in docs:
                            doc.page_content = convert_text(doc.page_content)
                        converted_programmatically = True
                        print("    ✅ Programmatic font conversion complete!")

                # If after programmatic conversion or otherwise it's still empty, try OCR
                if not docs or (not converted_programmatically and not is_text_extractable(docs)):
                    if ext == ".pdf":
                        print(f"    - No extractable text. Attempting OCR fallback...")
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

    # Clear existing database points only if running a full ingestion
    if len(sys.argv) <= 1:
        clear_qdrant_collection(qdrant_client)
    else:
        print("⏭️  Skipping collection clear because this is a targeted single-file ingestion.")

    # Ingest ansari AI folder programmatically
    ingest_ansari_folder(qdrant_client, dense_embedder, sparse_embedder)

    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"📊 Total vectors now in '{COLLECTION_NAME}': {info.points_count:,}\n")
    print("✅ Ansari AI ingestion complete!")
