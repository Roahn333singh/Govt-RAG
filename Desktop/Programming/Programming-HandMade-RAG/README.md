# 🏛️ Govt-RAG — Smart AI Helpdesk for UP Government Manuals

A production-grade **Retrieval-Augmented Generation (RAG)** application that answers questions about Uttar Pradesh Government policies, departments, portals, and regulations — in both **Hindi** and **English**.

Built from scratch with a handmade RAG pipeline (no frameworks like LlamaIndex), featuring hybrid search, cross-encoder re-ranking, semantic caching, and streaming responses.

---

## ✨ Key Features

- **Bilingual Support** — Answers in Hindi or English based on the user's query language
- **Hybrid Search** — Dense (Gemini Embeddings) + Sparse (BM25) retrieval with RRF fusion
- **Cross-Encoder Re-ranking** — `ms-marco-MiniLM-L-6-v2` for precise relevance scoring
- **Semantic Query Cache** — Qdrant-based cache that returns instant answers for similar questions (~10x speedup)
- **LangGraph Agent Pipeline** — Query Rewriter → Retriever → Generator (3-node graph)
- **Streaming API** — Real-time token streaming via SSE for a responsive chat experience
- **Rate Limiting** — SlowAPI-based protection (15 requests/minute per IP)
- **Off-topic Guardrails** — Rejects queries unrelated to government data

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Query Rewriter     │  ← Gemini 2.5 Flash (temp=0.0)
│   (Bilingual keyword │     Converts conversational queries into
│    optimization)     │     keyword-dense bilingual search queries
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Hybrid Retriever   │  ← Dense: Gemini Embedding-2 (1536d)
│   + Cross-Encoder    │     Sparse: Qdrant/BM25
│     Re-ranker        │     Reranker: ms-marco-MiniLM-L-6-v2
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Generator          │  ← Gemini 2.5 Flash (temp=0.1)
│   (Factual, strict   │     Strict subject-attribution rules
│    bilingual answer) │     for government accuracy
└─────────────────────┘
```

**Vector Database:** [Qdrant Cloud](https://cloud.qdrant.io/) with two collections:
- `my-collection` — Document chunks (dense + sparse vectors)
- `semantic-query-cache` — Cached Q&A pairs for instant retrieval

---

## 📁 Project Structure

```
├── main.py                    # FastAPI server (chat + streaming endpoints)
├── agent/
│   ├── __init__.py
│   └── lang_graph.py          # LangGraph pipeline (rewriter → retriever → generator)
├── vectorDB/
│   ├── __init__.py
│   ├── search.py              # Hybrid search + cross-encoder re-ranking
│   ├── vectorize.py           # Base vectorizer for PDF ingestion
│   ├── iso_vectorize.py       # ISO document vectorizer
│   ├── nivida_vectorize.py    # Nivida (procurement) document vectorizer
│   ├── ansari_vectorize.py    # Ansari document vectorizer
│   ├── vittpath_vectorize.py  # Vittpath (finance) document vectorizer
│   ├── programmatic_vectorize.py  # Programmatic batch vectorizer
│   └── test.py                # Vector DB quick tests
├── govt_manual_ui/            # Frontend UI
│   ├── index.html             # Main HTML (dashboard layout)
│   ├── script.js              # Chat logic, streaming, markdown rendering
│   └── style.css              # Glassmorphism dark theme styling
├── frontend.py                # Alternative Streamlit frontend
├── i055.site/fmisc/           # Embeddable chat widget
│   ├── index.html             # Widget demo page
│   └── widget.js              # Standalone chat widget (can embed on any site)
├── Nivida docs/               # Parsed Nivida procurement manual (Hindi markdown)
├── vittpath2021 docs/         # Parsed Vittpath finance manual (Hindi markdown)
├── reingest_data.py           # Full data re-ingestion pipeline
├── pyproject.toml             # Python dependencies (uv/pip)
├── uv.lock                    # Lockfile for reproducible installs
├── .env.example               # Environment variable template
├── evaluation_qa.md           # Evaluation Q&A (portal, IGRS, ISO docs)
├── nivida_evaluation_qa.md    # Evaluation Q&A (Nivida procurement docs)
├── chunked_docs_inventory.csv # Inventory of all chunked documents
│
├── # Utility Scripts
├── check_cluster.py           # Inspect Qdrant cluster state
├── clear_qdrant_cache.py      # Clear the semantic query cache
├── delete_chapter_13.py       # Remove specific chapter from vector DB
├── delete_specific_file.py    # Remove specific file from vector DB
├── remove_duplicates.py       # Deduplicate vectors in Qdrant
├── debug_search.py            # Debug hybrid search results
│
├── # Test Scripts
├── test_latency.py            # Cache hit vs miss latency benchmark
├── test_load.py               # Concurrent user load test (3 threads)
└── test_rewriter.py           # Test the bilingual query rewriter
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`
- A **Qdrant Cloud** account (free tier works)
- A **Google AI Studio** API key (for Gemini)

### 1. Clone the Repository

```bash
git clone https://github.com/Roahn333singh/Govt-RAG.git
cd Govt-RAG/Desktop/Programming/Programming-HandMade-RAG
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
QDRANT_API_KEY=your_qdrant_api_key_here
CLUSTER_ENDPOINT=https://your-cluster-id.region.aws.cloud.qdrant.io
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Install Dependencies

Using **uv** (recommended):
```bash
uv sync
```

Or using **pip**:
```bash
pip install -e .
```

### 4. Ingest Data into Qdrant (First Time Only)

If your Qdrant collection is empty, ingest the PDF documents:

```bash
uv run python reingest_data.py
```

This reads PDFs from `data/`, chunks them, generates dense + sparse embeddings, and uploads to Qdrant Cloud.

### 5. Run the Application

Start the FastAPI backend:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will:
1. Connect to Qdrant Cloud
2. Load embedding models (dense, sparse, and reranker)
3. Start listening on `http://localhost:8000`

### 6. Open the Frontend

Open the frontend UI in your browser:

```
govt_manual_ui/index.html
```

Or use the Streamlit frontend:

```bash
uv run streamlit run frontend.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/chat` | Send a question, get a complete answer |
| `POST` | `/chat/stream` | Send a question, get a streaming (SSE) answer |

### Example Request

```bash
# Simple chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the Pragati Portal?"}'

# Streaming chat
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "ISO के उद्देश्य क्या हैं?"}'
```

---

## 🧪 Testing

Make sure the FastAPI server is running (`uvicorn main:app ...`) before running tests.

### Latency Benchmark (Cache Hit vs Miss)

Tests the semantic cache by sending two queries — one fresh (cache miss) and one semantically similar (cache hit):

```bash
uv run python test_latency.py
```

**Expected output:**
```
📊 --- FINAL LATENCY REPORT ---
Original Lag (Reading PDFs + Gemini): ~8.00 seconds
Cached Lag   (Qdrant Semantic Math):  ~0.50 seconds
⚡ Your cache made the app 16x Faster!
```

### Load Test (Concurrent Users)

Simulates 3 users hitting the server simultaneously to verify async/non-blocking behavior:

```bash
uv run python test_load.py
```

**Expected:** All 3 responses return in ~8s total (not 24s), proving the server handles concurrent requests.

### Query Rewriter Test

Tests the bilingual query optimization node in isolation:

```bash
uv run python test_rewriter.py
```

**Expected:** Converts `"Bid Capacity कैसे निर्धारित की जाती है?"` into a keyword-dense bilingual search query.

### Evaluation Q&A

The files `evaluation_qa.md` and `nivida_evaluation_qa.md` contain curated question-answer pairs derived from the source documents. Use these to manually evaluate retrieval accuracy and generation quality:

- **`evaluation_qa.md`** — Covers portal URLs, IGRS system, ISO objectives, Computer Centre, CMIS
- **`nivida_evaluation_qa.md`** — Covers Nivida (procurement) rules, tender committees, bid capacity, e-tendering

---

## 🛠️ Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `check_cluster.py` | Inspect Qdrant collection stats | `uv run python check_cluster.py` |
| `clear_qdrant_cache.py` | Clear the semantic query cache | `uv run python clear_qdrant_cache.py` |
| `delete_chapter_13.py` | Remove a specific chapter's vectors | `uv run python delete_chapter_13.py` |
| `delete_specific_file.py` | Remove vectors for a specific file | `uv run python delete_specific_file.py` |
| `remove_duplicates.py` | Deduplicate vectors in Qdrant | `uv run python remove_duplicates.py` |
| `debug_search.py` | Debug hybrid search results | `uv run python debug_search.py` |

---

## 📚 Document Sources

The RAG system is trained on these UP Government documents:

| Document | Description |
|----------|-------------|
| WaterSectorPortals.pdf | Water sector portal URLs and descriptions |
| IGRSFAQ.pdf | Jan-Sunwai (IGRS) system FAQ |
| ISO.pdf | Information Systems Organisation objectives |
| CMISFAQ.pdf | CMIS system FAQ |
| ComputerCentre.pdf | Computer Centre functions |
| FMISC.pdf | FMISC portal details |
| UPSWIC.pdf | UP State Water Informatics Centre |
| Allportaldetail.pdf | All portal details |
| Nivida Path (procurement) | Tender/procurement rules (Hindi) |
| Vittpath 2021 (finance) | Financial rules and procedures (Hindi) |

Pre-parsed markdown versions of the Hindi documents are available in `Nivida docs/` and `vittpath2021 docs/`.

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI + Uvicorn |
| **Agent Framework** | LangGraph (StateGraph) |
| **LLM** | Google Gemini 2.5 Flash |
| **Dense Embeddings** | Gemini Embedding-2 (1536d) |
| **Sparse Embeddings** | Qdrant/BM25 (FastEmbed) |
| **Re-ranker** | ms-marco-MiniLM-L-6-v2 (FastEmbed) |
| **Vector Database** | Qdrant Cloud |
| **Frontend** | Vanilla HTML/CSS/JS + Streamlit |
| **Rate Limiting** | SlowAPI |
| **Package Manager** | uv |
