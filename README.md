# 🧠 Oncology Agentic RAG

**An Intelligent, Evidence-Grounded Medical Retrieval-Augmented Generation System for Oncology Research & Clinical Decision Support.**

> [!IMPORTANT]
> **Clinical Research Disclaimer**: This system is developed strictly for research, educational exploration, and academic benchmarking. It is not certified for direct medical diagnosis, treatment planning, or autonomous clinical decision support. Always consult a qualified oncologist or physician for clinical care.

Oncology Agentic RAG integrates advanced query prep (**LAQA**), hybrid vector-keyword retrieval (**MRL FAISS + BM25**), cross-encoder reranking (**BAAI/bge-reranker-large**), specialized medical LLMs (**MedGemma**), automated response evaluation with retry loops, and an **Explainable AI (XAI)** layer.

---

## 🌟 Key Features

- 🏥 **Clinical Query Expansion (LAQA)** — Language-Aware Query Analyzer parses intent, medical category (treatment, diagnosis, symptoms, prognosis), cancer type, and expands short clinical queries into rich prompts.
- ⚡ **Dual Indexing & Matryoshka Embeddings (MRL)** — Supports instant switching between MRL 512-dim (minimal, high speed) and Full 768-dim vector stores without reindexing.
- 🔍 **Hybrid Retrieval & Reranking** — Fuses dense semantic vector search (Nomic Embed v1.5 MRL) with sparse BM25 keyword search, boosted by medical entity matching and section alignment, then reranked via CrossEncoder (`BAAI/bge-reranker-large`).
- 🩺 **MedGemma Generator** — Medical-specialized LLM produces evidence-backed, structured clinical answers.
- 🔁 **Evaluator-Driven Retry Loop** — Autonomous verification checks grounding score, retrieval quality, answer relevance, and hallucination risk (Low/Medium/High), automatically retrying up to 3 times if quality falls below thresholds.
- 💡 **XAI & Supporting Evidence Trace** — Generates step-by-step explainability traces and extracts exact supporting evidence sentences from retrieved documents.
- 💻 **Interactive CLI Dashboard (`backend/app.py`)** — Boxed 6-step terminal output with ANSI bold labels and pipeline timing.
- 🌐 **Modern Web Application (`backend/server.py` + `frontend/index.html`)** — Glassmorphism UI featuring live query intent chips, supporting evidence quote cards, interactive XAI drawer, radial confidence gauge, quality metrics subgrid, and history management.
- 📊 **Evaluation & Statistics Suite** — Benchmark evaluation script (`evaluation.py`) and Paired $t$-test statistical significance tool (`paired_t_test.py`).

---

## 🏗️ System Architecture & Pipeline Flow

```
[ User Medical Query ]
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│ 1. LAQA (Language-Aware Query Analyzer)                 │
│    • Intent & Category Parsing                          │
│    • Medical Entity & Metadata Extraction               │
│    • Query Expansion & Keyword Generation               │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Hybrid Retrieval Engine                              │
│    • Dense MRL FAISS Search (512-dim / 768-dim)          │
│    • Sparse BM25 Keyword Search                         │
│    • Adaptive Weight Fusion & Semantic Boosting        │
│    • CrossEncoder Reranking                             │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 3. MedGemma Generator & Evaluator Retry Loop            │
│    • Grounded Answer Generation                         │
│    • Evaluator Scoring (Grounding, Relevance, Risk)     │
│    • Auto-Retry Loop (Max 3 attempts if Score < 7)      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 4. XAI & Explainability Layer                           │
│    • Supporting Sentence Extraction                     │
│    • Step-by-Step Reasoning Trace                       │
│    • Confidence Score Calibration                       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
    [ Structured Output / Web UI / CLI Dashboard ]
```

---

## 🛠️ Project Structure

```
oncology-agentic-rag/
├── backend/
│   ├── app.py                      # Interactive CLI Dashboard & Pipeline Entrypoint
│   ├── server.py                   # Flask REST API Server & Web App Host
│   ├── index_data.py               # Data Indexing Pipeline (Dual Indexing & Safeguards)
│   ├── evaluation.py               # Benchmark Evaluation Runner
│   ├── metrics.py                  # Text Similarity & Quality Metric Evaluators
│   ├── paired_t_test.py            # Statistical Significance Test Suite
│   ├── settings.py                 # Core System Configuration & Dynamic Loading
│   ├── runtime_settings.json       # Dynamic Runtime Config State
│   │
│   ├── modules/                    # Core RAG Modular Pipeline
│   │   ├── agent/
│   │   │   ├── agent_controller.py # Agentic Retry Loop & Strategy Selection
│   │   │   ├── evaluator.py        # Response Quality & Hallucination Evaluator
│   │   │   ├── semantic_cache.py   # Disk-Cached Query Similarity Store
│   │   │   └── memory.py           # Context & Attempt Memory
│   │   ├── laqa/
│   │   │   └── laqa.py             # Language-Aware Query Analyzer
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py # FAISS + BM25 Hybrid Retriever
│   │   │   └── reranker.py         # CrossEncoder Document Reranker (BAAI/bge-reranker-large)
│   │   ├── chunking/
│   │   │   └── chunker.py          # Clinical Sentence-Sliding Window Chunker & Classifier
│   │   ├── embeddings/
│   │   │   └── mrl_embeddings.py   # MRL Dynamic Embedding Generator (Nomic v1.5)
│   │   ├── generator/
│   │   │   └── medgemma.py         # MedGemma LLM Generation Interface
│   │   ├── optimization/
│   │   │   └── response_optimizer.py # Deterministic Post-Generation Sanitization & Formatting
│   │   └── xai/
│   │       └── explain.py          # Explainable AI (XAI) Sentence Extractor
│   │
│   ├── database/                   # Persistent Stores
│   │   ├── mrl/                    # MRL-enabled Vector Store (512-dim FAISS + BM25)
│   │   ├── full/                   # Full-resolution Vector Store (768-dim FAISS + BM25)
│   │   └── semantic_cache.json     # Cached Query-Answer Pairs
│   │
│   ├── data/
│   │   └── oncology_docs/          # Source Medical PDF/Text Documents
│   ├── questions/
│   │   └── cleaned_output.json     # Benchmark Evaluation Dataset
│   └── report/                     # Metric Reports & Statistical Results
│
├── frontend/
│   └── index.html                  # Single-Page Web Application Interface
│
├── DUAL_INDEX_IMPLEMENTATION.md    # Technical Specs for Dual Database Architecture
└── ARCHITECTURE.md                 # In-Depth System Pipeline & Module Specification
```

---

## ⚡ Quick Setup & Installation

### Prerequisites
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 or higher (Conda environment recommended)
- **Ollama** (for local MedGemma LLM inference):
  - Download & Install: [ollama.ai](https://ollama.ai)
  - Pull model: `ollama pull medgemma`

### 1. Clone & Environment Setup
```powershell
# Clone workspace
git clone <repository-url>
cd oncology-agentic-rag

# Activate conda environment (or create venv)
conda create -n rag python=3.10 -y
conda activate rag
```

### 2. Install Dependencies
```powershell
cd backend
pip install -r requirements.txt
```

---

## 🗂️ Document Indexing

Build the vector stores from source documents in `backend/data/oncology_docs/`.

### Build Both Databases (Recommended Single-Pass):
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; $env:PYTHONNOUSERSITE=1; python index_data.py --build-both
```

### Build Specific Store:
```powershell
# Build MRL 512-dim store
python index_data.py --store-type mrl

# Build Full 768-dim store
python index_data.py --store-type full
```

> **Note on Safety:** The indexing pipeline includes thread capping for PyTorch and incremental disk checkpointing every 500 chunks to prevent system thermal throttling and power interrupts.

---

## 🚀 Running the Application

### Option 1: Interactive CLI Dashboard (`app.py`)
Run the terminal dashboard for direct query processing:
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; $env:PYTHONNOUSERSITE=1; python app.py
```
**Output Highlights:**
- Step 1: Query Expansion (LAQA Intent, Category, Keywords)
- Step 2: Top Retrieved Docs (Doc ID, Source File, Section, Scores, Snippet)
- Step 3: Agent Answer
- Step 4: Supporting Evidence Sentences
- Step 5: XAI Reasoning Trace
- Step 6: Quality Metrics (Confidence, Grounding, Retrieval, Hallucination Risk, Evaluator Status, Pipeline Latency)

### Option 2: REST Server & Web Application (`server.py`)
Launch the Flask backend server:
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; $env:PYTHONNOUSERSITE=1; python server.py
```
- **Web App**: Open `http://localhost:5000/app` or `http://localhost:5000/` in your browser.
- **REST API Endpoint**: `http://localhost:5000/query`

---

## 📡 REST API Reference

### 1. Submit Medical Query
`POST /query`
```json
// Request Body
{
  "query": "What are the targeted therapy options for EGFR mutated NSCLC?"
}
```

```json
// Response Body
{
  "answer": "Targeted therapy options for EGFR-mutated non-small cell lung cancer (NSCLC) include EGFR tyrosine kinase inhibitors (TKIs) such as Gefitinib, Erlotinib, Afatinib, and Osimertinib...",
  "confidence": 0.82,
  "reasoning": "1. Query identified as targeted therapy for EGFR NSCLC...\n2. Retrieved high-scoring clinical guidelines...",
  "supporting_sentences": [
    "Gefitinib, erlotinib, afatinib, osimertinib are used in histological variants of adenocarcinoma..."
  ],
  "grounded": true,
  "quality": "High",
  "sources": ["doc_3059", "doc_21189"],
  "source_texts": ["Full text snippet 1...", "Full text snippet 2..."],
  "evaluation": {
    "score": 9,
    "grounding_score": 0.92,
    "retrieval_score": 0.62,
    "answer_relevance": 1.0,
    "hallucination_risk": "low"
  },
  "query_analysis": {
    "intent": "clinical_guidance",
    "query_type": "treatment",
    "expanded_query": "what are the targeted therapy options for egfr mutated non-small cell lung cancer",
    "keywords": ["targeted", "therapy", "egfr", "nsclc"],
    "query_metadata": {
      "cancer_type": "lung cancer",
      "treatment_type": "targeted therapy"
    }
  },
  "metrics": {
    "laqa_time": 3.07,
    "rag_time": 0.06,
    "xai_time": 0.02,
    "total_time": 3.16
  }
}
```

### 2. Runtime Settings & Database Toggle
`GET /settings` — Get current active settings.  
`POST /settings/update` — Update settings dynamically:
```json
{
  "enable_laqa": true,
  "enable_mrl": true,
  "active_database": "mrl"
}
```

### 3. System Validation
`GET /system/validate-index` — Validates FAISS vector dimension and database metadata consistency.

---

## 📊 Evaluation & Benchmarking

Run the pipeline benchmark evaluation against `questions/cleaned_output.json`:
```powershell
python evaluation.py
```
Run paired statistical significance tests:
```powershell
python paired_t_test.py
```

---

## ⚙️ Configuration Parameters

Configuration is managed dynamically via `backend/settings.py` and saved state in `backend/runtime_settings.json`:
- `ENABLE_LAQA` — Enable/disable LAQA pre-retrieval query expansion.
- `ENABLE_MRL` — Toggle MRL 512-dim vs Full 768-dim mode.
- `RETRIEVAL_RELEVANCE_THRESHOLD` — Minimum relevance score for evidence.
- `GENERATOR_MODEL` — `medgemma` (via Ollama or HuggingFace).

---

## 📜 Technical Documentation

For detailed technical explanations, refer to:
- 📖 [ARCHITECTURE.md](file:///c:/Users/Sandhiya%20P/NIT%20INTERN/oncology-agentic-rag/ARCHITECTURE.md) — Module-by-module architectural deep dive.
- 📖 [DUAL_INDEX_IMPLEMENTATION.md](file:///c:/Users/Sandhiya%20P/NIT%20INTERN/oncology-agentic-rag/DUAL_INDEX_IMPLEMENTATION.md) — Technical specs for dual vector indexing.
