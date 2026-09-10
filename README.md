# 🧠 Oncology Agentic RAG

**A Clinical-Grade, Evidence-Grounded Medical Retrieval-Augmented Generation System for Precision Oncology Research & Clinical Decision Support.**

> [!IMPORTANT]
> **Clinical Research Disclaimer**: This system is engineered strictly for research, educational exploration, and academic benchmarking. It is not certified for direct medical diagnosis, autonomous treatment planning, or emergency clinical intervention. Always consult a board-certified oncologist or licensed physician for patient care.

Oncology Agentic RAG integrates advanced clinical query analysis (**LAQA**), hybrid vector-keyword retrieval (**MRL FAISS + BM25**), cross-encoder neural reranking (**BAAI/bge-reranker-large**), specialized medical LLMs (**MedGemma**), automated response evaluation with autonomous state-machine retry loops, deterministic clinical negation/contraindication detection, enterprise security & HIPAA Safe Harbor de-identification, and an **Explainable AI (XAI)** layer with calibrated confidence scores.

---

## 🌟 Key Capabilities & Production Features

- 🏥 **Clinical Query Expansion (LAQA)** — Language-Aware Query Analyzer parses clinical intent (`factual`, `clinical_guidance`, `analytical`, `research`, `outcome_analysis`), extracts cancer types, therapy lines, and biomarker entities (`KRAS G12C`, `BRAF V600E`, `MSI-H`, `ADC`, `EGFR`, `PD-L1`), expanding shorthand queries into domain-enriched prompts with fast-fail safety timeouts.
- ⚡ **Dual Indexing & Matryoshka Embeddings (MRL)** — Dynamically toggles between high-speed 512-dimensional MRL vectors (`database/mrl`) and full 768-dimensional vectors (`database/full`) with zero reindexing required.
- 🔍 **Hybrid Retrieval, Adaptive Fusion & Reranking** — Fuses dense semantic vector search (`nomic-ai/nomic-embed-text-v1.5`) with sparse BM25 keyword matching using intent-specific weights, boosts documents for medical entity and section alignment, penalizes statistical noise, and reranks via a CrossEncoder (`BAAI/bge-reranker-large`).
- 🩺 **MedGemma Generator & Context Compression** — Specialised medical LLM synthesizes evidence-backed answers with strict clinical isolation. Context compression packs evidence strictly within token budgets without truncating mid-sentence.
- 🔁 **Evaluator-Driven State Machine Retry Loop** — Evaluates answers across Grounding Score, Retrieval Quality, Answer Relevance, and Hallucination Risk (Low/Medium/High). Automatically adjusts retrieval strategy (escalating $k$ or query expansion) up to 3 attempts, aborting early if query drift is detected.
- 🛡️ **Clinical Negation & Contraindication Guardrails** — Deterministic clinical regex engine checks pairwise assertions between generated answers and retrieved context, flagging contraindications, black-box warnings, and refractory disease misstatements.
- 🔒 **HIPAA Safe Harbor PHI Redaction & Security** — Automatic redaction of 18 HIPAA Safe Harbor identifiers (MRNs, patient names, dates, phone numbers, SSNs) across queries, logs, and caches. Built-in token authentication (RBAC), prompt-injection sanitization, and sliding-window rate limiting.
- ⚡ **Semantic Vector Cache** — In-memory cosine similarity cache provides sub-10ms response latency for repeated or highly similar clinical queries without storing raw PHI.
- 💡 **Explainable AI (XAI) & Sentence Extraction** — Extracts exact supporting evidence sentences from retrieved documents, synthesizes clinical reasoning traces (with thought-leak and prompt-leak filtering), and calculates calibrated confidence scores.
- 📡 **Streaming & Enterprise Observability** — Server-Sent Events (SSE) streaming (`POST /query/stream`), deep health diagnostics (`GET /health`), Prometheus telemetry (`GET /metrics`), and structured JSON logging.
- 💻 **Multi-Interface Deployment** — Interactive boxed CLI dashboard (`backend/app.py`), modern Glassmorphism Web App (`frontend/index.html`), production WSGI runner (`backend/wsgi.py`), and multi-stage Docker orchestration (`docker-compose.yml`).

---

## 🏗️ End-to-End System Architecture

```mermaid
flowchart TD
    User([User / Clinician Query]) --> Sec[Security & Privacy Layer]
    
    subgraph Security ["1. Security, Privacy & Ingestion"]
        Sec --> Auth[API Key / RBAC Auth]
        Auth --> Rate[Sliding-Window Rate Limiter]
        Rate --> Scrub[HIPAA PHI Redaction Engine]
        Scrub --> Guard[Prompt Injection & XSS Guardrails]
    end

    Guard --> CacheCheck{Semantic Cache Hit?\nCos Sim >= 0.96}
    CacheCheck -- Yes --> CachedResp([Sub-10ms Cached Payload])
    CacheCheck -- No --> LAQA[LAQA Query Analyzer]

    subgraph PreRetrieval ["2. Pre-Retrieval Optimization (LAQA)"]
        LAQA --> Intent[Intent & Query Type Classifier]
        Intent --> Biomarkers[Biomarker & Synonym Mapping\nEGFR, KRAS, BRAF, MSI-H, ADC]
        Biomarkers --> Expand[Domain-Safe Query Expansion]
    end

    Expand --> Hybrid[Hybrid Retrieval Engine]

    subgraph Retrieval ["3. Dual-Index Hybrid Retrieval & Reranking"]
        Hybrid --> FAISS[Dense FAISS Index\nMRL 512-dim or Full 768-dim]
        Hybrid --> BM25[Sparse BM25 Index]
        FAISS & BM25 --> Fusion[Adaptive Intent-Weighted Fusion]
        Fusion --> Boost[Semantic & Entity Boosting]
        Boost --> Rerank[CrossEncoder Reranker\nBAAI/bge-reranker-large]
    end

    Rerank --> Compress[Context Compression\nToken Budget & Sentence Packing]
    Compress --> Gen[MedGemma Generator\nLocal / Cloud Ollama Engine]

    subgraph AgentLoop ["4. Evaluator & Autonomous Retry State Machine"]
        Gen --> Eval[Clinical Evaluator]
        Eval --> NegDetect[Clinical Negation & Contraindication Detector]
        NegDetect --> Decider{Score >= 7 & No Contradiction?}
        Decider -- No (Retry < 3) --> Action[Action Selection:\nescalate_k | expand_query]
        Action --> Expand
        Decider -- Yes / Max Attempts --> Best[Select Best Scored Answer]
    end

    Best --> XAI[Explainable AI Layer]

    subgraph Explainability ["5. Explainability & Output Delivery"]
        XAI --> SentExtract[Supporting Sentence Extraction]
        XAI --> CalibConf[Calibrated Confidence Scoring]
        XAI --> ClinReason[Clinical Reasoning Synthesis]
        XAI --> CacheStore[Update Semantic Vector Cache]
    end

    CacheStore --> Output([CLI Dashboard / REST API / Web UI / SSE Stream])
```

---

## 🛠️ Repository Directory Structure

The project maintains a modular, decoupled structure:

```
oncology-agentic-rag/
├── backend/
│   ├── app.py                      # Interactive Boxed CLI Dashboard & Pipeline Entrypoint
│   ├── server.py                   # Flask REST API Server, Streaming SSE & Web App Host
│   ├── wsgi.py                     # High-Performance Multi-Threaded WSGI Runner
│   ├── index_data.py               # Dual-Index Pipeline (MRL 512-dim & Full 768-dim)
│   ├── evaluation.py               # Benchmark Evaluation Suite (Grounding, Faithfulness, Relevance)
│   ├── metrics.py                  # Evaluation Metric Calculations (Semantic Sim, Overlap)
│   ├── paired_t_test.py            # Statistical Significance Test Suite (p-values, t-stats)
│   ├── settings.py                 # Central Configuration & Atomic Dynamic Runtime Updates
│   ├── runtime_settings.json       # Dynamic Runtime Settings State Store
│   ├── Dockerfile                  # Production Multi-Stage Container Definition
│   ├── requirements.txt            # Development Dependencies
│   ├── requirements-prod.txt       # Production-Pinned Dependencies
│   │
│   ├── modules/                    # Core Pipeline Modules
│   │   ├── agent/
│   │   │   ├── agent_controller.py # Agentic Retry Loop, Memory, & Fallbacks
│   │   │   ├── evaluator.py        # Response Evaluation & Negation-Penalized Scoring
│   │   │   ├── memory.py           # Context, Query History, & Drift Detection
│   │   │   └── semantic_cache.py   # Vector Similarity Cache with PHI Scrubbing
│   │   ├── clinical/
│   │   │   └── negation_detector.py # Deterministic Clinical Negation & Contraindication Detector
│   │   ├── security/
│   │   │   ├── auth.py             # API Key & Role-Based Access Control (RBAC)
│   │   │   ├── phi_scrubber.py     # HIPAA Safe Harbor 18-Identifier Redactor
│   │   │   ├── guardrails.py       # Prompt Injection, Script & Adversarial Filters
│   │   │   └── rate_limiter.py     # In-Memory Sliding-Window Rate Limiter
│   │   ├── observability/
│   │   │   ├── logger.py           # Structured JSON Production Logger
│   │   │   └── metrics_collector.py # Prometheus Latency, Throughput & Cache Metrics
│   │   ├── laqa/
│   │   │   └── laqa.py             # Language-Aware Query Analyzer & Biomarker Expansion
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py # FAISS + BM25 Hybrid Retriever with Safe Fallbacks
│   │   │   └── reranker.py         # CrossEncoder Reranker (BAAI/bge-reranker-large)
│   │   ├── chunking/
│   │   │   └── chunker.py          # Clinical Sentence-Sliding Window Chunker & Classifier
│   │   ├── embeddings/
│   │   │   └── mrl_embeddings.py   # MRL Dynamic Embedding Generator (Nomic v1.5 Lazy-Loaded)
│   │   ├── generator/
│   │   │   └── medgemma.py         # MedGemma LLM Generation Interface & Circuit Breaker
│   │   ├── optimization/
│   │   │   └── response_optimizer.py # Post-Generation Formatting & Markdown Sanitization
│   │   └── xai/
│   │       └── explain.py          # Explainable AI, Sentence Overlap & Confidence Calibration
│   │
│   ├── database/                   # Persistent Vector & Keyword Stores
│   │   ├── mrl/                    # MRL 512-dim FAISS Index + BM25 Store
│   │   ├── full/                   # Full 768-dim FAISS Index + BM25 Store
│   │   └── vector_store/           # Legacy Compatibility Store
│   ├── data/
│   │   └── oncology_docs/          # Curated Clinical & Guideline Source Documents
│   ├── questions/
│   │   └── cleaned_output.json     # Benchmark Evaluation Dataset
│   ├── report/                     # Benchmark Metric Reports & Visualizations
│   ├── results/                    # Statistical Evaluation Output Artifacts
│   ├── utils/                      # Helper Utilities & Metadata Classifiers
│   └── tests/                      # Automated Test Suite (42 unit & integration tests)
│       ├── test_clinical_guardrails.py
│       ├── test_pipeline.py
│       ├── test_production_api.py
│       ├── test_production_hardening.py
│       └── test_production_security.py
│
├── frontend/
│   └── index.html                  # Responsive Glassmorphism Single-Page Application
├── docker-compose.yml              # Multi-Service Production Compose Setup
├── .github/workflows/ci.yml        # Continuous Integration Pipeline
├── ARCHITECTURE.md                 # Complete System Architecture & Technical Specification
└── DUAL_INDEX_IMPLEMENTATION.md    # Dual Database Indexing Technical Reference
```

---

## ⚡ Quick Setup & Installation

### Prerequisites
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: 3.10 or higher
- **Ollama** (for local MedGemma LLM inference):
  - Download & Install: [ollama.ai](https://ollama.ai)
  - Pull model: `ollama pull medgemma` (or configure your model in `settings.py`)

### 1. Environment Setup
```powershell
# Clone workspace
git clone <repository-url>
cd oncology-agentic-rag

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Linux/macOS: source venv/bin/activate
```

### 2. Install Dependencies
```powershell
pip install -r backend/requirements.txt
```

---

## 🗂️ Document Indexing & Dual Index Setup

Source oncology documents reside in `backend/data/oncology_docs/`.

### Build Both Databases in a Single Pass (Recommended):
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; python backend/index_data.py --build-both
```

### Build Specific Stores:
```powershell
# Build MRL 512-dim store
python backend/index_data.py --store-type mrl

# Build Full 768-dim store
python backend/index_data.py --store-type full
```

*The indexing pipeline includes CPU thread capping and incremental disk checkpointing every 500 chunks to prevent memory spikes.*

---

## 🚀 Running the Application

### Option 1: Interactive CLI Dashboard (`app.py`)
Run the terminal dashboard with formatted 6-step diagnostic boxes:
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8=1; python backend/app.py
```

### Option 2: Production WSGI Server (`wsgi.py`)
For production serving with multi-threading:
```powershell
python backend/wsgi.py
```
Or with Gunicorn on Linux/Containers:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 --chdir backend wsgi:app
```

### Option 3: Development Server (`server.py`)
```powershell
python backend/server.py
```
- **Web Dashboard**: Open `http://localhost:5000/app` or `http://localhost:5000/`
- **REST API Endpoint**: `http://localhost:5000/query`

### Option 4: Full Docker Compose Deployment
```bash
docker compose up --build
```
Spins up:
1. `rag-api`: Hardened container running Python 3.10 with non-root security and health checks.
2. `ollama`: Ollama server with persistent volume for local MedGemma weights.

---

## 📡 REST API Reference

### 1. Synchronous Clinical Query
`POST /query`  
*Headers*: `Content-Type: application/json`, `X-API-Key: <key>` (optional in dev)

```json
{
  "query": "What are the first-line targeted therapies for EGFR Exon 19 deletion NSCLC?"
}
```

**Response (200 OK):**
```json
{
  "answer": "For advanced non-small cell lung cancer (NSCLC) harboring EGFR exon 19 deletions, third-generation EGFR tyrosine kinase inhibitors (TKIs), notably Osimertinib, represent the preferred first-line standard of care based on the FLAURA trial...",
  "confidence": 0.89,
  "quality": "High",
  "grounded": true,
  "reasoning": "- EGFR exon 19 deletion is an activating mutation sensitive to EGFR TKIs.\n- Osimertinib demonstrates superior progression-free survival compared to earlier generation TKIs.\n- Evidence confirms intracranial efficacy in central nervous system metastases.",
  "supporting_sentences": [
    "Osimertinib is a third-generation irreversible EGFR-TKI that selectively inhibits both EGFR-sensitizing and EGFR T790M resistance mutations."
  ],
  "sources": ["chunk_3059", "chunk_21189"],
  "evaluation": {
    "score": 9,
    "grounding_score": 0.94,
    "retrieval_score": 0.78,
    "answer_relevance": 0.95,
    "hallucination_risk": "low"
  },
  "query_analysis": {
    "intent": "clinical_guidance",
    "query_type": "treatment",
    "expanded_query": "first-line targeted therapies for egfr exon 19 deletion non small cell lung cancer osimertinib tagrisso egfr",
    "keywords": ["targeted", "therapies", "egfr", "exon", "deletion", "nsclc"]
  },
  "metrics": {
    "laqa_time": 0.08,
    "rag_time": 1.42,
    "xai_time": 0.04,
    "total_time": 1.54
  }
}
```

### 2. Streaming Query (Server-Sent Events)
`POST /query/stream`  
Emits real-time event stages: `laqa_start`, `laqa_complete`, `retrieval_complete`, `generating`, `evaluating`, `xai_complete`, `complete`.

### 3. Deep Health & Liveness Probe
`GET /health`  
Returns deep diagnostic checks on FAISS index files, dimension consistency, and live Ollama connectivity:
```json
{
  "status": "ok",
  "deep_status": "healthy",
  "active_database": "mrl",
  "mrl_enabled": true,
  "checks": {
    "index_files": {"status": "ok", "faiss_exists": true, "bm25_exists": true},
    "dimension": {"status": "ok", "configured": 512, "actual": 512},
    "ollama": {"status": "ok", "model": "medgemma", "latency_ms": 14.2}
  }
}
```

### 4. Prometheus Telemetry Metrics
`GET /metrics`  
Exposes Prometheus-formatted metrics:
```
# HELP rag_query_total Total clinical queries submitted
# TYPE rag_query_total counter
rag_query_total 128
# HELP rag_query_latency_seconds Latency percentiles
rag_query_latency_p50 1.45
rag_query_latency_p95 2.80
rag_query_latency_p99 3.95
# HELP rag_cache_hits_total Number of semantic cache hits
rag_cache_hits_total 18
```

### 5. Dynamic Runtime Settings
- `GET /settings` — View active configuration.
- `POST /settings/update` — Atomic, zero-downtime setting update (`admin` role required if auth enabled):
```json
{
  "enable_mrl": true,
  "enable_laqa": true,
  "retrieval_relevance_threshold": 0.45
}
```

---

## 🧪 Automated Testing Suite

The repository contains 42 automated tests covering security, HIPAA compliance, clinical guardrails, API contracts, and resilience:

```powershell
pytest backend/tests/ -v
```

**Test Coverage Summary:**
- `test_clinical_guardrails.py`: Clinical negation triggers, contraindication detection, and score penalty mechanics.
- `test_production_security.py`: API key RBAC, HIPAA PHI redaction, prompt injection filtering, and sliding-window rate limiting.
- `test_production_api.py`: Synchronous `/query`, streaming `/query/stream`, deep `/health`, and Prometheus `/metrics`.
- `test_production_hardening.py`: Atomic settings persistence, MRL lazy loading, FAISS resilience, and generator circuit breaker.
- `test_pipeline.py`: End-to-end integration across LAQA, Hybrid Retrieval, Evaluator, and XAI.

---

## 📊 Evaluation & Statistical Benchmarking

Benchmark the pipeline against clinical datasets (`backend/questions/cleaned_output.json`):

```powershell
# Run benchmark evaluation
python backend/evaluation.py

# Run paired t-test statistical significance analysis
python backend/paired_t_test.py
```

Outputs comprehensive metrics reports in `backend/report/` and statistical significance artifacts in `backend/results/`.

---

## 📖 In-Depth Technical References

- 📘 [ARCHITECTURE.md](file:///c:/Users/Sandhiya%20P/NIT%20INTERN/oncology-agentic-rag/ARCHITECTURE.md) — Exhaustive system architecture, component mathematical formulations, state-machine specs, and security design.
- 📙 [DUAL_INDEX_IMPLEMENTATION.md](file:///c:/Users/Sandhiya%20P/NIT%20INTERN/oncology-agentic-rag/DUAL_INDEX_IMPLEMENTATION.md) — Matryoshka Representation Learning (MRL) truncation mechanics and dual vector store routing.
- 💡 [Brainstorming & Enhancement Analysis](file:///c:/Users/Sandhiya%20P/.gemini/antigravity-ide/brain/3317ee65-3b3f-4e3d-ba41-0465010973e4/superpowers_brainstorming_analysis.md) — Strategic roadmap for scaling precision oncology capabilities.
