# 📐 Oncology Agentic RAG — System Architecture & Technical Specification

This document provides a detailed architectural specification of the **Oncology Agentic RAG** system, covering query expansion, hybrid retrieval, reranking, agentic retry loops, response evaluation, explainability, and database persistence.

---

## 1. Overview & Pipeline Dataflow

```
[ User Input Query ]
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: Language-Aware Query Analyzer (LAQA)              │
│   • Intent: Factual | Guidance | Comparison | Exploratory│
│   • Type: Treatment | Diagnosis | Symptoms | Prognosis   │
│   • Metadata: Cancer Type & Treatment Type Extraction    │
│   • Query Expansion & Cleaning                           │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: Hybrid Retrieval & Fusion                        │
│   • Dense MRL FAISS Search (512-dim or 768-dim)           │
│   • Sparse BM25 Keyword Matching                         │
│   • Adaptive Fusion (Dense/Sparse Weights by Intent)     │
│   • Semantic Boosting (Entity, Definition, Section,      │
│     Metadata Match, minus Noise Penalty)                 │
│   • Diversity Filter (Diversity threshold < 0.60)        │
│   • CrossEncoder Reranking (BAAI/bge-reranker-large)     │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: Agent Controller & Evaluator Retry Loop          │
│   • Candidate Context Preparation                        │
│   • MedGemma Response Generation                         │
│   • Evaluator Assessment (Grounding, Relevance, Risk)    │
│   • Auto-Retry (Up to 3 attempts with escalated K)      │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 4: Explainable AI (XAI) & Output Formatting         │
│   • Supporting Evidence Sentence Extraction              │
│   • XAI Step-by-Step Reasoning Trace Generation          │
│   • Confidence Score Calibration                         │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 5: Presentation Layer                               │
│   • Boxed ANSI Interactive CLI Dashboard (app.py)        │
│   • REST API & Single-Page Web Application (server.py)   │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Language-Aware Query Analyzer (LAQA)
- **Location**: `backend/modules/laqa/laqa.py`
- **Purpose**: Pre-retrieval optimization to overcome vocabulary mismatch in clinical queries.
- **Functions**:
  - `classify_intent(query)`: Categorizes query intent into `clinical_guidance`, `factual`, `comparison`, `exploratory`, or `epidemiology`.
  - `classify_query_type(query)`: Identifies query target (`treatment`, `diagnosis`, `symptoms`, `prognosis`, `ranking`, `definition`, `general`).
  - `extract_keywords(query)`: Filters stop words and extracts domain-specific medical tokens.
  - `expand_query(query)`: Expands short or ambiguous queries using prompt templates or rule-based expansion while filtering banned statistical noise patterns.
  - `classify_query_metadata(...)`: Detects cancer types (e.g. `lung cancer`, `breast cancer`, `melanoma`) and treatment categories (`targeted therapy`, `chemotherapy`, `immunotherapy`).

---

### 2.2 Dynamic Matryoshka Embedding Engine (MRL)
- **Location**: `backend/modules/embeddings/mrl_embeddings.py`
- **Model**: `nomic-ai/nomic-embed-text-v1.5`
- **Functionality**:
  - Matryoshka Representation Learning allows truncating high-dimensional embeddings to lower dimensions without losing significant semantic structure.
  - **MRL Enabled**: 512-dimensional truncated vectors (`database/mrl`). Fast search & lower RAM footprint.
  - **Full Mode**: 768-dimensional full vectors (`database/full`). Maximum recall precision.

---

### 2.3 Hybrid Search, Fusion & Reranking Engine
- **Location**: `backend/modules/retrieval/hybrid_retriever.py`
- **Algorithm**:
  1. **Candidate Retrieval**:
     - **Dense FAISS Search**: Retrieves top candidate indices using normalized L2 / Inner Product distance.
     - **Sparse BM25 Search**: Scores documents against extracted query tokens.
  2. **Adaptive Weight Fusion**:
     - Weights vary dynamically depending on query type:
       - `list / ranking`: Dense 0.35, Sparse 0.65
       - `definition`: Dense 0.70, Sparse 0.30
       - `default`: Dense 0.40, Sparse 0.60
  3. **Semantic Boosting & Penalties**:
     - Keyword Overlap score (+0.05 max)
     - Definition Boost (+0.05 for definition patterns)
     - Section Boost (+0.05 for section match)
     - Medical Entity Boost (+0.05)
     - Metadata Alignment Boost (+0.05)
     - Statistical Noise Penalty (-0.08 to -0.25 for non-epidemiology queries containing p-values, study populations, etc.)
  4. **Diversity Filter**:
     - Deduplicates candidate chunks using 60% token overlap threshold (`diversify_results`).
  5. **Reranking**:
     - `backend/modules/retrieval/reranker.py` utilizes `BAAI/bge-reranker-large`.
     - Final document score formula:
       $$\text{Score}_{\text{final}} = 0.70 \times \text{Score}_{\text{semantic}} + 0.20 \times \text{Score}_{\text{reranker}} + 0.10 \times \text{Score}_{\text{metadata}}$$

---

### 2.4 Agent Controller & Evaluator Retry Loop
- **Locations**:
  - `backend/modules/agent/agent_controller.py`
  - `backend/modules/agent/evaluator.py`
  - `backend/modules/agent/semantic_cache.py`
- **Execution Workflow**:
  1. **Semantic Cache Check**: If a query has $\ge 0.95$ cosine similarity with a cached query in `database/semantic_cache.json`, returns instant response.
  2. **Attempt 1**: Runs default retrieval $k=5$.
  3. **Evaluator Verification**:
     - Calculates Grounding score, Answer Relevance, and Hallucination Risk (`low`, `medium`, `high`).
     - Assigns an Overall Score out of 10.
  4. **Retry Loop Escalation**:
     - If Overall Score $< 7$ or `needs_retry == True`, escalates $k$ to $k=10$ (Attempt 2) and $k=15$ (Attempt 3).
     - Keeps history of attempts in memory (`modules/agent/memory.py`).

---

### 2.5 Generator & Response Optimization Layer
- **Locations**:
  - `backend/modules/generator/medgemma.py`
  - `backend/modules/optimization/response_optimizer.py`
- **Functionality**:
  - **MedGemma**: Specialized LLM prompt template enforcing strict medical evidence grounding and direct clinical answers.
  - **Response Optimization Layer**:
    - Strips thinking tokens (`<think>`, `<analysis>`, `<reasoning>`), prompt echo markers (`"the user wants"`, `"system prompt"`), and noisy chain-of-thought prefixes.
    - Prunes repetitive sentences and redundant bullet points.
    - Applies clinical typography and intent-aware structural formatting (`•` bullet hierarchies, clean capitalization, numbered lists).
    - Computes real-time optimization diff diagnostics (character delta, reduction percentage, stripped artifact counts).

---

### 2.6 XAI Explainability Layer
- **Location**: `backend/modules/xai/explain.py`
- **Functionality**:
  - Scans generated answer sentences against retrieved document chunks.
  - Extracts exact supporting sentences (`supporting_sentences`).
  - Constructs step-by-step reasoning trace (`reasoning`).
  - Calibrates final confidence score.

---

## 3. Database Persistence & Dual Indexing Architecture

### Directory Layout:
```
backend/database/
├── mrl/                  # Active when ENABLE_MRL = True
│   ├── faiss.index       # 512-dim FAISS index
│   ├── bm25.pkl          # Pickled BM25 search state
│   ├── ids.pkl           # List of document IDs
│   ├── id_to_text.pkl    # Text lookup map
│   ├── chunks.pkl        # Chunk metadata records
│   └── metadata.json     # Validation metadata (version, dimension, mode)
├── full/                 # Active when ENABLE_MRL = False
│   ├── faiss.index       # 768-dim FAISS index
│   └── metadata.json
└── semantic_cache.json   # Query-Answer disk cache
```

### Dynamic Path Selection (`backend/settings.py`):
```python
def get_database_path():
    if is_mrl_enabled():
        return "backend/database/mrl"
    return "backend/database/full"
```

---

## 4. Interfaces & Presentation

### 4.1 Terminal CLI Dashboard (`backend/app.py`)
- Standardized ANSI-formatted terminal runner.
- Output divided into 6 distinct steps:
  1. `Step 1 · Query Expansion (LAQA)`
  2. `Step 2 · Top Retrieved Docs` (Doc ID, Source File, Section, Semantic/Reranker/Final Scores, Text Snippet)
  3. `Step 3 · Agent Answer`
  4. `Step 4 · Supporting Evidence`
  5. `Step 5 · XAI (Explainability)`
  6. `Step 6 · Quality Metrics` (Confidence, Grounding, Retrieval, Hallucination Risk, Evaluator Status, Pipeline Latency Breakdown)

### 4.2 Web Application (`frontend/index.html` + `backend/server.py`)
- Glassmorphism dark mode UI.
- Real-time features:
  - Live Query Analysis Badges (Intent, Category, Cancer Type, Treatment Type)
  - Medical Term Auto-Highlighting (Teal pills for oncology vocabulary)
  - Supporting Evidence Quote Cards `[1]`, `[2]`
  - Collapsible XAI Reasoning Trace Drawer
  - Radial SVG Confidence Score Gauge
  - Quality Metrics Subgrid (Grounding %, Retrieval Score, Relevance %, Hallucination Risk)
  - Pipeline Execution Latency Bar (LAQA, RAG, XAI, Total)
  - Interactive Recent Chat History with Search & Session Storage
