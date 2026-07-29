# pyrefly: ignore [missing-import]
import faiss
import pickle
import numpy as np
import re
import json
import settings
import os
import shutil
from datetime import datetime
from threading import Lock
from modules.retrieval.reranker import rerank
from modules.embeddings.mrl_embeddings import (
    get_dynamic_mrl_embedding
)
from utils.metadata_tools import (
    classify_query_metadata,
    metadata_match_score,
    normalize_metadata_record
)

# =========================================================
# 🔹 BACKWARD COMPATIBILITY MIGRATION
# =========================================================
def migrate_legacy_database():
    """
    If old single database exists at backend/database/vector_store/,
    migrate it to the new structure (mrl/ or full/).
    """
    legacy_path = os.path.join(settings.BACKEND_DIR, "database", "vector_store")
    metadata_path = os.path.join(legacy_path, "metadata.json")

    if not os.path.exists(legacy_path):
        return

    print("🔍 Found legacy database, checking for migration...")

    # Determine if this was an MRL or full database
    legacy_faiss_path = f"{legacy_path}/faiss.index"

    if not os.path.exists(legacy_faiss_path):
        return

    # Load FAISS to determine dimension
    try:
        legacy_index = faiss.read_index(legacy_faiss_path)
        dimension = legacy_index.d
    except Exception:
        return

    # Infer MRL mode from dimension
    if dimension == 512:
        was_mrl = True
    elif dimension in (256, 768):
        was_mrl = False
    else:
        print(f"⚠️  Legacy database has unknown dimension {dimension}, skipping migration")
        return

    # Determine target directory
    target_db = "mrl" if was_mrl else "full"
    target_path = os.path.join(settings.BACKEND_DIR, "database", target_db)

    # Check if target already exists
    if os.path.exists(target_path):
        print(f"✅ Target database {target_path} already exists, skipping migration")
        return

    # Migrate
    print(f"📦 Migrating legacy database to {target_path}...")

    os.makedirs(target_path, exist_ok=True)

    # Copy all files
    files_to_copy = [
        "faiss.index", "bm25.pkl", "ids.pkl", "id_to_text.pkl",
        "chunks.pkl", "section_map.pkl", "docs.pkl", "index_settings.json"
    ]

    for file in files_to_copy:
        src = f"{legacy_path}/{file}"
        if os.path.exists(src):
            dst = f"{target_path}/{file}"
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {file}")

    # Create metadata.json if it doesn't exist
    metadata_file = f"{target_path}/metadata.json"
    if not os.path.exists(metadata_file):
        metadata = {
            "mrl_enabled": was_mrl,
            "embedding_dimension": dimension,
            "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "version": 2,
            "migrated_from": "legacy_vector_store"
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print(f"  ✓ Created metadata.json")

    print(f"✅ Migration complete. Legacy database kept at {legacy_path}")

# Run migration on import
migrate_legacy_database()

# =========================================================
# 🔹 VALIDATION & DATABASE LOADING
# =========================================================
def validate_database_consistency():
    """
    Validate that the selected database is consistent with current settings.
    Checks: metadata exists, FAISS dimension matches, MRL mode matches.
    Raises clear error if mismatch exists.
    """
    db_path = settings.get_database_path()
    metadata_path = f"{db_path}/metadata.json"
    faiss_path = f"{db_path}/faiss.index"

    # Check metadata exists
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            f"Database may not be built. Run backend/index_data.py to build indexes."
        )

    # Load and validate metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Check MRL mode consistency
    if metadata.get('mrl_enabled') != settings.is_mrl_enabled():
        raise ValueError(
            f"MRL mode mismatch: "
            f"database was built with MRL {'enabled' if metadata.get('mrl_enabled') else 'disabled'}, "
            f"but current setting is MRL {'enabled' if settings.is_mrl_enabled() else 'disabled'}. "
            f"This database cannot be used. "
            f"Run backend/index_data.py to rebuild."
        )

    # Check FAISS index exists
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_path}. Database may be corrupted."
        )

    # Load FAISS and check dimension
    index = faiss.read_index(faiss_path)
    expected_dim = settings.effective_embedding_dimension()

    if index.d != expected_dim:
        raise ValueError(
            f"FAISS dimension mismatch: "
            f"index has dimension {index.d}, "
            f"but expected dimension {expected_dim}. "
            f"Database was built with different settings. "
            f"Run backend/index_data.py to rebuild."
        )

    # Check metadata dimension matches FAISS
    if metadata.get('embedding_dimension') != index.d:
        raise ValueError(
            f"Metadata dimension mismatch: "
            f"metadata says {metadata.get('embedding_dimension')}, "
            f"FAISS has {index.d}"
        )

    return metadata

# =========================================================
# 🔹 LOAD DATABASE
# =========================================================
print("🔥 Loading FAISS + BM25 indexes...")

try:
    metadata = validate_database_consistency()
    db_path = settings.get_database_path()

    print(f"✅ Database validation passed")
    print(f"   Active database: {db_path}")
    print(f"   MRL mode: {'ENABLED' if metadata['mrl_enabled'] else 'DISABLED'}")
    print(f"   Dimension: {metadata['embedding_dimension']}")

except (FileNotFoundError, ValueError) as e:
    print(f"❌ Database validation failed: {e}")
    raise

# =========================================================
# 🔹 GLOBAL RUNTIME CACHING
# =========================================================
# Cache state: track current database path and loaded objects
_CACHE = {
    "current_db_path": None,
    "faiss_index": None,
    "bm25": None,
    "ids": None,
    "id_to_text": None,
    "chunk_metadata": None
}

# Protects _CACHE from concurrent request corruption
_CACHE_LOCK = Lock()

def _reload_all_indexes(verbose=False):
    """Internal: reload all indexes from disk. Called only when db path changes."""
    db_path = settings.get_database_path()

    if verbose:
        print(f"🔄 Reloading indexes from {db_path}...")

    _CACHE["faiss_index"] = faiss.read_index(f"{db_path}/faiss.index")

    with open(f"{db_path}/bm25.pkl", "rb") as f:
        _CACHE["bm25"] = pickle.load(f)

    with open(f"{db_path}/ids.pkl", "rb") as f:
        _CACHE["ids"] = pickle.load(f)

    with open(f"{db_path}/id_to_text.pkl", "rb") as f:
        _CACHE["id_to_text"] = pickle.load(f)

    with open(f"{db_path}/chunks.pkl", "rb") as f:
        raw_chunks = pickle.load(f)

    if isinstance(raw_chunks, dict):
        _CACHE["chunk_metadata"] = {
            doc_id: normalize_metadata_record(record, fallback_id=doc_id)
            for doc_id, record in raw_chunks.items()
        }
    else:
        _CACHE["chunk_metadata"] = {
            record.get("id", f"chunk_{idx}"): normalize_metadata_record(
                record,
                fallback_id=record.get("id", f"chunk_{idx}")
            )
            for idx, record in enumerate(raw_chunks or [])
            if isinstance(record, dict)
        }

    _CACHE["current_db_path"] = db_path

    if verbose:
        print(f"✅ Indexes reloaded successfully")

def get_index():
    """Get FAISS index, reloading only if database path changed."""
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["faiss_index"]

def get_bm25():
    """Get BM25 index, reloading only if database path changed."""
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["bm25"]

def get_ids():
    """Get document IDs, reloading only if database path changed."""
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["ids"]

def get_id_to_text():
    """Get document text map, reloading only if database path changed."""
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["id_to_text"]

def get_chunk_metadata():
    """Get chunk metadata, reloading only if database path changed."""
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["chunk_metadata"]

# Try load once at module import (fails gracefully if index not built yet)
try:
    _reload_all_indexes()
except Exception:
    pass


# =========================================================
# 🔹 TOKENIZER
# =========================================================
def tokenize(text):

    text = text.lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9\-]+\b",
        text
    )

    return [
        t for t in tokens
        if len(t) > 1
    ]


# =========================================================
# 🔹 NORMALIZATION
# =========================================================
# Note: normalize_scores (z-score) was removed — it returned negative values
# and was never called. Use normalize_unit_scores (min-max, 0–1) instead.


def normalize_unit_scores(scores):

    scores = np.array(
        scores,
        dtype=np.float32
    )

    if len(scores) == 0:
        return scores

    min_score = float(scores.min())
    max_score = float(scores.max())

    if max_score == min_score:
        return np.ones_like(scores) * 0.5

    return (scores - min_score) / (max_score - min_score)


# =========================================================
# 🔹 QUERY SECTION DETECTION
# =========================================================
def detect_query_section(query):

    q = query.lower()

    section_map = {

        "symptoms": [
            "symptom",
            "sign",
            "indication"
        ],

        "diagnosis": [
            "diagnosis",
            "detect",
            "screening"
        ],

        "treatment": [
            "treatment",
            "therapy",
            "medicine",
            "drug"
        ],

        "prognosis": [
            "survival",
            "mortality",
            "death",
            "prognosis"
        ]
    }

    for section, kws in section_map.items():

        for kw in kws:

            if kw in q:
                return section

    return None


# =========================================================
# 🔹 QUERY ENTITY BOOST
# =========================================================
def extract_medical_entities(query):

    important = []

    tokens = tokenize(query)

    medical_terms = {

        "cancer",
        "tumor",
        "tumour",
        "oncology",
        "lymphoma",
        "carcinoma",
        "sarcoma",
        "glioblastoma",
        "leukemia",
        "melanoma",
        "breast",
        "lung",
        "larynx",
        "thyroid",
        "immunotherapy",
        "chemotherapy",
        "radiotherapy"
    }

    for t in tokens:

        if (
            len(t) > 3
            or
            t in medical_terms
        ):

            important.append(t)

    return important


# =========================================================
# 🔹 FUSION WEIGHTS
# =========================================================
def get_fusion_weights(intent, query_type):

    # ==========================================
    # LIST / RANKING
    # ==========================================
    if query_type in [

        "list",
        "ranking"
    ]:

        return {
            "dense": 0.35,
            "sparse": 0.65
        }

    # ==========================================
    # DEFINITIONS
    # ==========================================
    if query_type == "definition":

        return {
            "dense": 0.30,
            "sparse": 0.70
        }

    # ==========================================
    # COMPARISON
    # ==========================================
    if intent == "comparison":

        return {
            "dense": 0.50,
            "sparse": 0.50
        }

    # ==========================================
    # EXPLORATORY
    # ==========================================
    if intent == "exploratory":

        return {
            "dense": 0.60,
            "sparse": 0.40
        }

    # ==========================================
    # DEFAULT
    # ==========================================
    return {
        "dense": 0.40,
        "sparse": 0.60
    }


# =========================================================
# 🔹 KEYWORD OVERLAP
# =========================================================
def keyword_overlap_score(
    query_tokens,
    doc_text
):

    doc_tokens = set(
        tokenize(doc_text)
    )

    overlap = len(
        set(query_tokens)
        &
        doc_tokens
    )

    return overlap / max(
        len(query_tokens),
        1
    )


# =========================================================
# 🔹 DEFINITION BOOST
# =========================================================
def definition_boost(query, text):

    q = query.lower()

    if (
        "what is" in q
        or "define" in q
    ):

        patterns = [

            "is a disease",

            "is defined as",

            "refers to",

            "characterized by",

            "abnormal cell growth",
        ]

        t = text.lower()

        for p in patterns:

            if p in t:
                return 0.05

    return 0.0


# =========================================================
# 🔹 SECTION BOOST
# =========================================================
def section_boost(
    query_section,
    doc_id
):

    if query_section is None:
        return 0

    meta = get_chunk_metadata().get(
        doc_id,
        {}
    )

    section = meta.get(
        "section",
        "general"
    )

    if section == query_section:
        return 0.05

    return 0


# =========================================================
# 🔹 EDUCATIONAL BOOST
# =========================================================
def educational_boost(
    query_type,
    text
):

    if query_type != "definition":
        return 0

    educational_patterns = [

        "is a disease",

        "defined as",

        "refers to",

        "abnormal growth",

        "condition in which",

        "characterized by",
    ]

    text_lower = text.lower()

    for p in educational_patterns:

        if p in text_lower:
            return 0.05

    return 0


# =========================================================
# 🔹 NOISE PENALTY
# =========================================================
# Statistical/methodological terms are penalized for most query types,
# but NOT for epidemiology/clinical_trials/prognosis queries where
# terms like 'prevalence', 'incidence', 'p-value' are relevant.
_STAT_NOISE_PATTERNS = [
    "confidence interval",
    "statistically significant",
    "p-value",
    "prevalence",
    "incidence",
    "study population",
    "adult population",
    "retrospective study",
    "prospective study"
]

_EPIDEMIOLOGY_QUERY_TYPES = {
    "epidemiology", "clinical_trials", "prognosis", "ranking"
}

def noise_penalty(text, query_type=None):

    # Skip penalty entirely for query types where stats ARE the content
    if query_type in _EPIDEMIOLOGY_QUERY_TYPES:
        return 0.0

    text_lower = text.lower()

    penalty = 0.0

    for p in _STAT_NOISE_PATTERNS:

        if p in text_lower:
            penalty += 0.08

    return min(
        penalty,
        0.25
    )


# =========================================================
# 🔹 METADATA ALIGNMENT BOOST
# =========================================================
def metadata_alignment_boost(
    query_tokens,
    doc_id
):

    meta = get_chunk_metadata().get(
        doc_id,
        {}
    )

    if not meta:
        return 0.0

    metadata_text = " ".join([

        str(meta.get("section", "")),

        str(meta.get("source_doc", "")),

        str(meta.get("metadata", ""))
    ]).lower()

    if metadata_text.strip().startswith("{"):

        try:

            metadata_text = json.dumps(
                json.loads(metadata_text)
            ).lower()

        except Exception:

            pass

    boost = 0.0

    for token in set(query_tokens):

        if token in metadata_text:

            boost += 0.01

    return min(
        boost,
        0.05
    )


# =========================================================
# 🔹 ENTITY BOOST
# =========================================================
def entity_boost(
    entities,
    text
):

    if not entities:
        return 0

    text_lower = text.lower()

    score = 0

    for e in entities:

        if e in text_lower:
            score += 0.02

    return min(
        score,
        0.05
    )


# =========================================================
# 🔹 DIVERSITY FILTER
# =========================================================
def diversify_results(
    texts,
    ids,
    max_docs=4
):

    final_texts = []

    final_ids = []

    seen = []

    for text, doc_id in zip(texts, ids):

        current = set(
            tokenize(text[:300])
        )

        duplicate = False

        for prev in seen:

            overlap = len(current & prev)

            similarity = overlap / max(
                len(current),
                1
            )

            if similarity > 0.60:

                duplicate = True
                break

        if duplicate:
            continue

        seen.append(current)

        final_texts.append(text)

        final_ids.append(doc_id)

        if len(final_texts) >= max_docs:
            break

    return final_texts, final_ids


# =========================================================
# 🔹 MAIN HYBRID SEARCH
# =========================================================
def hybrid_search(
    laqa_output,
    _
):

    current_index = get_index()
    expected_dim = settings.effective_embedding_dimension()

    if current_index.d != expected_dim:

        raise ValueError(
            "WARNING: FAISS index dimension mismatch. "
            f"Index dimension is {current_index.d}, but active embedding "
            f"dimension is {expected_dim}. Reindex with "
            "backend/index_data.py after changing MRL mode."
        )

    query = laqa_output.get(
        "expanded_query",
        laqa_output.get("original_query", "")
    )

    original_query = laqa_output.get(
        "original_query",
        query
    )

    intent = laqa_output.get(
        "intent",
        "factual"
    )

    query_type = laqa_output.get(
        "query_type",
        "general"
    )

    k = laqa_output.get(
        "retrieval_k",
        5
    )

    query_tokens = tokenize(query)

    query_entities = extract_medical_entities(
        query
    )

    query_section = detect_query_section(
        query
    )

    query_metadata = laqa_output.get(
        "query_metadata"
    ) or classify_query_metadata(
        query=original_query,
        keywords=laqa_output.get(
            "keywords",
            []
        ),
        query_type=query_type,
        expanded_query=query
    )

    # =====================================================
    # 🔹 FUSION
    # =====================================================
    weights = get_fusion_weights(
        intent,
        query_type
    )

    dense_weight = weights["dense"]

    sparse_weight = weights["sparse"]

    print(
        f"🔍 Fusion → Dense={dense_weight} Sparse={sparse_weight}"
    )

    # =====================================================
    # 🔹 CANDIDATE POOL
    # =====================================================
    candidate_k = min(
        max(k * 6, 25),
        50
    )

    # =====================================================
    # 🔹 DENSE SEARCH
    # =====================================================
    query_vec = get_dynamic_mrl_embedding(
        [query],
        intent=intent
    )

    current_bm25 = get_bm25()
    current_ids = get_ids()
    current_id_to_text = get_id_to_text()
    current_chunk_metadata = get_chunk_metadata()

    D, I = current_index.search(
        query_vec,
        candidate_k
    )

    dense_ids = []

    dense_scores = []

    for idx, dist in zip(I[0], D[0]):

        if idx < 0 or idx >= len(current_ids):
            continue

        dense_ids.append(current_ids[idx])

        dense_scores.append(float(dist))

    dense_scores = normalize_unit_scores(
        dense_scores
    )

    # =====================================================
    # 🔹 SPARSE SEARCH
    # =====================================================
    bm25_scores = current_bm25.get_scores(
        query_tokens
    )

    top_idx = np.argsort(
        bm25_scores
    )[-candidate_k:]

    sparse_ids = [
        current_ids[i]
        for i in top_idx
    ]

    sparse_scores = normalize_unit_scores(
        [bm25_scores[i] for i in top_idx]
    )

    # =====================================================
    # 🔹 SCORE FUSION
    # =====================================================
    doc_score_map = {}

    # Dense
    for doc_id, score in zip(
        dense_ids,
        dense_scores
    ):

        doc_score_map[doc_id] = (
            doc_score_map.get(doc_id, 0)
            +
            dense_weight * score
        )

    # Sparse
    for doc_id, score in zip(
        sparse_ids,
        sparse_scores
    ):

        doc_score_map[doc_id] = (
            doc_score_map.get(doc_id, 0)
            +
            sparse_weight * score
        )

    # =====================================================
    # 🔹 SEMANTIC BOOSTING
    # =====================================================
    for doc_id in list(doc_score_map.keys()):

        text = current_id_to_text[doc_id]

        overlap_score = keyword_overlap_score(
            query_tokens,
            text
        )

        doc_score_map[doc_id] += (
            0.05 * overlap_score
        )

        doc_score_map[doc_id] += definition_boost(
            query,
            text
        )

        doc_score_map[doc_id] += section_boost(
            query_section,
            doc_id
        )

        doc_score_map[doc_id] += educational_boost(
            query_type,
            text
        )

        doc_score_map[doc_id] += entity_boost(
            query_entities,
            text
        )

        doc_score_map[doc_id] -= noise_penalty(
            text,
            query_type=query_type
        )

    semantic_scores_raw = {
        doc_id: max(score, 0.0)
        for doc_id, score in doc_score_map.items()
    }

    semantic_scores = {
        doc_id: float(score)
        for doc_id, score in zip(
            semantic_scores_raw.keys(),
            normalize_unit_scores(
                list(semantic_scores_raw.values())
            )
        )
    }

    # =====================================================
    # 🔹 SEMANTIC RANKING
    # =====================================================
    ranked_semantic_docs = sorted(
        semantic_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # =====================================================
    # 🔹 TOP CANDIDATES
    # =====================================================
    candidate_ids = [
        doc_id
        for doc_id, _
        in ranked_semantic_docs[:20]
    ]

    candidate_texts = [
        current_id_to_text[doc_id]
        for doc_id in candidate_ids
    ]

    # =====================================================
    # 🔹 RERANK
    # =====================================================
    reranked_texts, reranker_scores = rerank(
        query=query,
        docs=candidate_texts,
        top_k=10,
        return_scores=True,
        query_type=query_type
    )

    reranker_score_map = {}

    # =====================================================
    # 🔹 RECOVER IDS
    # =====================================================
    reranked_ids = []

    used_ids = set()

    for idx, text in enumerate(reranked_texts):

        reranker_score = 0.0

        if idx < len(reranker_scores):

            reranker_score = float(reranker_scores[idx])

        for doc_id in candidate_ids:

            if doc_id in used_ids:
                continue

            if current_id_to_text[doc_id] == text:

                reranked_ids.append(doc_id)
                reranker_score_map[doc_id] = reranker_score
                used_ids.add(doc_id)
                break

    for doc_id in candidate_ids:
        reranker_score_map.setdefault(
            doc_id,
            0.0
        )

    # =====================================================
    # 🔹 FINAL SCORE ASSEMBLY
    # =====================================================
    candidate_scores = {}

    for doc_id in candidate_ids:

        chunk_record = current_chunk_metadata.get(
            doc_id,
            {}
        )

        chunk_meta = chunk_record.get(
            "metadata",
            chunk_record
        )

        metadata_score = metadata_match_score(
            query_metadata,
            chunk_meta
        )

        semantic_score = float(
            semantic_scores.get(
                doc_id,
                0.0
            )
        )

        # reranker_score_map already stores calibrated confidence (0.0 to 1.0)
        # Re-applying sigmoid here was squashing all scores into a compressed [0.51, 0.72] range
        reranker_score = float(
            reranker_score_map.get(
                doc_id,
                0.0
            )
        )

        final_score = (
            0.70 * semantic_score
            + 0.20 * reranker_score
            + 0.10 * metadata_score
        )

        candidate_scores[doc_id] = {
            "semantic_score": round(semantic_score, 3),
            "metadata_score": round(metadata_score, 3),
            "reranker_score": round(reranker_score, 3),
            "final_score": round(max(0.0, min(final_score, 1.0)), 3)
        }

    ranked_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1]["final_score"],
        reverse=True
    )

    # =====================================================
    # 🔹 DIVERSITY FILTER
    # =====================================================
    ranked_texts = [
        current_id_to_text[doc_id]
        for doc_id, _
        in ranked_candidates
    ]

    ranked_ids = [
        doc_id
        for doc_id, _
        in ranked_candidates
    ]

    final_texts, final_ids = diversify_results(
        ranked_texts,
        ranked_ids,
        max_docs=4
    )

    # =====================================================
    # 🔹 RETRIEVAL SCORE
    # =====================================================
    final_scores = [
        candidate_scores.get(doc_id, {}).get("final_score", 0.0)
        for doc_id in final_ids
    ]

    retrieval_score = float(
        np.mean(final_scores) if final_scores else 0.0
    )

    retrieval_score = max(
        0.0,
        min(
            retrieval_score,
            1.0
        )
    )

    retrieval_score = round(
        retrieval_score,
        3
    )

    # =====================================================
    # 🔹 RERANKER CONFIDENCE
    # =====================================================
    if reranker_scores:

        reranker_confidence = float(
            np.mean(
                reranker_scores[:3]
            )
        )

        reranker_confidence = round(
            1 / (
                1 +
                np.exp(
                    -reranker_confidence
                )
            ),
            3
        )

    else:

        reranker_confidence = 0.0

    # =====================================================
    # 🔹 DIAGNOSTICS
    # =====================================================
    retrieval_diagnostics = []

    for doc_id in final_ids:

        chunk_record = current_chunk_metadata.get(
            doc_id,
            {}
        )

        scores = candidate_scores.get(
            doc_id,
            {}
        )

        retrieval_diagnostics.append({

            "doc_id": doc_id,

            "text": current_id_to_text.get(
                doc_id,
                ""
            ),

            "metadata_score": scores.get(
                "metadata_score",
                0.0
            ),

            "semantic_score": scores.get(
                "semantic_score",
                0.0
            ),

            "reranker_score": scores.get(
                "reranker_score",
                0.0
            ),

            "final_score": scores.get(
                "final_score",
                0.0
            ),

            "metadata": chunk_record.get(
                "metadata",
                {}
            )
        })

    # =====================================================
    # 🔹 DEBUG
    # =====================================================
    print("\n📄 Top Retrieved Docs:")

    for i, d in enumerate(final_texts):

        print(
            f"{i+1}.",
            d[:180],
            "..."
        )

    # =====================================================
    # 🔹 RETURN
    # =====================================================
    return {

        "texts": final_texts,

        "ids": final_ids,

        "retrieval_score": retrieval_score,

        "reranker_confidence": reranker_confidence,

        "query_metadata": query_metadata,

        "retrieval_diagnostics": retrieval_diagnostics,

        "candidate_texts": candidate_texts
    }