# pyrefly: ignore [missing-import]
import os
import re
import json
import shutil
import pickle
import faiss
import numpy as np
from datetime import datetime
from threading import Lock

import settings
from modules.retrieval.reranker import rerank
from modules.embeddings.mrl_embeddings import get_dynamic_mrl_embedding
from utils.metadata_tools import (
    classify_query_metadata,
    metadata_match_score,
    normalize_metadata_record
)


def migrate_legacy_database():
    """Migrates legacy single database at backend/database/vector_store to new dual structure."""
    legacy_path = os.path.join(settings.BACKEND_DIR, "database", "vector_store")
    if not os.path.exists(legacy_path):
        return

    print("🔍 Found legacy database, checking for migration...")
    legacy_faiss_path = f"{legacy_path}/faiss.index"
    if not os.path.exists(legacy_faiss_path):
        return

    try:
        legacy_index = faiss.read_index(legacy_faiss_path)
        dimension = legacy_index.d
    except Exception:
        return

    if dimension == 512:
        was_mrl = True
    elif dimension in (256, 768):
        was_mrl = False
    else:
        print(f"⚠️ Legacy database has unknown dimension {dimension}, skipping migration")
        return

    target_db = "mrl" if was_mrl else "full"
    target_path = os.path.join(settings.BACKEND_DIR, "database", target_db)

    if os.path.exists(target_path):
        print(f"✅ Target database {target_path} already exists, skipping migration")
        return

    print(f"📦 Migrating legacy database to {target_path}...")
    os.makedirs(target_path, exist_ok=True)

    files_to_copy = [
        "faiss.index", "bm25.pkl", "ids.pkl", "id_to_text.pkl",
        "chunks.pkl", "section_map.pkl", "docs.pkl", "index_settings.json"
    ]

    for file in files_to_copy:
        src = f"{legacy_path}/{file}"
        if os.path.exists(src):
            shutil.copy2(src, f"{target_path}/{file}")
            print(f"  ✓ Copied {file}")

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
        print("  ✓ Created metadata.json")

    print(f"✅ Migration complete. Legacy database kept at {legacy_path}")


# Run migration on import
migrate_legacy_database()


def validate_database_consistency():
    """Validates that active database matches current runtime settings."""
    db_path = settings.get_database_path()
    metadata_path = f"{db_path}/metadata.json"
    faiss_path = f"{db_path}/faiss.index"

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run backend/index_data.py to build indexes.")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    if metadata.get('mrl_enabled') != settings.is_mrl_enabled():
        raise ValueError(
            f"MRL mode mismatch: database built with MRL {'enabled' if metadata.get('mrl_enabled') else 'disabled'}, "
            f"current setting is MRL {'enabled' if settings.is_mrl_enabled() else 'disabled'}."
        )

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}.")

    index = faiss.read_index(faiss_path)
    expected_dim = settings.effective_embedding_dimension()

    if index.d != expected_dim:
        raise ValueError(f"FAISS dimension mismatch: index has {index.d}, expected {expected_dim}.")

    if metadata.get('embedding_dimension') != index.d:
        raise ValueError(f"Metadata dimension mismatch: metadata={metadata.get('embedding_dimension')}, FAISS={index.d}")

    return metadata


# Load Database Validation
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


# Global Runtime Cache
_CACHE = {
    "current_db_path": None,
    "faiss_index": None,
    "bm25": None,
    "ids": None,
    "id_to_text": None,
    "chunk_metadata": None
}
_CACHE_LOCK = Lock()


def _reload_all_indexes(verbose=False):
    """Reloads indexes from disk when active database path changes."""
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
                record, fallback_id=record.get("id", f"chunk_{idx}")
            )
            for idx, record in enumerate(raw_chunks or [])
            if isinstance(record, dict)
        }

    _CACHE["current_db_path"] = db_path
    if verbose:
        print(f"✅ Indexes reloaded successfully")


def get_index():
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["faiss_index"]


def get_bm25():
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["bm25"]


def get_ids():
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["ids"]


def get_id_to_text():
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["id_to_text"]


def get_chunk_metadata():
    db_path = settings.get_database_path()
    with _CACHE_LOCK:
        if _CACHE["current_db_path"] != db_path:
            _reload_all_indexes(verbose=True)
        return _CACHE["chunk_metadata"]


# Module import attempt
try:
    _reload_all_indexes()
except Exception:
    pass


def tokenize(text: str) -> list:
    tokens = re.findall(r"\b[a-zA-Z0-9\-]+\b", text.lower())
    return [t for t in tokens if len(t) > 1]


def normalize_unit_scores(scores):
    scores = np.array(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score == min_score:
        return np.ones_like(scores) * 0.5
    return (scores - min_score) / (max_score - min_score)


def detect_query_section(query: str):
    q = query.lower()
    section_map = {
        "symptoms": ["symptom", "sign", "indication"],
        "diagnosis": ["diagnosis", "detect", "screening"],
        "treatment": ["treatment", "therapy", "medicine", "drug"],
        "prognosis": ["survival", "mortality", "death", "prognosis"]
    }
    for section, kws in section_map.items():
        for kw in kws:
            if kw in q:
                return section
    return None


def extract_medical_entities(query: str) -> list:
    tokens = tokenize(query)
    medical_terms = {
        "cancer", "tumor", "tumour", "oncology", "lymphoma", "carcinoma",
        "sarcoma", "glioblastoma", "leukemia", "melanoma", "breast", "lung",
        "larynx", "thyroid", "immunotherapy", "chemotherapy", "radiotherapy"
    }
    return [t for t in tokens if len(t) > 3 or t in medical_terms]


def get_fusion_weights(intent: str, query_type: str) -> dict:
    if query_type in ["list", "ranking"]:
        return {"dense": 0.35, "sparse": 0.65}
    if query_type == "epidemiology":
        return {"dense": 0.50, "sparse": 0.50}
    if query_type == "definition":
        return {"dense": 0.70, "sparse": 0.30}
    if intent == "comparison":
        return {"dense": 0.50, "sparse": 0.50}
    if intent == "exploratory":
        return {"dense": 0.60, "sparse": 0.40}
    return {"dense": 0.40, "sparse": 0.60}


def keyword_overlap_score(query_tokens: list, doc_text: str) -> float:
    doc_tokens = set(tokenize(doc_text))
    overlap = len(set(query_tokens) & doc_tokens)
    return overlap / max(len(query_tokens), 1)


def definition_boost(query: str, text: str) -> float:
    q = query.lower()
    if "what is" in q or "define" in q:
        patterns = ["is a disease", "is defined as", "refers to", "characterized by", "abnormal cell growth"]
        t = text.lower()
        for p in patterns:
            if p in t:
                return 0.05
    return 0.0


def section_boost(query_section, doc_id: str) -> float:
    if query_section is None:
        return 0
    meta = get_chunk_metadata().get(doc_id, {})
    section = meta.get("section", "general")
    return 0.05 if section == query_section else 0


def educational_boost(query_type: str, text: str) -> float:
    if query_type != "definition":
        return 0
    educational_patterns = ["is a disease", "defined as", "refers to", "abnormal growth", "condition in which", "characterized by"]
    text_lower = text.lower()
    for p in educational_patterns:
        if p in text_lower:
            return 0.05
    return 0


_STAT_NOISE_PATTERNS = [
    "confidence interval", "statistically significant", "p-value", "prevalence",
    "incidence", "study population", "adult population", "retrospective study", "prospective study"
]
_EPIDEMIOLOGY_QUERY_TYPES = {"epidemiology", "clinical_trials", "prognosis", "ranking"}


def noise_penalty(text: str, query_type: str = None) -> float:
    if query_type in _EPIDEMIOLOGY_QUERY_TYPES:
        return 0.0
    text_lower = text.lower()
    penalty = sum(0.08 for p in _STAT_NOISE_PATTERNS if p in text_lower)
    return min(penalty, 0.25)


def metadata_alignment_boost(query_tokens: list, doc_id: str) -> float:
    meta = get_chunk_metadata().get(doc_id, {})
    if not meta:
        return 0.0
    metadata_text = " ".join([str(meta.get("section", "")), str(meta.get("source_doc", "")), str(meta.get("metadata", ""))]).lower()
    if metadata_text.strip().startswith("{"):
        try:
            metadata_text = json.dumps(json.loads(metadata_text)).lower()
        except Exception:
            pass

    boost = sum(0.02 for token in set(query_tokens) if token in metadata_text)
    return min(boost, 0.05)


def entity_boost(entities: list, text: str) -> float:
    text_lower = text.lower()
    count = sum(1 for e in entities if e in text_lower)
    return min(count * 0.03, 0.10)


def diversify_results(candidate_docs: list, id_to_text_map: dict, max_overlap: float = 0.60) -> list:
    """Removes near-duplicate document chunks based on token overlap threshold."""
    selected = []

    for doc_id in candidate_docs:
        text = id_to_text_map.get(doc_id, "")
        tokens = set(tokenize(text))
        if not tokens:
            selected.append(doc_id)
            continue

        duplicate = False
        for s_id in selected:
            s_tokens = set(tokenize(id_to_text_map.get(s_id, "")))
            if not s_tokens:
                continue
            overlap = len(tokens & s_tokens) / min(len(tokens), len(s_tokens))
            if overlap > max_overlap:
                duplicate = True
                break

        if not duplicate:
            selected.append(doc_id)

    return selected


def hybrid_search(query_payload, k=None):
    """Executes hybrid retrieval combining FAISS dense vector search and BM25 sparse search."""
    if isinstance(query_payload, str):
        query = query_payload
        intent = "factual"
        query_type = "general"
        keywords = tokenize(query)
        retrieval_k = k or 5
        query_metadata = classify_query_metadata(
            query=query, keywords=keywords, query_type=query_type, expanded_query=query
        )
    else:
        query = query_payload.get("expanded_query") or query_payload.get("original_query", "")
        intent = query_payload.get("intent", "factual")
        query_type = query_payload.get("query_type", "general")
        keywords = query_payload.get("keywords", [])
        retrieval_k = k or query_payload.get("retrieval_k", 5)
        query_metadata = query_payload.get("query_metadata") or classify_query_metadata(
            query=query, keywords=keywords, query_type=query_type, expanded_query=query
        )

    current_faiss = get_index()
    current_bm25 = get_bm25()
    current_ids = get_ids()
    current_id_to_text = get_id_to_text()
    current_chunk_metadata = get_chunk_metadata()

    # Dense Search
    q_emb = get_dynamic_mrl_embedding(query, target_dim=settings.effective_embedding_dimension())
    q_emb = np.array([q_emb], dtype=np.float32)

    candidate_multiplier = 4
    search_k = min(retrieval_k * candidate_multiplier, current_faiss.ntotal)
    dense_scores, dense_indices = current_faiss.search(q_emb, search_k)

    dense_candidates = {}
    for idx, score in zip(dense_indices[0], dense_scores[0]):
        if idx != -1 and idx < len(current_ids):
            doc_id = current_ids[idx]
            dense_candidates[doc_id] = float(score)

    # Sparse BM25 Search
    q_tokens = tokenize(query)
    bm25_scores = current_bm25.get_scores(q_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:search_k]

    sparse_candidates = {}
    for idx in top_bm25_indices:
        score = bm25_scores[idx]
        if score > 0:
            doc_id = current_ids[idx]
            sparse_candidates[doc_id] = float(score)

    all_candidate_ids = list(set(dense_candidates.keys()).union(set(sparse_candidates.keys())))
    if not all_candidate_ids:
        return {
            "texts": [], "ids": [], "retrieval_score": 0.0,
            "reranker_confidence": 0.0, "query_metadata": query_metadata,
            "retrieval_diagnostics": [], "candidate_texts": []
        }

    # Normalize scores
    raw_dense = [dense_candidates.get(doc_id, 0.0) for doc_id in all_candidate_ids]
    raw_sparse = [sparse_candidates.get(doc_id, 0.0) for doc_id in all_candidate_ids]

    norm_dense = normalize_unit_scores(raw_dense)
    norm_sparse = normalize_unit_scores(raw_sparse)

    weights = get_fusion_weights(intent, query_type)
    w_dense = weights["dense"]
    w_sparse = weights["sparse"]

    query_section = detect_query_section(query)
    medical_entities = extract_medical_entities(query)

    candidate_scores = {}
    for idx, doc_id in enumerate(all_candidate_ids):
        doc_text = current_id_to_text.get(doc_id, "")
        chunk_record = current_chunk_metadata.get(doc_id, {})

        base_score = (w_dense * norm_dense[idx]) + (w_sparse * norm_sparse[idx])

        kw_score = keyword_overlap_score(q_tokens, doc_text)
        def_boost = definition_boost(query, doc_text)
        sec_boost = section_boost(query_section, doc_id)
        edu_boost = educational_boost(query_type, doc_text)
        meta_boost = metadata_alignment_boost(q_tokens, doc_id)
        ent_boost = entity_boost(medical_entities, doc_text)
        n_penalty = noise_penalty(doc_text, query_type=query_type)

        rec_meta = chunk_record.get("metadata", {})
        m_score = metadata_match_score(query_metadata, rec_meta)

        semantic_score = base_score + (kw_score * 0.05) + def_boost + sec_boost + edu_boost + meta_boost + ent_boost - n_penalty

        candidate_scores[doc_id] = {
            "semantic_score": float(semantic_score),
            "metadata_score": float(m_score),
            "reranker_score": 0.0,
            "final_score": float(semantic_score)
        }

    # Sort & Rerank Candidates
    sorted_candidate_ids = sorted(all_candidate_ids, key=lambda d: candidate_scores[d]["semantic_score"], reverse=True)
    diversified_ids = diversify_results(sorted_candidate_ids, current_id_to_text, max_overlap=0.60)
    top_candidates = diversified_ids[:retrieval_k * 2]

    candidate_texts = [current_id_to_text.get(d, "") for d in top_candidates]

    rerank_results = rerank(query, candidate_texts)
    for res in rerank_results:
        c_idx = res["candidate_index"]
        if c_idx < len(top_candidates):
            doc_id = top_candidates[c_idx]
            rnk_score = float(res["reranker_score"])
            candidate_scores[doc_id]["reranker_score"] = rnk_score

            sem_score = candidate_scores[doc_id]["semantic_score"]
            meta_score = candidate_scores[doc_id]["metadata_score"]

            final_score = (0.70 * sem_score) + (0.20 * rnk_score) + (0.10 * meta_score)
            candidate_scores[doc_id]["final_score"] = float(final_score)

    final_sorted_ids = sorted(top_candidates, key=lambda d: candidate_scores[d]["final_score"], reverse=True)
    final_ids = final_sorted_ids[:retrieval_k]
    final_texts = [current_id_to_text.get(d, "") for d in final_ids]

    top_final_scores = [candidate_scores[d]["final_score"] for d in final_ids]
    retrieval_score = round(float(np.mean(top_final_scores)), 3) if top_final_scores else 0.0

    top_rerank_scores = [candidate_scores[d]["reranker_score"] for d in final_ids if candidate_scores[d]["reranker_score"] > 0]
    if top_rerank_scores:
        reranker_confidence = float(np.mean(top_rerank_scores))
        reranker_confidence = round(1 / (1 + np.exp(-reranker_confidence)), 3)
    else:
        reranker_confidence = 0.0

    # Diagnostics
    retrieval_diagnostics = []
    for doc_id in final_ids:
        chunk_record = current_chunk_metadata.get(doc_id, {})
        scores = candidate_scores.get(doc_id, {})
        retrieval_diagnostics.append({
            "doc_id": doc_id,
            "text": current_id_to_text.get(doc_id, ""),
            "metadata_score": scores.get("metadata_score", 0.0),
            "semantic_score": scores.get("semantic_score", 0.0),
            "reranker_score": scores.get("reranker_score", 0.0),
            "final_score": scores.get("final_score", 0.0),
            "metadata": chunk_record.get("metadata", {})
        })

    # CLI Output
    print("\n" + "─" * 66)
    print("  \033[1mStep 2 · Top Retrieved Docs\033[0m")
    print("─" * 66)

    for i, doc_id in enumerate(final_ids):
        text = current_id_to_text.get(doc_id, "")
        scores = candidate_scores.get(doc_id, {})
        chunk_record = current_chunk_metadata.get(doc_id, {})
        meta = chunk_record.get("metadata", {})
        if not isinstance(meta, dict):
            try:
                meta = json.loads(meta) if meta else {}
            except Exception:
                meta = {}

        source = meta.get("source_file") or meta.get("source_doc") or meta.get("source") or chunk_record.get("source_doc", "") or "unknown"
        section = meta.get("section") or chunk_record.get("section", "") or "—"
        page = meta.get("page", "") or meta.get("page_number", "")
        page_str = f"  p.{page}" if page else ""

        sem = scores.get("semantic_score", 0.0)
        rnk = scores.get("reranker_score", 0.0)
        fin = scores.get("final_score", 0.0)

        print(f"\n  \033[1m[{i+1}] {doc_id}\033[0m")
        print(f"      \033[1mSource  :\033[0m {source}{page_str}")
        print(f"      \033[1mSection :\033[0m {section}")
        print(f"      \033[1mScores  :\033[0m Semantic=\033[1m{sem:.3f}\033[0m  Reranker=\033[1m{rnk:.3f}\033[0m  Final=\033[1m{fin:.3f}\033[0m")
        print(f"      \033[1mSnippet :\033[0m {text[:150].strip()} ...")

    return {
        "texts": final_texts,
        "ids": final_ids,
        "retrieval_score": retrieval_score,
        "reranker_confidence": reranker_confidence,
        "query_metadata": query_metadata,
        "retrieval_diagnostics": retrieval_diagnostics,
        "candidate_texts": candidate_texts
    }