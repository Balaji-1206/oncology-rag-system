import time
import numpy as np
from threading import Lock
from modules.embeddings.mrl_embeddings import get_mrl_embedding

# =========================================================
# 🔹 SEMANTIC QUERY VECTOR CACHE
# Sub-10ms response latency for repeated or highly similar queries
# =========================================================

_SEMANTIC_CACHE = []
_CACHE_LOCK = Lock()
MAX_SEMANTIC_CACHE_SIZE = 500
SIMILARITY_THRESHOLD = 0.96


def lookup_semantic_cache(query_text, threshold=SIMILARITY_THRESHOLD):
    """
    Search in-memory semantic query cache using cosine vector similarity.
    If cosine similarity >= threshold (default 0.96), returns cached result payload.
    Otherwise returns None.
    """
    if not query_text or not query_text.strip():
        return None

    with _CACHE_LOCK:
        if not _SEMANTIC_CACHE:
            return None

        # Compute query vector
        query_vec = get_mrl_embedding(query_text.strip(), log=False)[0]

        # Stack cached vectors
        cached_vecs = np.array([item["vector"] for item in _SEMANTIC_CACHE], dtype=np.float32)

        # Dot product cosine similarity (since vectors are L2 unit normalized)
        similarities = np.dot(cached_vecs, query_vec)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= threshold:
            cached_item = _SEMANTIC_CACHE[best_idx]
            print(f"[SEMANTIC CACHE HIT] Similarity: {round(best_sim, 4)} | Matched: '{cached_item['query']}'")
            
            # Return copy of result payload with cache metadata attached
            payload = dict(cached_item["payload"])
            payload["semantic_cache_hit"] = True
            payload["matched_query"] = cached_item["query"]
            payload["similarity_score"] = round(best_sim, 4)
            return payload

    return None


def add_to_semantic_cache(query_text, result_payload):
    """
    Add a successfully answered query and its output payload to the vector cache.
    """
    if not query_text or not result_payload:
        return

    query_vec = get_mrl_embedding(query_text.strip(), log=False)[0]

    with _CACHE_LOCK:
        # Evict oldest if full
        if len(_SEMANTIC_CACHE) >= MAX_SEMANTIC_CACHE_SIZE:
            _SEMANTIC_CACHE.pop(0)

        _SEMANTIC_CACHE.append({
            "query": query_text.strip(),
            "vector": query_vec,
            "payload": result_payload,
            "timestamp": time.time()
        })
