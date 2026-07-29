import json
import os
from threading import Lock

from utils.metadata_tools import classify_query_metadata


# =========================================================
# GLOBAL FEATURE FLAGS - DO NOT EDIT
# =========================================================
# These are defaults only. To change settings, edit:
# -> runtime_settings.json (actual config file)
# Or use: settings.update_settings({'enable_mrl': False})
# =========================================================
ENABLE_LAQA = True
ENABLE_MRL = False
ENABLE_RAG = True

MRL_DIMENSION = 512
# nomic-ai/nomic-embed-text-v1.5 produces 768 dims at full resolution.
# 256 was incorrect and silently truncated embeddings in full (non-MRL) mode.
FULL_EMBEDDING_DIMENSION = 768
RETRIEVAL_RELEVANCE_THRESHOLD = 0.65

# =========================================================
# MODULE-LEVEL SETTINGS CACHE
# Avoids repeated disk reads on every is_rag_enabled() call.
# Invalidated on update_settings().
# =========================================================
_settings_cache = None

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

_SETTINGS_PATH = os.path.join(
    BACKEND_DIR,
    "runtime_settings.json"
)

_LOCK = Lock()


def _defaults():

    return {
        "enable_rag": ENABLE_RAG,
        "enable_laqa": ENABLE_LAQA,
        "enable_mrl": ENABLE_MRL,
        "mrl_dimension": MRL_DIMENSION,
        "full_embedding_dimension": FULL_EMBEDDING_DIMENSION,
        "retrieval_relevance_threshold": RETRIEVAL_RELEVANCE_THRESHOLD,
        "active_database": "mrl"
    }


def _coerce_bool(value, default):

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        lowered = value.strip().lower()

        if lowered in {"true", "1", "yes", "on"}:
            return True

        if lowered in {"false", "0", "no", "off"}:
            return False

    return default


def load_settings():

    data = _defaults()

    if os.path.exists(_SETTINGS_PATH):

        try:

            with open(
                _SETTINGS_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                stored = json.load(f)

            if isinstance(stored, dict):
                data.update(stored)

        except Exception as exc:

            print(
                f"WARNING: Failed to read runtime settings: {exc}"
            )

    data["enable_rag"] = _coerce_bool(
        data.get("enable_rag"),
        ENABLE_RAG
    )

    data["enable_laqa"] = _coerce_bool(
        data.get("enable_laqa"),
        ENABLE_LAQA
    )

    data["enable_mrl"] = _coerce_bool(
        data.get("enable_mrl"),
        ENABLE_MRL
    )

    data["mrl_dimension"] = int(
        data.get("mrl_dimension", MRL_DIMENSION)
    )

    data["full_embedding_dimension"] = int(
        data.get(
            "full_embedding_dimension",
            FULL_EMBEDDING_DIMENSION
        )
    )

    data["retrieval_relevance_threshold"] = float(
        data.get(
            "retrieval_relevance_threshold",
            RETRIEVAL_RELEVANCE_THRESHOLD
        )
    )

    data["active_database"] = str(
        data.get("active_database", "mrl")
    )

    return data


def save_settings(data):

    with _LOCK:

        with open(
            _SETTINGS_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                sort_keys=True
            )


def get_settings():
    """Return cached settings, loading from disk only once per change."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = load_settings()
    return _settings_cache


def update_settings(payload):
    global _settings_cache

    current = load_settings()

    if "enable_rag" in payload:

        current["enable_rag"] = _coerce_bool(
            payload.get("enable_rag"),
            current["enable_rag"]
        )

    if "enable_laqa" in payload:

        current["enable_laqa"] = _coerce_bool(
            payload.get("enable_laqa"),
            current["enable_laqa"]
        )

    if "enable_mrl" in payload:

        current["enable_mrl"] = _coerce_bool(
            payload.get("enable_mrl"),
            current["enable_mrl"]
        )

    if "retrieval_relevance_threshold" in payload:

        current["retrieval_relevance_threshold"] = float(
            payload.get(
                "retrieval_relevance_threshold",
                current["retrieval_relevance_threshold"]
            )
        )

    if "active_database" in payload:

        current["active_database"] = str(
            payload.get("active_database")
        )

    save_settings(current)

    # Invalidate cache so next call reflects the new values
    _settings_cache = None

    return current


def is_laqa_enabled():

    return bool(
        get_settings()["enable_laqa"]
    )


def is_rag_enabled():

    return bool(
        get_settings()["enable_rag"]
    )


def get_active_database():

    env_val = os.environ.get("ACTIVE_DATABASE")
    if env_val:
        return env_val
    return get_settings().get("active_database", "mrl")


def is_mrl_enabled():

    active_db = get_active_database()
    if active_db == "dockling_mrl":
        return True

    return bool(
        get_settings()["enable_mrl"]
    )


def effective_embedding_dimension():

    active_db = get_active_database()
    if active_db == "dockling_mrl":
        return 512

    data = get_settings()

    if data["enable_mrl"]:
        return int(data["mrl_dimension"])

    return int(data["full_embedding_dimension"])


def retrieval_relevance_threshold():

    return float(
        get_settings()["retrieval_relevance_threshold"]
    )


def get_database_path():
    """
    Returns the active database path based on MRL setting.
    MRL enabled: returns absolute path to 'backend/database/mrl/'
    MRL disabled: returns absolute path to 'backend/database/full/'
    ACTIVE_DATABASE=dockling_mrl: returns 'backend/dockling/database/mrl/'
    """
    active_db = get_active_database()
    if active_db == "dockling_mrl":
        return os.path.join(BACKEND_DIR, "dockling", "database", "mrl")

    if is_mrl_enabled():
        return os.path.join(BACKEND_DIR, "database", "mrl")
    return os.path.join(BACKEND_DIR, "database", "full")


def build_raw_query_payload(user_query):

    query_metadata = classify_query_metadata(
        query=user_query,
        keywords=[],
        query_type="general",
        expanded_query=user_query
    )

    return {
        "intent": "factual",
        "query_type": "general",
        "keywords": [],
        "query_metadata": query_metadata,
        "expanded_query": user_query,
        "retrieval_k": 5,
        "original_query": user_query,
        "laqa_enabled": False
    }


def public_settings():

    data = get_settings()

    return {
        "enable_rag": data["enable_rag"],
        "enable_laqa": data["enable_laqa"],
        "enable_mrl": data["enable_mrl"],
        "active_database": data.get("active_database", "mrl")
    }
