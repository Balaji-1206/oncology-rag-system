import json
import os
from threading import Lock


# =========================================================
# GLOBAL FEATURE FLAGS - DO NOT EDIT
# =========================================================
# These are defaults only. To change settings, edit:
# -> runtime_settings.json (actual config file)
# Or use: settings.update_settings({'enable_mrl': False})
# =========================================================
ENABLE_LAQA = True
ENABLE_MRL = False

MRL_DIMENSION = 512
FULL_EMBEDDING_DIMENSION = 256
RETRIEVAL_RELEVANCE_THRESHOLD = 0.65

_SETTINGS_PATH = os.path.join(
    os.path.dirname(__file__),
    "runtime_settings.json"
)

_LOCK = Lock()


def _defaults():

    return {
        "enable_laqa": ENABLE_LAQA,
        "enable_mrl": ENABLE_MRL,
        "mrl_dimension": MRL_DIMENSION,
        "full_embedding_dimension": FULL_EMBEDDING_DIMENSION,
        "retrieval_relevance_threshold": RETRIEVAL_RELEVANCE_THRESHOLD
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

    return load_settings()


def update_settings(payload):

    current = load_settings()

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

    save_settings(current)

    return current


def is_laqa_enabled():

    return bool(
        load_settings()["enable_laqa"]
    )


def is_mrl_enabled():

    return bool(
        load_settings()["enable_mrl"]
    )


def effective_embedding_dimension():

    data = load_settings()

    if data["enable_mrl"]:
        return int(data["mrl_dimension"])

    return int(data["full_embedding_dimension"])


def retrieval_relevance_threshold():

    return float(
        load_settings()["retrieval_relevance_threshold"]
    )


def get_database_path():
    """
    Returns the active database path based on MRL setting.
    MRL enabled: returns 'backend/database/mrl/'
    MRL disabled: returns 'backend/database/full/'
    """
    if is_mrl_enabled():
        return "backend/database/mrl"
    return "backend/database/full"


def build_raw_query_payload(user_query):

    return {
        "intent": "factual",
        "query_type": "general",
        "keywords": [],
        "expanded_query": user_query,
        "retrieval_k": 5,
        "original_query": user_query,
        "laqa_enabled": False
    }


def public_settings():

    data = load_settings()

    return {
        "enable_laqa": data["enable_laqa"],
        "enable_mrl": data["enable_mrl"]
    }
