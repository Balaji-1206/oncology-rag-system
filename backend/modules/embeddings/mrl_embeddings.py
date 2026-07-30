import os
import time
import numpy as np
from threading import Lock
from sentence_transformers import (
    SentenceTransformer
)

import settings


import pickle

# =========================================================
# GLOBAL MODEL & PERSISTENT DISK CACHE
# =========================================================
_model = None
_cache_lock = Lock()
MAX_CACHE_SIZE = 50000

MRL_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
CACHE_FILE_PATH = os.path.join(settings.BACKEND_DIR, "database", "embedding_cache.pkl")


def _load_disk_cache():
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"[CACHE] Warning: Failed to load disk cache ({e})")
    return {}


def _save_disk_cache(cache_data):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        with open(CACHE_FILE_PATH, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[CACHE] Warning: Failed to save disk cache ({e})")


_embedding_cache = _load_disk_cache()
if _embedding_cache:
    print(f"[CACHE] Loaded {len(_embedding_cache)} cached embeddings from disk.")


# =========================================================
# LOAD MODEL
# =========================================================
def get_model():

    global _model

    if _model is None:

        print(
            "Loading embedding model ONCE..."
        )

        os.environ[
            "TOKENIZERS_PARALLELISM"
        ] = "false"

        import torch
        # Cap PyTorch CPU threads to max 2 to prevent hardware power spikes
        try:
            torch.set_num_threads(min(2, max(1, (os.cpu_count() or 4) // 2)))
        except Exception:
            pass

        _device = "cuda" if torch.cuda.is_available() else "cpu"

        if _device == "cuda":
            print(f"⚡ [DEVICE] Using GPU acceleration: {torch.cuda.get_device_name(0)} (CUDA)")
        else:
            print("💻 [DEVICE] Running in CPU mode (safe 2-thread cap enabled)")

        _model = SentenceTransformer(
            MRL_MODEL_NAME,
            device=_device,
            trust_remote_code=True
        )

        if _device == "cuda":
            # FP16 Half Precision halves GPU VRAM usage and enables Tensor Core acceleration
            _model.half()

    return _model


# =========================================================
# MRL EMBEDDINGS
# =========================================================
def get_mrl_embedding(
    texts,
    dim=None,
    use_cache=True,
    show_progress_bar=False,
    log=True
):

    model = get_model()

    if isinstance(texts, str):

        texts = [texts]

    texts = list(texts)

    requested_dim = dim

    if requested_dim is None:

        requested_dim = settings.effective_embedding_dimension()

    total = len(texts)

    if total == 0:

        return np.empty(
            (0, requested_dim),
            dtype=np.float32
        )

    cached = {}

    missing = []

    missing_keys = []

    if use_cache:

        with _cache_lock:

            for text in texts:

                key = (
                    requested_dim,
                    text
                )

                if key in _embedding_cache:

                    cached[key] = _embedding_cache[key]

                else:

                    missing.append(text)

                    missing_keys.append(key)

    else:

        missing = texts

        missing_keys = [
            (
                requested_dim,
                text
            )
            for text in texts
        ]

    if log and missing:

        print(
            f"Embedding uncached texts: {len(missing)} "
            f"(requested={total}, dim={requested_dim})"
        )

    start = time.time()

    if missing:
        import torch

        # Prevent GPU memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Use conservative batch size (16 for CPU, 32 for GPU) to keep hardware thermal load low
        safe_batch_size = 32 if torch.cuda.is_available() else 16

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.inference_mode():
                encoded = model.encode(
                    missing,
                    batch_size=safe_batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
        except Exception as e:
            print(f"[EMBEDDINGS] Memory pressure warning: {e}. Retrying with batch_size=8...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.inference_mode():
                encoded = model.encode(
                    missing,
                    batch_size=8,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

        if encoded.shape[1] < requested_dim:

            raise ValueError(
                f"Requested dimension {requested_dim}, "
                f"but model returned {encoded.shape[1]} dimensions"
            )

        encoded = encoded[:, :requested_dim]

        # Re-normalize vector after slicing to guarantee exact unit L2 length for MRL dot-product
        norms = np.linalg.norm(encoded, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        encoded = encoded / norms

        encoded = np.array(
            encoded,
            dtype=np.float32
        )

        if use_cache:

            with _cache_lock:

                for key, vector in zip(
                    missing_keys,
                    encoded
                ):

                    key = (
                        requested_dim,
                        key[1]
                    )

                    # Evict oldest entry if cache capacity reached
                    if len(_embedding_cache) >= MAX_CACHE_SIZE:
                        first_key = next(iter(_embedding_cache))
                        _embedding_cache.pop(first_key, None)

                    _embedding_cache[key] = vector

                    cached[key] = vector

                _save_disk_cache(_embedding_cache)

        else:

            for key, vector in zip(
                missing_keys,
                encoded
            ):

                key = (
                    requested_dim,
                    key[1]
                )

                cached[key] = vector


    vectors = [
        cached[
            (
                requested_dim,
                text
            )
        ]
        for text in texts
    ]

    embeddings = np.vstack(
        vectors
    ).astype(
        np.float32,
        copy=False
    )

    elapsed = round(
        time.time() - start,
        2
    )

    if log:

        print(
            f"Embeddings ready in {elapsed}s"
        )

        print(
            f"Final embedding shape: {embeddings.shape}"
        )

    return embeddings


def prime_mrl_embedding_cache(
    texts,
    dim=None
):

    return get_mrl_embedding(
        texts,
        dim=dim,
        use_cache=True,
        show_progress_bar=False,
        log=False
    )


# =========================================================
# DYNAMIC MRL
# =========================================================
def get_dynamic_mrl_embedding(
    texts,
    intent="factual"
):

    active_dim = settings.effective_embedding_dimension()

    if settings.is_mrl_enabled():

        print(
            f"Using MRL dimension: {active_dim}"
        )

    else:

        print(
            f"MRL disabled: using full embedding dimension {active_dim}"
        )

    return get_mrl_embedding(
        texts,
        dim=active_dim,
        use_cache=True,
        show_progress_bar=False,
        log=False
    )
