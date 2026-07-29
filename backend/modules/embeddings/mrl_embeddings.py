import os
import time
import numpy as np
from threading import Lock
from sentence_transformers import (
    SentenceTransformer
)

import settings


# =========================================================
# GLOBAL MODEL
# =========================================================
_model = None
_embedding_cache = {}
_cache_lock = Lock()
MAX_CACHE_SIZE = 10000

MRL_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


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
        _device = "cuda" if torch.cuda.is_available() else "cpu"

        _model = SentenceTransformer(
            MRL_MODEL_NAME,
            device=_device,
            trust_remote_code=True
        )

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

        encoded = model.encode(
            missing,
            batch_size=64,
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
