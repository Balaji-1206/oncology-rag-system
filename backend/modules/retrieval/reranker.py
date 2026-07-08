from sentence_transformers import CrossEncoder
import numpy as np
import re

# =========================================================
# 🔹 LOAD ONCE
# =========================================================
print("🔥 Loading reranker model ONCE...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-large",
    device="cuda"
)


# =========================================================
# 🔹 TOKENIZER
# =========================================================
def tokenize(text):

    return set(
        re.findall(
            r"\b[a-zA-Z0-9\-]+\b",
            text.lower()
        )
    )


# =========================================================
# 🔹 DIVERSITY FILTER
# =========================================================
def diversify(
    ranked_docs,
    max_similarity=0.70
):

    final_docs = []

    seen = []

    for doc, score in ranked_docs:

        tokens = tokenize(doc[:400])

        duplicate = False

        for prev in seen:

            overlap = len(tokens & prev)

            similarity = overlap / max(
                len(tokens),
                1
            )

            if similarity > max_similarity:

                duplicate = True
                break

        if duplicate:
            continue

        final_docs.append((doc, score))

        seen.append(tokens)

    return final_docs


# =========================================================
# 🔹 QUERY OVERLAP BOOST
# =========================================================
def query_overlap_boost(
    query,
    doc
):

    q_tokens = tokenize(query)

    d_tokens = tokenize(doc)

    overlap = len(
        q_tokens & d_tokens
    )

    score = overlap / max(
        len(q_tokens),
        1
    )

    return min(
        score,
        0.35
    )


# =========================================================
# 🔹 DEFINITION BOOST
# =========================================================
def definition_boost(
    query,
    doc
):

    q = query.lower()

    if (
        "what is" in q
        or "define" in q
    ):

        patterns = [

            "is a disease",

            "defined as",

            "refers to",

            "characterized by",

            "condition in which"
        ]

        d = doc.lower()

        for p in patterns:

            if p in d:
                return 0.18

    return 0.0


# =========================================================
# 🔹 NOISE PENALTY
# =========================================================
def noise_penalty(doc):

    bad_patterns = [

        "confidence interval",

        "statistically significant",

        "study population",

        "retrospective study",

        "prospective study",

        "p-value",

        "hazard ratio",

        "prevalence",

        "incidence"
    ]

    text = doc.lower()

    penalty = 0.0

    for p in bad_patterns:

        if p in text:
            penalty += 0.08

    return min(
        penalty,
        0.25
    )


# =========================================================
# 🔹 LENGTH PENALTY
# =========================================================
def length_penalty(doc):

    length = len(doc.split())

    if length > 350:
        return 0.12

    if length > 250:
        return 0.06

    return 0.0


# =========================================================
# 🔹 SCORE NORMALIZATION
# =========================================================
def normalize_scores(scores):

    return scores  # Deprecated: CrossEncoder outputs calibrated logits.


# =========================================================
# 🔹 CONFIDENCE CALIBRATION
# =========================================================
def calibrate_confidence(score):

    calibrated = 1 / (
        1 +
        np.exp(-score)
    )

    calibrated = float(calibrated)

    calibrated = max(
        0.05,
        min(calibrated, 0.95)
    )

    return calibrated


# =========================================================
# 🔹 RERANK
# =========================================================
def rerank(
    query,
    docs,
    top_k=5,
    return_scores=False
):

    if not docs:

        if return_scores:
            return [], []

        return []

    # =====================================================
    # 🔹 REMOVE EMPTY DOCS
    # =====================================================
    docs = [

        d.strip()

        for d in docs

        if d and d.strip()
    ]

    if not docs:

        if return_scores:
            return [], []

        return []

    # =====================================================
    # 🔹 QUERY-DOC PAIRS
    # =====================================================
    pairs = [

        [query, doc]

        for doc in docs
    ]

    # =====================================================
    # 🔹 CROSS ENCODER
    # =====================================================
    scores = reranker.predict(
        pairs,
        batch_size=8
    )

    # =====================================================
    # 🔹 HYBRID RERANK SCORE
    # =====================================================
    ranked = []

    for doc, base_score in zip(
        docs,
        scores
    ):

        penalty = (

            noise_penalty(doc)

            +

            length_penalty(doc)
        )

        final_score = (

            float(base_score)

            -

            penalty
        )

        ranked.append(
            (doc, final_score)
        )

    # =====================================================
    # 🔹 SORT
    # =====================================================
    ranked = sorted(
        ranked,
        key=lambda x: x[1],
        reverse=True
    )

    # =====================================================
    # 🔹 REMOVE VERY WEAK DOCS
    # =====================================================
    filtered = []

    for doc, score in ranked:

        if score >= -2.0:

            filtered.append(
                (doc, score)
            )

    if not filtered:

        filtered = ranked[:top_k]

    # =====================================================
    # 🔹 DIVERSITY FILTER
    # =====================================================
    filtered = diversify(
        filtered
    )

    # =====================================================
    # 🔹 FINAL TOP-K
    # =====================================================
    filtered = filtered[:top_k]

    final_docs = [

        doc

        for doc, score

        in filtered
    ]

    # =====================================================
    # 🔹 SCORES
    # =====================================================
    if return_scores:

        final_scores = [

            round(
                calibrate_confidence(score),
                3
            )

            for _, score

            in filtered
        ]

        return final_docs, final_scores

    return final_docs