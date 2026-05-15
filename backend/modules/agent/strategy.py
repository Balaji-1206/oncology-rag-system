def choose_strategy(
    eval_result,
    attempt
):

    score = eval_result.get(
        "score",
        5
    )

    needs_retry = eval_result.get(
        "needs_retry",
        True
    )

    answered_question = eval_result.get(
        "answered_question",
        True
    )

    answer_relevance = eval_result.get(
        "answer_relevance",
        0.5
    )

    hallucination_risk = eval_result.get(
        "hallucination_risk",
        "medium"
    )

    missing_information = eval_result.get(
        "missing_information",
        False
    )

    grounding_score = eval_result.get(
        "grounding_score",
        0.3
    )

    contradiction_risk = eval_result.get(
        "contradiction_risk",
        0.0
    )

    refusal_detected = eval_result.get(
        "refusal_detected",
        False
    )

    confidence = eval_result.get(
        "confidence",
        0.5
    )

    retrieval_score = eval_result.get(
        "retrieval_score",
        0.4
    )

    reranker_confidence = eval_result.get(
        "reranker_confidence",
        0.0
    )

    lexical_grounding = eval_result.get(
        "lexical_grounding_score",
        grounding_score
    )

    semantic_grounding = eval_result.get(
        "semantic_grounding_score",
        grounding_score
    )

    # =====================================================
    # 🔹 STRICT ACCEPTANCE
    # =====================================================
    if (

        not needs_retry

        and score >= 8

        and answered_question

        and hallucination_risk == "low"

        and answer_relevance >= 0.75

        and grounding_score >= 0.55

        and lexical_grounding >= 0.45

        and contradiction_risk < 0.20

        and confidence >= 0.72

        and retrieval_score >= 0.55

        and reranker_confidence >= 0.40

        and not missing_information

    ):

        return "accept"

    # =====================================================
    # 🔹 SAFE REFUSAL ACCEPT
    # =====================================================
    if refusal_detected:

        if retrieval_score < 0.30:

            return "accept"

        if grounding_score < 0.20:

            return "accept"

    # =====================================================
    # 🔹 HIGH HALLUCINATION RISK
    # =====================================================
    if hallucination_risk == "high":

        return "increase_k"

    # =====================================================
    # 🔹 CONTRADICTION DETECTED
    # =====================================================
    if contradiction_risk > 0.45:

        return "increase_k"

    # =====================================================
    # 🔹 VERY LOW GROUNDING
    # =====================================================
    if grounding_score < 0.25:

        return "expand_query"

    # =====================================================
    # 🔹 QUESTION NOT ANSWERED
    # =====================================================
    if not answered_question:

        return "expand_query"

    # =====================================================
    # 🔹 WEAK RETRIEVAL
    # =====================================================
    if retrieval_score < 0.40:

        return "increase_k"

    # =====================================================
    # 🔹 RERANKER UNCERTAINTY
    # =====================================================
    if (

        reranker_confidence < 0.30

        and retrieval_score < 0.60

    ):

        return "increase_k"

    # =====================================================
    # 🔹 LOW RELEVANCE
    # =====================================================
    if answer_relevance < 0.60:

        return "expand_query"

    # =====================================================
    # 🔹 MISSING INFORMATION
    # =====================================================
    if missing_information:

        return "expand_query"

    # =====================================================
    # 🔹 LOW CONFIDENCE
    # =====================================================
    if confidence < 0.45:

        return "increase_k"

    # =====================================================
    # 🔹 LOW SCORE
    # =====================================================
    if score <= 4:

        return "expand_query"

    # =====================================================
    # 🔹 WEAK LEXICAL GROUNDING
    # =====================================================
    if lexical_grounding < 0.30:

        return "expand_query"

    # =====================================================
    # 🔹 SEMANTIC ONLY HALLUCINATION RISK
    # =====================================================
    if (

        semantic_grounding > 0.70

        and lexical_grounding < 0.20

    ):

        return "expand_query"

    # =====================================================
    # 🔹 RETRY LIMIT
    # =====================================================
    if attempt >= 2:

        return "accept"

    # =====================================================
    # 🔹 DEFAULT FALLBACK
    # =====================================================
    if needs_retry:

        return "increase_k"

    return "accept"