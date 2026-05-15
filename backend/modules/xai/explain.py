import re
import requests
import numpy as np

SESSION = requests.Session()


# =========================================================
# 🔹 TOKENIZER
# =========================================================
def tokenize(text):

    return set(
        re.findall(
            r'\b[a-zA-Z]{3,}\b',
            text.lower()
        )
    )


# =========================================================
# 🔹 GROUNDING OVERLAP
# =========================================================
def grounding_overlap(
    answer,
    docs
):

    context = " ".join(docs).lower()

    answer_words = tokenize(answer)

    if not answer_words:
        return 0

    overlap = sum(

        1 for w in answer_words

        if f" {w} " in f" {context} "
    )

    return overlap / max(
        len(answer_words),
        1
    )


# =========================================================
# 🔹 CONTRADICTION CHECK
# =========================================================
def contradiction_risk(
    answer,
    docs
):

    negative_patterns = [

        "not associated",

        "no evidence",

        "unclear",

        "not proven",

        "limited evidence",

        "controversial",
    ]

    answer_lower = answer.lower()

    context = " ".join(docs).lower()

    risk = 0

    for p in negative_patterns:

        if p in context and p not in answer_lower:
            risk += 1

    return min(risk / 4, 1)


# =========================================================
# 🔹 SENTENCE CLEANER
# =========================================================
def clean_sentence(sent):

    sent = re.sub(
        r"\s+",
        " ",
        sent
    ).strip()

    sent = re.sub(
        r"\[[^\]]+\]",
        "",
        sent
    )

    sent = re.sub(
        r"\([^)]{0,25}\)",
        "",
        sent
    )

    return sent.strip()


# =========================================================
# 🔹 SENTENCE RELEVANCE
# =========================================================
def sentence_relevance(
    answer_words,
    question_words,
    sentence
):

    sent_words = tokenize(sentence)

    answer_overlap = len(
        answer_words & sent_words
    )

    question_overlap = len(
        question_words & sent_words
    )

    answer_score = answer_overlap / max(
        len(answer_words),
        1
    )

    question_score = question_overlap / max(
        len(question_words),
        1
    )

    # Prefer concise evidence
    length = len(sentence.split())

    if length > 45:
        length_penalty = 0.85

    elif length < 8:
        length_penalty = 0.60

    else:
        length_penalty = 1.0

    return (
        0.55 * answer_score
        +
        0.45 * question_score
    ) * length_penalty


# =========================================================
# 🔹 SUPPORTING SENTENCES
# =========================================================
def extract_supporting_sentences(
    query,
    answer,
    docs
):

    answer_words = tokenize(answer)

    question_words = tokenize(query)

    scored = []

    for doc in docs[:4]:

        sentences = re.split(
            r'(?<=[.!?])\s+',
            doc
        )

        for sent in sentences:

            sent = clean_sentence(
                sent
            )

            if len(sent) < 40:
                continue

            # Ignore noisy academic text
            banned = [

                "et al",

                "study",

                "prevalence",

                "incidence",

                "confidence interval",

                "statistically significant"
            ]

            if any(
                b in sent.lower()
                for b in banned
            ):
                continue

            relevance = sentence_relevance(
                answer_words,
                question_words,
                sent
            )

            if relevance > 0.18:

                scored.append(
                    (relevance, sent)
                )

    scored.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    final = []

    seen = set()

    for _, sent in scored:

        short = sent[:120]

        if short in seen:
            continue

        seen.add(short)

        if len(sent) > 180:
            sent = sent[:180] + "..."

        final.append(sent)

        if len(final) >= 3:
            break

    # Fallback
    if not final:

        for doc in docs[:1]:

            sentences = re.split(
                r'(?<=[.!?])\s+',
                doc
            )

            for sent in sentences:

                sent = clean_sentence(sent)

                if len(sent) > 50:

                    final.append(sent[:180])

                    break

    return final


# =========================================================
# 🔹 CONFIDENCE CALIBRATION
# =========================================================
def calibrate_confidence(
    eval_result,
    grounding_score,
    contradiction_score
):

    eval_conf = eval_result.get(
        "confidence",
        0.5
    )

    retrieval_score = eval_result.get(
        "retrieval_score",
        0.4
    )

    answer_relevance = eval_result.get(
        "answer_relevance",
        0.4
    )

    hallucination = eval_result.get(
        "hallucination_risk",
        "medium"
    )

    confidence = (

        0.30 * eval_conf +

        0.30 * retrieval_score +

        0.25 * grounding_score +

        0.15 * answer_relevance
    )

    # Penalize contradictions
    confidence -= (
        contradiction_score * 0.30
    )

    # Penalize hallucination
    if hallucination == "high":
        confidence *= 0.50

    elif hallucination == "medium":
        confidence *= 0.80

    confidence = max(
        0.05,
        min(confidence, 0.88)
    )

    return round(confidence, 2)


# =========================================================
# 🔹 QUALITY LABEL
# =========================================================
def quality_label(score):

    if score >= 8:
        return "High"

    if score >= 5:
        return "Medium"

    return "Low"


# =========================================================
# 🔹 FAST REASONING
# =========================================================
def fast_reasoning(
    confidence,
    grounding,
    contradiction
):

    if confidence >= 0.82:

        return (
            "- The answer is strongly supported by retrieved oncology evidence.\n"
            "- Retrieved medical context aligns closely with the generated response."
        )

    if grounding >= 0.6:

        return (
            "- The answer is partially grounded in retrieved medical context.\n"
            "- Relevant oncology evidence was identified."
        )

    if contradiction > 0.4:

        return (
            "- Retrieved evidence contains uncertainty or limited support.\n"
            "- The generated answer may not be fully supported by the context."
        )

    return None


# =========================================================
# 🔹 LLM REASONING
# =========================================================
def generate_reasoning(
    query,
    answer,
    docs
):

    context = "\n".join(
        docs[:2]
    )[:1200]

    prompt = f"""
You are validating a medical oncology RAG response.

STRICT RULES:
- ONLY use provided context
- NEVER hallucinate
- Output EXACTLY 2 concise bullet points
- Mention whether evidence supports the answer
- Mention if evidence is incomplete
- No markdown headers
- No XML tags
- No chain-of-thought
- No reasoning traces

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
{answer}

VERIFICATION:
"""

    try:

        response = SESSION.post(
            "http://localhost:11434/api/generate",
            json={

                "model": "phi3:mini",

                "prompt": prompt,

                "stream": False,

                "options": {

                    "temperature": 0,

                    "top_p": 0.1,

                    "num_predict": 80,

                    "keep_alive": "10m"
                }
            },
            timeout=40
        )

        raw = response.json().get(
            "response",
            ""
        ).strip()

        cleaned = re.sub(
            r"<.*?>",
            "",
            raw
        )

        cleaned = cleaned.strip()

        cleaned = re.sub(
            r"\n{3,}",
            "\n",
            cleaned
        )

        if len(cleaned) < 15:

            return (
                "- Retrieved evidence partially supports the answer.\n"
                "- Some medical details may be incomplete."
            )

        return cleaned

    except Exception as e:

        print("⚠️ Reasoning failed:", e)

        return (
            "- Retrieved evidence could not be fully verified.\n"
            "- Supporting oncology context may be incomplete."
        )


# =========================================================
# 🔹 MAIN EXPLAINER
# =========================================================
def generate_explanation(
    answer,
    docs,
    eval_result,
    query
):

    explanation = {}

    # =====================================================
    # 🔹 SUPPORTING EVIDENCE
    # =====================================================
    supporting = extract_supporting_sentences(
        query,
        answer,
        docs
    )

    explanation[
        "supporting_sentences"
    ] = supporting

    # =====================================================
    # 🔹 GROUNDING
    # =====================================================
    grounding_score = grounding_overlap(
        answer,
        docs
    )

    contradiction_score = contradiction_risk(
        answer,
        docs
    )

    # =====================================================
    # 🔹 CONFIDENCE
    # =====================================================
    confidence = calibrate_confidence(
        eval_result,
        grounding_score,
        contradiction_score
    )

    explanation["confidence"] = confidence

    # =====================================================
    # 🔹 QUALITY
    # =====================================================
    score = eval_result.get(
        "score",
        5
    )

    explanation["quality"] = quality_label(
        score
    )

    # =====================================================
    # 🔹 GROUNDED STATUS
    # =====================================================
    explanation["grounded"] = (

        "not enough information"
        not in answer.lower()
    )

    # =====================================================
    # 🔹 FAST REASONING
    # =====================================================
    quick_reasoning = fast_reasoning(
        confidence,
        grounding_score,
        contradiction_score
    )

    if quick_reasoning is not None:

        explanation[
            "reasoning"
        ] = quick_reasoning

        return explanation

    # =====================================================
    # 🔹 LLM REASONING
    # =====================================================
    explanation["reasoning"] = generate_reasoning(
        query,
        answer,
        docs
    )

    return explanation