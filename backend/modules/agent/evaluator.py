import requests
import json
import re
import numpy as np
import settings
from sentence_transformers import (
    util
)

from modules.embeddings.mrl_embeddings import (
    get_mrl_embedding
)

PHI3MINI_URL = "http://localhost:11434/api/generate"

EVAL_MODEL = "phi3:mini"


# =========================================================
# 🔹 JSON EXTRACTION
# =========================================================
def extract_json(text):

    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            "No valid JSON found"
        )

    return json.loads(
        match.group(0)
    )


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
# 🔹 LEXICAL GROUNDING
# =========================================================
def grounding_score(
    answer,
    context
):

    answer_words = tokenize(answer)

    context_words = tokenize(context)

    if not answer_words:
        return 0

    overlap = len(
        answer_words & context_words
    )

    return overlap / max(
        len(answer_words),
        1
    )


# =========================================================
# 🔹 SEMANTIC GROUNDING
# =========================================================
def semantic_grounding_score(
    answer,
    context
):

    if (
        not answer.strip()
        or
        not context.strip()
    ):

        return 0.0

    try:

        sentences = [

            s.strip()

            for s in re.split(
                r'(?<=[.!?])\s+',
                context
            )

            if len(s.strip()) > 30
        ]

        if not sentences:

            sentences = [
                context[:1200]
            ]

        sentences = sentences[:10]

        embeddings = get_mrl_embedding(
            [answer[:1200]] + sentences,
            log=False
        )

        answer_embedding = embeddings[0]

        context_embeddings = embeddings[1:]

        similarities = util.cos_sim(
            answer_embedding,
            context_embeddings
        )[0]

        top_scores = similarities.topk(
            min(3, len(sentences))
        ).values

        return float(
            np.mean(
                top_scores.cpu().numpy()
            )
        )

    except Exception as e:

        print(
            "⚠️ Semantic grounding failed:",
            e
        )

        return 0.0


# =========================================================
# 🔹 COMBINED GROUNDING
# =========================================================
def combined_grounding_score(
    answer,
    context
):

    lexical = grounding_score(
        answer,
        context
    )

    semantic = semantic_grounding_score(
        answer,
        context
    )

    combined = (
        0.70 * lexical
        +
        0.30 * semantic
    )

    return (
        combined,
        lexical,
        semantic
    )


# =========================================================
# 🔹 REFUSAL DETECTION
# =========================================================
def refusal_detected(answer):

    refusal_patterns = [

        "not enough information",

        "insufficient information",

        "cannot determine",

        "not available in context",
    ]

    answer_lower = answer.lower()

    for p in refusal_patterns:

        if p in answer_lower:
            return True

    return False


# =========================================================
# 🔹 SAFE NORMALIZATION
# =========================================================
def safe_float(v, default=0.5):

    try:
        return float(v)

    except:
        return default


def safe_int(v, default=5):

    try:
        return int(v)

    except:
        return default


# =========================================================
# 🔹 QUERY REQUIREMENTS
# =========================================================
def query_requires_coverage(query):

    q = query.lower()

    triggers = [

        "list",
        "rank",
        "ranking",
        "types",
        "type of",
        "top",
        "common",
        "most common",
        "classification",
        "categories"
    ]

    return any(
        trigger in q
        for trigger in triggers
    )


def query_requires_ranking(query):

    q = query.lower()

    triggers = [

        "rank",
        "ranking",
        "top",
        "most common",
        "least common"
    ]

    return any(
        trigger in q
        for trigger in triggers
    )


# =========================================================
# 🔹 COUNT ITEMS
# =========================================================
def count_answer_items(answer):

    numbered = re.findall(
        r"(?:^|\n)\s*\d+[\.\)]\s+",
        answer
    )

    bullets = re.findall(
        r"(?:^|\n)\s*[-*]\s+",
        answer
    )

    if numbered or bullets:
        return len(numbered) + len(bullets)

    parts = re.split(
        r",|;|\band\b",
        answer
    )

    return len([
        part for part in parts
        if len(part.strip()) > 3
    ])


# =========================================================
# 🔹 ORDER CHECK
# =========================================================
def has_ordered_output(answer):

    if re.search(
        r"(?:^|\n)\s*\d+[\.\)]\s+",
        answer
    ):
        return True

    ordered_terms = [

        "first",
        "second",
        "third",
        "highest",
        "lowest",
        "most common",
        "least common"
    ]

    answer_lower = answer.lower()

    return any(
        term in answer_lower
        for term in ordered_terms
    )


# =========================================================
# 🔹 WEAK SUMMARY PENALTY
# =========================================================
def weak_summary_penalty(answer):

    answer_lower = answer.lower()

    weak_phrases = [

        "the context discusses",
        "the context describes",
        "based on the provided context",
        "based on the context",
        "the provided context",
        "this context",
        "it discusses",
        "it mentions"
    ]

    penalty = 0.0

    for phrase in weak_phrases:

        if phrase in answer_lower:
            penalty += 0.12

    vague_terms = [

        "various",
        "several",
        "some",
        "many",
        "may include",
        "can include",
        "related to"
    ]

    vague_hits = sum(
        1 for term in vague_terms
        if term in answer_lower
    )

    if vague_hits >= 2:
        penalty += 0.08

    if len(tokenize(answer)) < 12:
        penalty += 0.10

    return min(
        penalty,
        0.35
    )


# =========================================================
# 🔹 DIRECT ANSWER SCORE
# =========================================================
def direct_answer_score(
    query,
    answer
):

    sentences = [

        s.strip()

        for s in re.split(
            r'(?<=[.!?])\s+',
            answer
        )

        if s.strip()
    ]

    if not sentences:
        return 0.0

    query_terms = tokenize(query) - {

        "what",
        "which",
        "list",
        "rank",
        "types",
        "type",
        "common",
        "top",
        "the",
        "and",
        "of",
        "for"
    }

    if not query_terms:
        return 0.5

    first_terms = tokenize(
        sentences[0]
    )

    overlap = len(
        query_terms & first_terms
    )

    return overlap / max(
        len(query_terms),
        1
    )


# =========================================================
# 🔹 ENTITY EXTRACTION
# =========================================================
def extract_medical_entities(text):

    text_lower = text.lower()

    patterns = [

        r"\b[a-z]+ carcinoma\b",
        r"\b[a-z]+ sarcoma\b",
        r"\b[a-z]+ lymphoma\b",
        r"\b[a-z]+ leukemia\b",
        r"\b[a-z]+ melanoma\b",
        r"\b[a-z]+ myeloma\b",
        r"\b[a-z]+ cancer\b",
        r"\b[a-z]+ tumou?r\b",
        r"\bcarcinoma\b",
        r"\bsarcoma\b",
        r"\blymphoma\b",
        r"\bleukemia\b",
        r"\bmelanoma\b",
        r"\bmyeloma\b",
        r"\bglioma\b",
        r"\bglioblastoma\b",
        r"\badenocarcinoma\b"
    ]

    entities = set()

    for pattern in patterns:

        entities.update(
            re.findall(
                pattern,
                text_lower
            )
        )

    return {
        entity.strip()
        for entity in entities
        if entity.strip()
    }


# =========================================================
# 🔹 COVERAGE GAP
# =========================================================
def coverage_gap(
    query,
    context,
    answer
):

    if not query_requires_coverage(query):
        return False

    context_entities = extract_medical_entities(
        context
    )

    answer_entities = extract_medical_entities(
        answer
    )

    if (
        len(context_entities) >= 4
        and
        len(answer_entities) <= 2
    ):
        return True

    if (
        len(context_entities) >= 5
        and
        len(answer_entities)
        /
        max(len(context_entities), 1)
        < 0.4
    ):
        return True

    return False


# =========================================================
# 🔹 REPETITION PENALTY
# =========================================================
def repetition_penalty(answer):

    words = answer.lower().split()

    if not words:
        return 0.0

    unique_ratio = len(set(words)) / len(words)

    if unique_ratio < 0.45:
        return 0.15

    return 0.0


# =========================================================
# 🔹 DIRECT EVALUATION PROMPT
# =========================================================
def build_direct_evaluation_prompt(
    query,
    answer
):

    return f"""
You are an evaluator for a direct oncology medical answer.

STRICT RULES:
- No retrieved documents, context, or grounding evidence are available
- Evaluate only the answer itself and its relevance to the question
- Penalize vague, incomplete, or off-target answers
- Do not assume retrieval or context
- Do not use grounding-based judgments
- Evaluate 'score' as an integer from 1 to 10 rating the overall quality and medical correctness of the answer (10 is perfect, 1 is completely wrong/empty/refusal)

Return ONLY valid JSON.

JSON FORMAT:

{{
  "score": 0,
  "confidence": 0.0,
  "needs_retry": false,
  "answered_question": true,
  "answer_relevance": 0.0,
  "hallucination_risk": "low",
  "missing_information": false,
  "contradiction_risk": 0.0
}}

QUESTION:
{query}

ANSWER:
{answer[:1800]}
"""


# =========================================================
# 🔹 MAIN EVALUATOR
# =========================================================
def evaluate_answer(
    query,
    context,
    answer
):

    if not settings.is_rag_enabled():

        prompt = build_direct_evaluation_prompt(
            query,
            answer
        )

        try:

            response = requests.post(
                PHI3MINI_URL,
                json={

                    "model": EVAL_MODEL,

                    "prompt": prompt,

                    "stream": False,

                    "format": "json",

                    "options": {

                        "temperature": 0,

                        "top_p": 0.2,

                        "top_k": 20,

                        "repeat_penalty": 1.05,

                        "num_predict": 120,

                        "num_ctx": 4096,

                        "keep_alive": "30m"
                    }
                },
                timeout=60
            )

            data = response.json()

            raw_output = data.get(
                "response",
                ""
            ).strip()

            print("\n🧠 EVAL RAW OUTPUT:\n")
            print(raw_output)

            parsed = extract_json(
                raw_output
            )

            score = safe_int(
                parsed.get("score", 5)
            )

            confidence = safe_float(
                parsed.get("confidence", 0.5)
            )

            answer_relevance = safe_float(
                parsed.get(
                    "answer_relevance",
                    0.5
                )
            )

            answered_question = bool(
                parsed.get(
                    "answered_question",
                    True
                )
            )

            hallucination_risk = str(
                parsed.get(
                    "hallucination_risk",
                    "medium"
                )
            ).lower()

            missing_information = bool(
                parsed.get(
                    "missing_information",
                    False
                )
            )

            score = max(
                0,
                min(score, 10)
            )

            confidence = max(
                0,
                min(confidence, 1)
            )

            answer_relevance = max(
                0,
                min(answer_relevance, 1)
            )

            if hallucination_risk not in [
                "low",
                "medium",
                "high"
            ]:

                hallucination_risk = "medium"

            contradiction = 0.0

            refusal = refusal_detected(
                answer
            )

            weak_penalty = weak_summary_penalty(
                answer
            )

            directness = direct_answer_score(
                query,
                answer
            )

            needs_coverage = query_requires_coverage(
                query
            )

            needs_ranking = query_requires_ranking(
                query
            )

            item_count = count_answer_items(
                answer
            )

            insufficient_structure = (
                needs_coverage
                and item_count < 3
            )

            missing_ranking = (
                needs_ranking
                and not has_ordered_output(answer)
            )

            repeat_pen = repetition_penalty(
                answer
            )

            answer_relevance = max(
                0,
                answer_relevance - repeat_pen
            )

            if weak_penalty:

                answer_relevance = max(
                    0.0,
                    answer_relevance - weak_penalty
                )

                confidence = max(
                    0.0,
                    confidence - (
                        weak_penalty * 0.8
                    )
                )

                score = min(score, 6)

            if directness < 0.20:

                answer_relevance = min(
                    answer_relevance,
                    0.70
                )

                confidence = min(
                    confidence,
                    0.60
                )

            if (
                insufficient_structure
                or missing_ranking
            ):

                missing_information = True

                answer_relevance = min(
                    answer_relevance,
                    0.52
                )

                confidence = min(
                    confidence,
                    0.58
                )

                score = min(score, 5)

            retry = False

            if hallucination_risk == "high":
                retry = True

            elif not answered_question:
                retry = True

            elif answer_relevance < 0.50:
                retry = True

            elif score <= 3:
                retry = True

            if (
                insufficient_structure
                or missing_ranking
            ):

                retry = True

            if refusal:

                retry = True

                score = min(score, 4)

            if (
                score >= 8
                and confidence >= 0.75
                and answer_relevance >= 0.75
                and contradiction <= 0.25
                and not missing_information
                and not insufficient_structure
                and not missing_ranking
            ):

                retry = False

            if (
                confidence > 0.80
                and (
                    answer_relevance < 0.80
                    or contradiction > 0.20
                    or missing_information
                )
            ):

                confidence = 0.80

            return {

                "score": score,

                "confidence": round(
                    confidence,
                    2
                ),

                "needs_retry": retry,

                "answered_question": answered_question,

                "answer_relevance": round(
                    answer_relevance,
                    2
                ),

                "hallucination_risk": hallucination_risk,

                "missing_information": missing_information,

                "grounding_score": 0.0,

                "retrieval_score": 0.0,

                "lexical_grounding_score": 0.0,

                "semantic_grounding_score": 0.0,

                "contradiction_risk": 0.0,

                "refusal_detected": refusal
            }

        except Exception as e:

            print("❌ EVAL ERROR:", e)

            return {

                "score": 4,

                "confidence": 0.4,

                "needs_retry": True,

                "answered_question": False,

                "answer_relevance": 0.35,

                "hallucination_risk": "medium",

                "missing_information": True,

                "grounding_score": 0.0,

                "retrieval_score": 0.0,

                "lexical_grounding_score": 0.0,

                "semantic_grounding_score": 0.0,

                "contradiction_risk": 0.0,

                "refusal_detected": False
            }

    prompt = f"""
You are an evaluator for a medical oncology RAG system.

STRICT RULES:
- Be strict and realistic
- Penalize vague summaries
- Penalize unsupported synthesis
- Penalize incomplete lists
- Penalize missing rankings
- Penalize generic educational filler
- Require direct answers early
- High confidence requires strong grounding
- Evaluate contradiction_risk as a float from 0.0 (no contradiction) to 1.0 (direct contradiction with context)
- Evaluate 'score' as an integer from 1 to 10 rating the overall quality and medical correctness of the answer based on the context (10 is perfect, 1 is completely wrong/empty/refusal)

Return ONLY valid JSON.

JSON FORMAT:

{{
  "score": 0,
  "confidence": 0.0,
  "needs_retry": false,
  "answered_question": true,
  "answer_relevance": 0.0,
  "hallucination_risk": "low",
  "missing_information": false,
  "contradiction_risk": 0.0
}}

QUESTION:
{query}

CONTEXT:
{context[:3500]}

ANSWER:
{answer[:1800]}
"""

    try:

        response = requests.post(
            PHI3MINI_URL,
            json={

                "model": EVAL_MODEL,

                "prompt": prompt,

                "stream": False,

                "format": "json",

                "options": {

                    "temperature": 0,

                    "top_p": 0.2,

                    "top_k": 20,

                    "repeat_penalty": 1.05,

                    "num_predict": 120,

                    "num_ctx": 4096,

                    "keep_alive": "30m"
                }
            },
            timeout=60
        )

        data = response.json()

        raw_output = data.get(
            "response",
            ""
        ).strip()

        print("\n🧠 EVAL RAW OUTPUT:\n")
        print(raw_output)

        parsed = extract_json(
            raw_output
        )

        # =====================================================
        # 🔹 NORMALIZATION
        # =====================================================
        score = safe_int(
            parsed.get("score", 5)
        )

        confidence = safe_float(
            parsed.get("confidence", 0.5)
        )

        answer_relevance = safe_float(
            parsed.get(
                "answer_relevance",
                0.5
            )
        )

        answered_question = bool(
            parsed.get(
                "answered_question",
                True
            )
        )

        hallucination_risk = str(
            parsed.get(
                "hallucination_risk",
                "medium"
            )
        ).lower()

        missing_information = bool(
            parsed.get(
                "missing_information",
                False
            )
        )

        # =====================================================
        # 🔹 SAFE CLAMPING
        # =====================================================
        score = max(
            0,
            min(score, 10)
        )

        confidence = max(
            0,
            min(confidence, 1)
        )

        answer_relevance = max(
            0,
            min(answer_relevance, 1)
        )

        if hallucination_risk not in [
            "low",
            "medium",
            "high"
        ]:

            hallucination_risk = "medium"

        # =====================================================
        # 🔹 GROUNDING
        # =====================================================
        grounding, lexical_grounding, semantic_grounding = combined_grounding_score(
            answer,
            context
        )

        contradiction = safe_float(
            parsed.get(
                "contradiction_risk",
                0.0
            )
        )

        refusal = refusal_detected(
            answer
        )

        weak_penalty = weak_summary_penalty(
            answer
        )

        directness = direct_answer_score(
            query,
            answer
        )

        needs_coverage = query_requires_coverage(
            query
        )

        needs_ranking = query_requires_ranking(
            query
        )

        item_count = count_answer_items(
            answer
        )

        insufficient_structure = (
            needs_coverage
            and item_count < 3
        )

        missing_ranking = (
            needs_ranking
            and not has_ordered_output(answer)
        )

        has_coverage_gap = coverage_gap(
            query,
            context,
            answer
        )

        repeat_pen = repetition_penalty(
            answer
        )

        # =====================================================
        # 🔹 PENALTIES
        # =====================================================
        answer_relevance = max(
            0,
            answer_relevance - repeat_pen
        )

        if weak_penalty:

            answer_relevance = max(
                0.0,
                answer_relevance - weak_penalty
            )

            confidence = max(
                0.0,
                confidence - (
                    weak_penalty * 0.8
                )
            )

            score = min(score, 6)

        if directness < 0.20:

            answer_relevance = min(
                answer_relevance,
                0.70
            )

            confidence = min(
                confidence,
                0.60
            )

        if (
            insufficient_structure
            or missing_ranking
            or has_coverage_gap
        ):

            missing_information = True

            answer_relevance = min(
                answer_relevance,
                0.52
            )

            confidence = min(
                confidence,
                0.58
            )

            score = min(score, 5)

        if grounding < 0.40:

            confidence = min(
                confidence,
                0.60
            )

        if grounding < 0.25:

            answer_relevance = min(
                answer_relevance,
                0.52
            )

        # =====================================================
        # 🔹 RELEVANCE CALIBRATION
        # =====================================================
        answer_relevance = min(
            answer_relevance,
            grounding + 0.30
        )

        # =====================================================
        # 🔹 RETRY LOGIC
        # =====================================================
        retry = False

        if hallucination_risk == "high":
            retry = True

        elif not answered_question:
            retry = True

        elif answer_relevance < 0.50:
            retry = True

        elif grounding < 0.25:
            retry = True

        elif contradiction > 0.6:
            retry = True

        elif score <= 3:
            retry = True

        if (
            insufficient_structure
            or missing_ranking
            or has_coverage_gap
        ):

            retry = True

        if refusal and grounding > 0.15:

            retry = True

            score = min(score, 4)

        # =====================================================
        # 🔹 HIGH QUALITY OVERRIDE
        # =====================================================
        if (
            score >= 8
            and confidence >= 0.75
            and answer_relevance >= 0.75
            and grounding >= 0.55
            and contradiction <= 0.25
            and not missing_information
            and not insufficient_structure
            and not missing_ranking
            and not has_coverage_gap
        ):

            retry = False

        # =====================================================
        # 🔹 CONFIDENCE CALIBRATION
        # =====================================================
        if (
            confidence > 0.80
            and (
                grounding < 0.70
                or answer_relevance < 0.80
                or contradiction > 0.20
                or missing_information
            )
        ):

            confidence = 0.80

        # =====================================================
        # 🔹 RETRIEVAL SCORE
        # =====================================================
        retrieval_score = (

            0.55 * grounding

            +

            0.25 * answer_relevance

            +

            0.20 * confidence
        )

        retrieval_score = max(
            0.0,
            min(
                1.0,
                retrieval_score
            )
        )

        # =====================================================
        # 🔹 FINAL OUTPUT
        # =====================================================
        result = {

            "score": score,

            "confidence": round(
                confidence,
                2
            ),

            "needs_retry": retry,

            "answered_question": answered_question,

            "answer_relevance": round(
                answer_relevance,
                2
            ),

            "hallucination_risk": hallucination_risk,

            "missing_information": missing_information,

            "grounding_score": round(
                grounding,
                2
            ),

            "retrieval_score": round(
                retrieval_score,
                2
            ),

            "lexical_grounding_score": round(
                lexical_grounding,
                2
            ),

            "semantic_grounding_score": round(
                semantic_grounding,
                2
            ),

            "contradiction_risk": round(
                contradiction,
                2
            ),

            "refusal_detected": refusal
        }

        return result

    except Exception as e:

        print("❌ EVAL ERROR:", e)

        return {

            "score": 4,

            "confidence": 0.4,

            "needs_retry": True,

            "answered_question": False,

            "answer_relevance": 0.35,

            "hallucination_risk": "medium",

            "missing_information": True,

            "grounding_score": 0.1,

            "retrieval_score": 0.1,

            "lexical_grounding_score": 0.1,

            "semantic_grounding_score": 0.0,

            "contradiction_risk": 0.2,

            "refusal_detected": False
        }
