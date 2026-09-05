import os
import re
import requests
import numpy as np
import settings
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

            banned = [

                "et al",

                "study design",

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

        if len(sent) > 220:
            sent = sent[:220] + "..."

        final.append(sent)

        if len(final) >= 4:
            break

    if not final:

        for doc in docs[:1]:

            sentences = re.split(
                r'(?<=[.!?])\s+',
                doc
            )

            for sent in sentences:

                sent = clean_sentence(sent)

                if len(sent) > 50:

                    final.append(sent[:220])

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

    confidence -= (
        contradiction_score * 0.30
    )

    if hallucination == "high":
        confidence *= 0.50

    elif hallucination == "medium":
        confidence *= 0.80

    confidence = max(
        0.05,
        min(confidence, 0.95)
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
# REASONING OUTPUT CLEANING
# =========================================================
REASONING_MODEL = os.environ.get("OLLAMA_MODEL", "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M")

LEAK_PATTERNS = [
    r"\bthought\b",
    r"\banalysis\b",
    r"\bchain[-\s]*of[-\s]*thought\b",
    r"\bthe user wants\b",
    r"\bi need to\b",
    r"\bi should\b",
    r"\bi will\b",
    r"\blet me\b",
    r"\bfollow the instructions\b",
    r"\bstrict output rules\b",
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
    r"\bprompt instructions\b",
    r"\breasoning process\b",
    r"\bstep by step\b",
    r"\bstep-by-step\b",
]

CONVERSATION_PATTERNS = [
    r"\bplease provide\b",
    r"\bcan you provide\b",
    r"\bi can help\b",
    r"\bas an ai\b",
    r"\bi am unable\b",
    r"\bmore context\b",
    r"\bif you\b",
]

PROMPT_ECHO_PATTERNS = [
    r"(?is)question\s*:.*?(?=\n-|\Z)",
    r"(?is)context\s*:.*?(?=\n-|\Z)",
    r"(?is)answer\s*:.*?(?=\n-|\Z)",
    r"(?is)output contract\s*:.*?(?=\n-|\Z)",
    r"(?is)source material\s*:.*?(?=\n-|\Z)",
    r"(?is)clinical reasoning bullets\s*:?",
    r"(?is)oncology_clinical_reasoning_completion",
]


def compact_text(text, limit):

    text = re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip()

    return text[:limit]


def has_prompt_leak(text):

    lowered = str(text or "").lower()

    return any(
        re.search(pattern, lowered, re.IGNORECASE)
        for pattern in LEAK_PATTERNS + CONVERSATION_PATTERNS
    )


def normalize_reasoning_bullets(text):

    cleaned = str(text or "")

    cleaned = re.sub(
        r"```(?:\w+)?",
        "",
        cleaned
    )

    cleaned = re.sub(
        r"<[^>]*>",
        "",
        cleaned
    )

    cleaned = cleaned.replace(
        "\r",
        "\n"
    )

    for pattern in PROMPT_ECHO_PATTERNS:

        cleaned = re.sub(
            pattern,
            "",
            cleaned
        )

    cleaned = re.sub(
        r"(?im)^\s*(assistant|system|user|model|reasoning|analysis|thought)\s*:\s*",
        "",
        cleaned
    )

    cleaned = re.sub(
        r"(?im)^\s*(?:[-*]|\d+[.)])\s*",
        "- ",
        cleaned
    )

    candidates = []

    for line in cleaned.split("\n"):

        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if not line:
            continue

        if line.startswith("- "):
            line = line[2:].strip()

        if has_prompt_leak(line):
            continue

        if re.search(
            r"(?i)\b(requirements|instructions|do not|only use|return only)\b",
            line
        ):
            continue

        line = re.sub(
            r"(?i)\b(based on|according to)\s+(the\s+)?(provided\s+)?context[:,]?\s*",
            "",
            line
        ).strip()

        line = re.sub(
            r"(?i)\bclinical reasoning\b[:,]?\s*",
            "",
            line
        ).strip()

        line = line.strip(" -;:")

        if len(line) < 18:
            continue

        if len(line.split()) > 28:

            line = " ".join(
                line.split()[:28]
            ).rstrip(" ,;:")

        if line and line[-1] not in ".!?":
            line += "."

        candidates.append(line)

    final = []
    seen = set()

    for line in candidates:

        key = re.sub(
            r"\W+",
            " ",
            line.lower()
        )[:90]

        if key in seen:
            continue

        seen.add(key)
        final.append(line)

        if len(final) >= 4:
            break

    if len(final) < 2:
        return ""

    return "\n".join(
        f"- {line}"
        for line in final[:4]
    )


def fallback_reasoning(
    query,
    answer,
    docs
):

    supporting = extract_supporting_sentences(
        query,
        answer,
        docs or []
    )

    bullets = []

    answer_summary = compact_text(
        answer,
        180
    )

    if answer_summary:

        bullets.append(
            "The answer remains tied to retrieved oncology evidence: "
            + answer_summary
        )

    for sent in supporting[:2]:

        sent = compact_text(
            sent,
            190
        )

        if sent:

            bullets.append(
                "A retrieved evidence point supporting the answer is: "
                + sent
            )

    if not bullets:

        bullets = [
            "Retrieved oncology evidence was limited, so the explanation should be interpreted cautiously.",
            "The answer should remain tied to documented findings and avoid unsupported clinical conclusions."
        ]

    return "\n".join(
        f"- {bullet.rstrip('.')}."
        for bullet in bullets[:4]
    )


# =========================================================
# 🔹 MEDICAL REASONING GENERATION
# =========================================================
def generate_reasoning(
    query,
    answer,
    docs
):

    safe_docs = [
        compact_text(doc, 850)
        for doc in (docs or [])[:3]
        if str(doc or "").strip()
    ]

    if not safe_docs:

        return fallback_reasoning(
            query,
            answer,
            docs
        )

    context = "\n\n".join(
        safe_docs
    )[:2400]

    prompt = f"""ONCOLOGY_CLINICAL_REASONING_COMPLETION

Source material:
Question: {compact_text(query, 320)}
Retrieved evidence: {context}
Answer to explain: {compact_text(answer, 900)}

Output contract:
- Produce final clinical reasoning only.
- Write 2 to 4 bullets.
- Start every bullet with "- ".
- Each bullet must be one concise oncology sentence.
- Use only the retrieved evidence and answer.
- Do not mention prompts, instructions, context, users, or internal thinking.
- Do not ask questions.
- Do not add a diagnosis, treatment, or fact that is not supported above.

Clinical reasoning bullets:
- """

    fallback = fallback_reasoning(
        query,
        answer,
        docs
    )

    try:

        response = SESSION.post(
            "http://localhost:11434/api/generate",
            json={

                "model": REASONING_MODEL,

                "prompt": prompt,

                "stream": False,

                "options": {

                    "temperature": 0.05,

                    "top_p": 0.70,

                    "top_k": 20,

                    "repeat_penalty": 1.20,

                    "num_predict": 140,

                    "num_ctx": 3072,

                    "keep_alive": "10m",

                    "stop": [
                        "\nQuestion:",
                        "\nContext:",
                        "\nAnswer:",
                        "\nOutput contract:",
                        "\nSource material:",
                        "USER:",
                        "ASSISTANT:"
                    ]
                }
            },
            timeout=60
        )

        raw = response.json().get(
            "response",
            ""
        ).strip()

        if raw and not raw.startswith("-"):
            raw = "- " + raw

        cleaned = normalize_reasoning_bullets(
            raw
        )

        # =====================================================
        # 🔹 REMOVE THOUGHT LEAKS
        # =====================================================

        cleaned = re.sub(
            r"(?i)^thought\s*",
            "",
            cleaned
        )

        cleaned = re.sub(
            r"(?i)^reasoning\s*",
            "",
            cleaned
        )

        cleaned = re.sub(
            r"(?i)^analysis\s*",
            "",
            cleaned
        )

        cleaned = re.sub(
            r"<.*?>",
            "",
            cleaned
        )

        # =====================================================
        # 🔹 REMOVE PROMPT ECHOING
        # =====================================================

        bad_patterns = [

            "the user wants me",

            "i need to follow",

            "strict output rules",

            "reasoning process",

            "step-by-step",

            "clinical reasoning:"
        ]

        for p in bad_patterns:

            cleaned = re.sub(
                p + r".*",
                "",
                cleaned,
                flags=re.IGNORECASE | re.DOTALL
            )

        # =====================================================
        # 🔹 REMOVE NUMBERED CHAIN OF THOUGHT
        # =====================================================

        cleaned = re.sub(
            r"^\s*\d+\.\s*",
            "- ",
            cleaned,
            flags=re.MULTILINE
        )

        cleaned = re.sub(
            r"\n{3,}",
            "\n",
            cleaned
        )

        cleaned = cleaned.strip()

        # =====================================================
        # 🔹 FALLBACK
        # =====================================================

        if (
            len(cleaned) < 40
            or has_prompt_leak(cleaned)
        ):
            return fallback

        return cleaned

    except Exception as e:

        print("⚠️ Reasoning failed:", e)

        return fallback


def generate_direct_reasoning(
    answer,
    eval_result
):

    quality = quality_label(
        eval_result.get(
            "score",
            5
        )
    )

    confidence = eval_result.get(
        "confidence",
        0.5
    )

    hallucination_risk = str(
        eval_result.get(
            "hallucination_risk",
            "medium"
        )
    ).lower()

    bullets = [
        "The answer was generated directly by the LLM without retrieval.",
        f"The response is rated {quality.lower()} with confidence {confidence:.2f}."
    ]

    if hallucination_risk == "high":

        bullets.append(
            "The answer should be reviewed carefully because the evaluator flagged a high hallucination risk."
        )

    elif hallucination_risk == "medium":

        bullets.append(
            "The answer should be interpreted with normal clinical caution because the evaluator marked a medium hallucination risk."
        )

    else:

        bullets.append(
            "The answer appears consistent with the model's direct medical response pattern."
        )

    return "\n".join(
        f"- {bullet}"
        for bullet in bullets[:4]
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

    if not settings.is_rag_enabled():

        explanation[
            "supporting_sentences"
        ] = []

        explanation["confidence"] = eval_result.get(
            "confidence",
            0.5
        )

        explanation["quality"] = quality_label(
            eval_result.get(
                "score",
                5
            )
        )

        explanation["grounded"] = False

        explanation["reasoning"] = generate_direct_reasoning(
            answer,
            eval_result
        )

        return explanation

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
    # 🔹 FULL MEDICAL REASONING
    # =====================================================
    explanation["reasoning"] = generate_reasoning(
        query,
        answer,
        docs
    )

    return explanation
