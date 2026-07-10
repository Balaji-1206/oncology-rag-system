import requests
import re
import settings

SESSION = requests.Session()

OLLAMA_URL = "http://localhost:11434/api/generate"

#MODEL = "hf.co/QuantFactory/Llama3-Med42-8B-GGUF:Q4_K_M" 
MODEL = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"


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
# 🔹 SENTENCE SPLITTER
# =========================================================
def split_sentences(text):

    return [

        s.strip()

        for s in re.split(
            r'(?<=[.!?])\s+',
            text
        )

        if len(s.strip()) > 20
    ]


# =========================================================
# 🔹 SENTENCE RELEVANCE
# =========================================================
STOPWORDS = {
    "what", "which", "when", "where", "why", "how",
    "does", "can", "are", "is", "the", "and", "for",
    "with", "from", "that", "this", "into", "about",
    "list", "rank", "ranking", "top"
}


ONCOLOGY_TERMS = {
    "cancer", "cancers", "tumor", "tumour", "tumors", "tumours",
    "oncology", "carcinoma", "sarcoma", "lymphoma", "leukemia",
    "melanoma", "myeloma", "metastasis", "staging", "grade",
    "biomarker", "mutation", "therapy", "treatment", "chemotherapy",
    "immunotherapy", "radiotherapy"
}


def focused_tokens(tokens):

    focused = tokens - STOPWORDS

    if focused:
        return focused

    return tokens


def sentence_relevance(
    query_tokens,
    sentence,
    doc_rank=0,
    sentence_rank=0
):

    sent_tokens = tokenize(sentence)

    focused_query = focused_tokens(
        query_tokens
    )

    overlap = len(
        focused_query & sent_tokens
    )

    overlap_score = overlap / max(
        len(focused_query),
        1
    )

    broad_overlap = len(
        query_tokens & sent_tokens
    )

    broad_score = broad_overlap / max(
        len(query_tokens),
        1
    )

    oncology_boost = min(
        len(ONCOLOGY_TERMS & sent_tokens) * 0.04,
        0.16
    )

    position_boost = max(
        0,
        0.12 - (doc_rank * 0.025) - (sentence_rank * 0.01)
    )

    length = len(sentence.split())

    if length > 60:
        overlap_score *= 0.85

    if length < 6:
        overlap_score *= 0.7

    return (
        0.74 * overlap_score
        +
        0.18 * broad_score
        +
        oncology_boost
        +
        position_boost
    )


def chunk_relevance(
    query_tokens,
    doc,
    doc_rank
):

    doc_tokens = tokenize(doc)

    focused_query = focused_tokens(
        query_tokens
    )

    overlap = len(
        focused_query & doc_tokens
    ) / max(
        len(focused_query),
        1
    )

    oncology_overlap = min(
        len(ONCOLOGY_TERMS & doc_tokens) * 0.03,
        0.18
    )

    rank_prior = max(
        0,
        0.18 - (doc_rank * 0.025)
    )

    return overlap + oncology_overlap + rank_prior


def noisy_sentence(sentence):

    bad_patterns = [
        "et al",
        "copyright",
        "all rights reserved",
        "figure ",
        "table "
    ]

    sent_lower = sentence.lower()

    has_noise = any(
        bad in sent_lower
        for bad in bad_patterns
    )

    if not has_noise:
        return False

    sent_tokens = tokenize(sentence)

    has_medical_signal = bool(
        ONCOLOGY_TERMS & sent_tokens
    )

    return not has_medical_signal


# =========================================================
# 🔹 QUERY-ALIGNED CONTEXT
# =========================================================
def build_context(
    docs,
    query,
    intent,
    query_type
):

    if not docs:
        return ""

    query_tokens = tokenize(query)

    if query_type in [
        "list",
        "ranking"
    ]:
        max_docs = 7
        max_sentences = 14

    elif intent == "exploratory":
        max_docs = 6
        max_sentences = 12

    else:
        max_docs = 5
        max_sentences = 10

    normalized_docs = []

    for idx, doc in enumerate(docs[:max_docs + 2]):

        clean_doc = re.sub(
            r"\s+",
            " ",
            str(doc)
        ).strip()

        if not clean_doc:
            continue

        normalized_docs.append(
            (
                chunk_relevance(
                    query_tokens,
                    clean_doc,
                    idx
                ),
                idx,
                clean_doc
            )
        )

    normalized_docs.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    candidate_docs = normalized_docs[:max_docs]

    scored_sentences = []

    for doc_score, doc_idx, doc in candidate_docs:


        sentences = split_sentences(doc)

        for sent_idx, sent in enumerate(sentences):

            relevance = sentence_relevance(
                query_tokens,
                sent,
                doc_idx,
                sent_idx
            )

            if noisy_sentence(sent):
                continue

            if relevance >= 0.12:

                scored_sentences.append(
                    (
                        relevance + (doc_score * 0.18),
                        doc_idx,
                        sent
                    )
                )

    scored_sentences.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    final_sentences = []

    seen = []

    per_doc_counts = {}

    for score, doc_idx, sent in scored_sentences:

        sent_tokens = tokenize(sent)

        duplicate = False

        for prev in seen:

            overlap = len(
                sent_tokens & prev
            )

            similarity = overlap / max(
                len(sent_tokens),
                1
            )

            if similarity > 0.72:

                duplicate = True
                break

        if duplicate:
            continue

        seen.append(sent_tokens)

        per_doc_counts[doc_idx] = per_doc_counts.get(
            doc_idx,
            0
        )

        if per_doc_counts[doc_idx] >= 4:
            continue

        per_doc_counts[doc_idx] += 1

        final_sentences.append(
            (
                doc_idx,
                sent
            )
        )

        if len(final_sentences) >= max_sentences:
            break

    if not final_sentences:

        for _, doc_idx, d in candidate_docs:

            final_sentences.append(
                (
                    doc_idx,
                    d[:500]
                )
            )

    grouped = {}

    for doc_idx, sent in final_sentences:

        grouped.setdefault(
            doc_idx,
            []
        ).append(sent)

    context_blocks = []

    for block_num, doc_idx in enumerate(sorted(grouped.keys()), start=1):

        evidence = " ".join(
            grouped[doc_idx]
        )

        context_blocks.append(
            f"[Chunk {block_num}] {evidence}"
        )

    context = "\n\n".join(context_blocks)

    return context[:1800]


# =========================================================
# 🔹 CLEAN OUTPUT
# =========================================================
def clean_output(answer):

    if not answer:
        return ""

    markers = [

        "Answer:",

        "FINAL ANSWER:",

        "<unused95>"
    ]

    for marker in markers:

        if marker in answer:

            answer = answer.split(
                marker
            )[-1]

    patterns = [

        r"<unused\d+>",

        r"(?s)<think>.*?</think>",

        r"(?s)<analysis>.*?</analysis>",

        r"(?s)<reasoning>.*?</reasoning>",

        r"(?s)Reasoning:.*",

        r"(?s)thought.*",

        r"(?s)Observation:.*",

        r"(?s)Constraint Checklist.*",

        r"(?s)The user wants.*"
    ]

    for p in patterns:

        answer = re.sub(
            p,
            "",
            answer
        )

    lines = []

    seen = set()

    for line in answer.split("\n"):

        line = line.strip()

        if not line:
            continue

        prefix = " ".join(
            line.lower().split()[:6]
        )

        if prefix in seen:
            continue

        seen.add(prefix)

        lines.append(line)

    answer = "\n".join(lines)

    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer
    )

    answer = re.sub(
        r"\s{2,}",
        " ",
        answer
    )

    return answer.strip()


# =========================================================
# 🔹 DIRECT OUTPUT CLEANUP
# =========================================================
# Direct LLM mode should only remove formatting artifacts.
def clean_direct_output(answer):

    if not answer:
        return ""

    raw = answer.strip()

    reasoning_prefixes = [
        "thought",
        "thinking",
        "analysis",
        "reasoning",
        "step-by-step",
        "scratchpad",
        "chain of thought"
    ]

    starts_with_reasoning = any(
        raw.lower().startswith(prefix)
        for prefix in reasoning_prefixes
    )

    final_markers = [
        "final answer:",
        "answer:",
        "final answer",
        "final medical answer:"
    ]

    extracted = raw

    for marker in final_markers:

        marker_index = raw.lower().find(marker)

        if marker_index != -1:

            extracted = raw[marker_index + len(marker):]
            break

    if starts_with_reasoning and extracted == raw:

        return ""

    answer = extracted

    answer = re.sub(
        r"(?s)<think>.*?</think>",
        "",
        answer
    )

    answer = re.sub(
        r"(?s)<analysis>.*?</analysis>",
        "",
        answer
    )

    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer
    )

    answer = re.sub(
        r"(?im)^(thought|thinking|analysis|reasoning|step-by-step|scratchpad|chain of thought)\s*:?",
        "",
        answer
    )

    lines = []
    seen = set()

    for line in answer.split("\n"):

        line = line.strip()

        if not line:
            continue

        prefix = " ".join(
            line.lower().split()[:6]
        )

        if prefix in seen:
            continue

        seen.add(prefix)

        lines.append(line)

    answer = "\n".join(lines)

    return answer.strip()


# =========================================================
# 🔹 VALIDATION
# =========================================================
def validate_answer(answer):

    if len(answer.strip()) < 2:
        return False

    severe_patterns = [

        "the user wants",

        "system prompt",

        "internal instruction",

        "<think>",

        "<analysis>"
    ]

    answer_lower = answer.lower()

    for p in severe_patterns:

        if p in answer_lower:
            return False

    return True


# =========================================================
# 🔹 GENERALIZATION
# =========================================================
def allow_generalization(query_type):

    return query_type in [

        "definition",

        "symptoms",

        "yesno",

        "general"
    ]


# =========================================================
# 🔹 ANSWER STYLE
# =========================================================
def build_answer_style(
    intent,
    query_type
):

    if query_type == "list":

        return """
- Use concise bullet points
- Cover all major items supported by the chunks
- Mention the most important or broad categories first
- Use 4-8 bullets when evidence supports it
- Do not collapse multiple distinct entities into one vague bullet
"""

    if query_type == "ranking":

        return """
- Use ranked bullet points
- Mention highest importance first
- Include the ranking basis only if the chunks support it
- Keep concise
"""

    if query_type == "definition":

        return """
- Start with a direct definition
- Keep medically concise
"""

    if query_type == "symptoms":

        return """
- Mention common symptoms first
- Keep concise
"""

    if intent == "comparison":

        return """
- Compare major medical differences only
- Keep concise
"""

    return """
- Answer directly
- Stay medically relevant
- Avoid unnecessary detail
"""


# =========================================================
# 🔹 GROUNDING RULES
# =========================================================
def build_grounding_rules(query_type):

    return """
- Use retrieved oncology evidence as the ONLY source
- Answer MUST be strictly grounded in the provided chunks
- Do NOT use general medical knowledge if not present in chunks
- Do NOT hallucinate or invent medical claims
- If the retrieved evidence does not contain the answer, say "I don't have enough information to answer that."
"""


# =========================================================
# 🔹 DIRECTNESS RULES
# =========================================================
def build_directness_rules():

    return """
- Answer immediately
- Avoid:
  "Based on the context"
  "According to the context"
  "The context discusses"
- Avoid filler introductions
- Put the direct answer in the first sentence or first bullet
"""


# =========================================================
# 🔹 PROMPT TEMPLATES
# =========================================================
# Keep the RAG prompt and the direct-LLM prompt separate so
# enable_rag can switch behavior without changing generation code.
PROMPT_WITH_CONTEXT = """
You are an oncology medical assistant.

RULES:
- Use ONLY the provided oncology context
- Keep answers concise and medically accurate
- No chain-of-thought
- No reasoning traces
- No repeated points
- No XML tags
- No markdown headings
- If evidence is insufficient say:
"I don't have enough information in the retrieved documents."

STYLE:
{answer_style}

MEDICAL CONTEXT:
{context}

QUESTION:
{query}

Provide only the final answer:
"""

PROMPT_WITHOUT_CONTEXT = """
You are an expert oncology medical assistant.

Answer the user's medical question directly.

Return ONLY the final answer.

Do NOT show your reasoning.

Do NOT show your thinking.

Do NOT output words such as:

thought

thinking

analysis

reasoning

step-by-step

scratchpad

chain of thought

Only return the final medical answer.

If you are uncertain, state that clearly instead of inventing information.

Question:

{question}

Final Answer:

"""


# =========================================================
# 🔹 PROMPT BUILDER
# =========================================================
def build_prompt(
    query,
    context,
    intent,
    query_type
):

    answer_style = build_answer_style(
        intent,
        query_type
    )

    return PROMPT_WITH_CONTEXT.format(
        answer_style=answer_style,
        context=context,
        query=query
    )

# =========================================================
# 🔹 RESPONSE OPTIONS
# =========================================================
def build_generation_options(query_type="general"):

    if query_type in [
        "list",
        "ranking"
    ]:
        num_predict = 180

    elif query_type in [
        "definition",
        "yesno"
    ]:
        num_predict = 110

    else:
        num_predict = 140

    return {

        "temperature": 0,

        "top_p": 0.82,

        "top_k": 30,

        "repeat_penalty": 1.12,

        "num_predict": num_predict,

        "num_ctx": 2048,

        "num_thread": 8,

        "num_batch": 256,

        "keep_alive": "60m",

        "stop": [

            "Reasoning:",

            "Observation:",

            "Constraint Checklist",

            "<think>",

            "</think>",

            "<analysis>",

            "</analysis>",

            "<reasoning>",

            "</reasoning>",

            "The user wants",

            "MEDICAL CONTEXT:"
        ]
    }


def build_direct_generation_options(query_type="general"):

    if query_type in [
        "list",
        "ranking"
    ]:
        num_predict = 180

    elif query_type in [
        "definition",
        "yesno"
    ]:
        num_predict = 110

    else:
        num_predict = 140

    return {

        "temperature": 0,

        "top_p": 0.82,

        "top_k": 30,

        "repeat_penalty": 1.12,

        "num_predict": num_predict,

        "num_ctx": 2048,

        "num_thread": 8,

        "num_batch": 256,

        "keep_alive": "60m",

        "stop": [

            "</think>"
        ]
    }

# =========================================================
# 🔹 MAIN GENERATOR
# =========================================================
def generate_answer(agent_output):

    query_data = agent_output["query"]

    query = query_data["original_query"]

    intent = query_data.get(
        "intent",
        "factual"
    )

    query_type = query_data.get(
        "query_type",
        "general"
    )

    docs = agent_output["context"]

    if settings.is_rag_enabled():

        context = build_context(
            docs,
            query,
            intent,
            query_type
        )

        prompt = build_prompt(
            query,
            context,
            intent,
            query_type
        )

    else:

        prompt = PROMPT_WITHOUT_CONTEXT.format(
            question=query
        )

    if settings.is_rag_enabled():

        generation_options = build_generation_options(
            query_type
        )

    else:

        generation_options = build_direct_generation_options(
            query_type
        )

    try:

        response = SESSION.post(
            OLLAMA_URL,
            json={

                "model": MODEL,

                "prompt": prompt,

                "stream": False,

                "options": generation_options
            },
            timeout=45
        )

        raw_answer = response.json().get(
            "response",
            ""
        )
        # print(repr(raw_answer))
        
        print("\n🧠 RAW GENERATER ANSWER:")
        print(raw_answer)

        if settings.is_rag_enabled():

            answer = clean_output(
                raw_answer
            )

        else:

            answer = clean_direct_output(
                raw_answer
            )

        if not answer.strip():

            return (
                "Unable to generate a medical answer."
            )

        if not validate_answer(answer):

            print(
                "⚠️ Validation failed"
            )

            return (
                "Unable to generate a reliable medical answer."
            )

        return answer.strip()

    except Exception as e:

        print(
            "❌ Generator error:",
            e
        )

        return (
            "Unable to generate a medical answer."
        )
