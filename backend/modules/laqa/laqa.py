import requests
import json
import re
import settings

SESSION = requests.Session()

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "phi3:mini"


# =========================================================
# 🔹 JSON EXTRACTION
# =========================================================
def extract_json(output):

    output = output.replace("```json", "")
    output = output.replace("```", "")

    match = re.search(
        r"\{.*\}",
        output,
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
# 🔹 CLEAN QUERY
# =========================================================
def clean_query(text):

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    fixes = {

        "symtoms": "symptoms",

        "tretment": "treatment",

        "diagnsis": "diagnosis",

        "chemo therapy": "chemotherapy",

        "immuno therapy": "immunotherapy",
    }

    for wrong, correct in fixes.items():

        text = text.replace(
            wrong,
            correct
        )

    return text.strip()


# =========================================================
# 🔹 TOKENIZER
# =========================================================
def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9\-]+\b",
        text.lower()
    )


# =========================================================
# 🔹 QUERY TYPE DETECTION
# =========================================================
def detect_query_type(query):

    q = query.lower()

    list_patterns = [

        "list",

        "types",

        "type of",

        "kinds of",

        "categories",

        "classification"
    ]

    ranking_patterns = [

        "rank",

        "ranking",

        "most common",

        "least common",

        "top"
    ]

    definition_patterns = [

        "what is",

        "define",

        "meaning of",

        "explain"
    ]

    comparison_patterns = [

        "difference between",

        "compare",

        "vs",

        "versus"
    ]

    yesno_patterns = [

        "does",

        "can",

        "is",

        "are",

        "will"
    ]

    symptom_patterns = [

        "symptoms",

        "signs",

        "indications"
    ]

    treatment_patterns = [

        "treatment",

        "therapy",

        "drug",

        "medicine"
    ]

    for p in ranking_patterns:

        if p in q:
            return "ranking"

    for p in list_patterns:

        if p in q:
            return "list"

    for p in definition_patterns:

        if p in q:
            return "definition"

    for p in comparison_patterns:

        if p in q:
            return "comparison"

    for p in symptom_patterns:

        if p in q:
            return "symptoms"

    for p in treatment_patterns:

        if p in q:
            return "treatment"

    for p in yesno_patterns:

        if q.startswith(p):
            return "yesno"

    return "general"


# =========================================================
# 🔹 SIMPLE QUERY DETECTION
# =========================================================
def is_simple_query(query):

    tokens = tokenize(query)

    simple_triggers = {

        "list",
        "type",
        "types",
        "kind",
        "kinds",
        "common",
        "top",
        "what",
        "define"
    }

    return (
        len(tokens) <= 7
        or
        bool(simple_triggers & set(tokens))
    )


# =========================================================
# 🔹 DETERMINISTIC EXPANSION
# =========================================================
def deterministic_expansion(
    query,
    query_type
):

    # -----------------------------------
    # LIST QUERIES
    # -----------------------------------
    if query_type == "list":

        if re.search(
            r"\bcancers?\b|\btumou?rs?\b|\bmalignan",
            query
        ):

            return (
                query
                +
                " oncology classification"
            )

        return (
            query
            +
            " oncology categories"
        )

    # -----------------------------------
    # RANKING QUERIES
    # -----------------------------------
    if query_type == "ranking":

        return (
            query
            +
            " oncology ranked categories"
        )

    # -----------------------------------
    # DEFINITION
    # -----------------------------------
    if query_type == "definition":

        return (
            query
            +
            " disease definition overview"
        )

    # -----------------------------------
    # SYMPTOMS
    # -----------------------------------
    if query_type == "symptoms":

        return (
            query
            +
            " symptoms signs clinical presentation"
        )

    # -----------------------------------
    # TREATMENT
    # -----------------------------------
    if query_type == "treatment":

        return (
            query
            +
            " treatment therapy management"
        )

    # -----------------------------------
    # COMPARISON
    # -----------------------------------
    if query_type == "comparison":

        return (
            query
            +
            " comparison differences"
        )

    # -----------------------------------
    # YES / NO
    # -----------------------------------
    if query_type == "yesno":

        return (
            query
            +
            " medical evidence"
        )

    # -----------------------------------
    # GENERAL
    # -----------------------------------
    return (
        query
        +
        " oncology information"
    )


# =========================================================
# 🔹 SAFE QUERY ENRICHMENT
# =========================================================
def enrich_query(query):

    tokens = query.split()

    unique = []

    seen = set()

    for t in tokens:

        if t not in seen:

            unique.append(t)

            seen.add(t)

    return " ".join(unique)[:300]


# =========================================================
# 🔹 RETRIEVAL K
# =========================================================
def choose_k(query_type):

    if query_type in [
        "list",
        "ranking"
    ]:
        return 7

    if query_type == "definition":
        return 4

    if query_type == "symptoms":
        return 6

    if query_type == "comparison":
        return 6

    if query_type == "treatment":
        return 6

    return 5


# =========================================================
# 🔹 INTENT DETECTION
# =========================================================
def detect_intent(query_type):

    if query_type in [
        "definition",
        "symptoms",
        "yesno",
        "list",
        "ranking"
    ]:
        return "factual"

    if query_type == "comparison":
        return "comparison"

    return "exploratory"


# =========================================================
# 🔹 KEYWORD EXTRACTION
# =========================================================
def extract_keywords(query):

    stopwords = {

        "what",
        "is",
        "the",
        "of",
        "a",
        "an",
        "does",
        "can",
        "are",
        "and",
        "to"
    }

    words = tokenize(query)

    keywords = [

        w for w in words

        if w not in stopwords
    ]

    return keywords[:8]


# =========================================================
# 🔹 OPTIONAL LIGHTWEIGHT LLM EXPANSION
# =========================================================
def lightweight_llm_expand(query):

    prompt = f"""
Expand this medical oncology retrieval query slightly.

STRICT RULES:
- Preserve the original user wording
- Keep concise
- Add ONLY medically useful retrieval keywords
- Maximum 8 added words
- NEVER add:
  legal
  finance
  prevalence
  epidemiology
  population studies
  academic wording
  literature review
  retrieval query
  study design
- Do NOT rewrite the query
- Do NOT create long sentences

Return ONLY the improved query.

QUERY:
{query}
"""

    try:

        response = SESSION.post(
            OLLAMA_URL,
            json={

                "model": MODEL,

                "prompt": prompt,

                "stream": False,

                "options": {

                    "temperature": 0,

                    "top_p": 0.1,

                    "top_k": 10,

                    "num_predict": 40,

                    "keep_alive": "20m"
                }
            },
            timeout=20
        )

        text = response.json().get(
            "response",
            ""
        ).strip()

        if len(text) > 5:
            return text

    except:
        pass

    return query


# =========================================================
# 🔹 SHOULD USE LLM EXPANSION
# =========================================================
def should_use_llm_expansion(
    query,
    intent,
    query_type
):

    if intent != "exploratory":
        return False

    if query_type in [
        "list",
        "ranking",
        "definition",
        "yesno"
    ]:
        return False

    if is_simple_query(query):
        return False

    return True


# =========================================================
# 🔹 DOMAIN SAFE EXPANSION
# =========================================================
def domain_safe_expansion(
    original_query,
    expanded_query,
    query_type
):

    banned_patterns = [

        r"\blegal\b",
        r"\blaw\b",
        r"\bfinancial\b",
        r"\bfinance\b",
        r"\beconomic\b",
        r"\bprevalence\b",
        r"\bincidence\b",
        r"\bpopulation statistics?\b",
        r"\badult populations?\b",
        r"\bepidemiolog",
        r"\bstudy design\b",
        r"\bretrieval query\b",
        r"\bacademic\b",
        r"\bliterature review\b"
    ]

    safe = expanded_query.lower().strip()

    for pattern in banned_patterns:

        safe = re.sub(
            pattern,
            " ",
            safe
        )

    safe = re.sub(
        r"\s+",
        " ",
        safe
    ).strip()

    original = original_query.lower().strip()

    if not safe.startswith(original):

        safe = (
            original
            + " "
            + safe
        )

    original_tokens = tokenize(original)

    safe_tokens = tokenize(safe)

    added = [

        token for token in safe_tokens

        if token not in original_tokens
    ]

    max_added_words = 12

    if (
        query_type in [
            "general",
            "list",
            "ranking",
            "definition"
        ]
        or
        is_simple_query(original)
    ):

        max_added_words = 8

    final_tokens = (
        original_tokens
        +
        added[:max_added_words]
    )

    return " ".join(final_tokens)


# =========================================================
# 🔹 MAIN PROCESSOR
# =========================================================
def process_query(query):

    if not settings.is_laqa_enabled():

        return settings.build_raw_query_payload(
            query
        )

    query = clean_query(query)

    query_type = detect_query_type(
        query
    )

    intent = detect_intent(
        query_type
    )

    # =====================================================
    # 🔹 SAFE DETERMINISTIC EXPANSION
    # =====================================================
    expanded_query = deterministic_expansion(
        query,
        query_type
    )

    # =====================================================
    # 🔹 OPTIONAL LLM ENRICHMENT
    # =====================================================
    if should_use_llm_expansion(
        query,
        intent,
        query_type
    ):

        expanded_query = lightweight_llm_expand(
            expanded_query
        )

    # =====================================================
    # 🔹 DOMAIN FILTERING
    # =====================================================
    expanded_query = domain_safe_expansion(
        query,
        expanded_query,
        query_type
    )

    # =====================================================
    # 🔹 FINAL CLEANING
    # =====================================================
    expanded_query = enrich_query(
        expanded_query
    )

    keywords = extract_keywords(
        query
    )

    # =====================================================
    # 🔹 FINAL OUTPUT
    # =====================================================
    final_output = {

        "intent": intent,

        "query_type": query_type,

        "keywords": keywords,

        "expanded_query": expanded_query,

        "retrieval_k": choose_k(
            query_type
        ),

        "original_query": query
    }

    # =====================================================
    # 🔹 DEBUG
    # =====================================================
    print("\n🧠 LAQA PARSED:")

    print(
        json.dumps(
            final_output,
            indent=2
        )
    )

    return final_output
