import requests
import json
import re
import settings
from utils.metadata_tools import classify_query_metadata

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

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters except useful medical separators
    text = re.sub(r"[^\w\s\-\/]", " ", text)

    text = re.sub(r"\s+", " ", text)

    fixes = {

        # General medical typos
        "symtoms": "symptoms",
        "symptomps": "symptoms",
        "tretment": "treatment",
        "treatement": "treatment",
        "diagnsis": "diagnosis",
        "diagonsis": "diagnosis",
        "diagosis": "diagnosis",
        "prognsis": "prognosis",

        # Therapy typos
        "chemo therapy": "chemotherapy",
        "chemo-therapy": "chemotherapy",
        "immuno therapy": "immunotherapy",
        "radio therapy": "radiotherapy",
        "targeted therapys": "targeted therapy",

        # Cancer typos
        "tumour": "tumor",
        "carcinomaa": "carcinoma",
        "metastatis": "metastasis",

        # Common abbreviations
        "nsclc": "non small cell lung cancer",
        "sclc": "small cell lung cancer",
        "aml": "acute myeloid leukemia",
        "all": "acute lymphoblastic leukemia",
        "cll": "chronic lymphocytic leukemia",
        "cml": "chronic myeloid leukemia",

        # Treatment abbreviations
        "rt": "radiotherapy",
        "ct": "chemotherapy",
        "io": "immunotherapy",

        # Oncology abbreviations
        "os": "overall survival",
        "dfs": "disease free survival",
        "pfs": "progression free survival",
        "orr": "objective response rate",

        # Diagnostic abbreviations
        "mri": "magnetic resonance imaging",
        "ct scan": "computed tomography scan",
        "pet ct": "positron emission tomography computed tomography"
    }

    for wrong, correct in fixes.items():
        pattern = r'\b' + re.escape(wrong) + r'\b'
        text = re.sub(pattern, correct, text, flags=re.IGNORECASE)

    words = text.split()

    normalized = []

    for word in words:
        normalized.append(
            fixes.get(word, word)
        )

    text = " ".join(normalized)

    return text.strip()


# =========================================================
# 🔹 TOKENIZER
# =========================================================
def tokenize(text):

    if not text:
        return []

    text = text.lower().strip()

    tokens = re.findall(
        r"[a-zA-Z0-9]+(?:[-+/][a-zA-Z0-9]+)*",
        text
    )

    return tokens


# =========================================================
# 🔹 QUERY TYPE DETECTION
# =========================================================
def detect_query_type(query):

    q = query.lower().strip()

    patterns = {

        "comparison": [
            "difference between",
            "compare",
            "comparison",
            "vs",
            "versus"
        ],

        "ranking": [
            "rank",
            "ranking",
            "top",
            "most common",
            "least common",
            "best",
            "highest",
            "lowest"
        ],

        "list": [
            "list",
            "types",
            "type of",
            "kinds of",
            "categories",
            "classification"
        ],

        "definition": [
            "what is",
            "define",
            "meaning of",
            "explain",
            "describe"
        ],

        "symptoms": [
            "symptoms",
            "signs",
            "clinical presentation",
            "manifestations"
        ],

        "diagnosis": [
            "diagnosis",
            "diagnostic",
            "detect",
            "identify",
            "screening",
            "biopsy",
            "test"
        ],

        "treatment": [
            "treatment",
            "therapy",
            "drug",
            "medicine",
            "management",
            "intervention"
        ],

        "side_effects": [
            "side effects",
            "adverse effects",
            "toxicity",
            "complications",
            "adverse events"
        ],

        "prognosis": [
            "prognosis",
            "survival",
            "outcome",
            "life expectancy",
            "mortality"
        ],

        "staging": [
            "staging",
            "stage",
            "tnm",
            "cancer stage"
        ],

        "prevention": [
            "prevention",
            "prevent",
            "risk reduction"
        ],

        "risk_factors": [
            "risk factors",
            "causes",
            "cause",
            "etiology",
            "predispose"
        ],

        "clinical_trials": [
            "clinical trial",
            "phase i",
            "phase ii",
            "phase iii",
            "research study"
        ],

        "epidemiology": [
            "incidence",
            "prevalence",
            "epidemiology",
            "frequency"
        ]
    }

    # Highest priority checks
    for query_type in [
        "comparison",
        "ranking",
        "list",
        "definition"
    ]:

        for p in patterns[query_type]:

            if p in q:
                return query_type

    # Domain-specific checks
    for query_type, keywords in patterns.items():

        if query_type in {
            "comparison",
            "ranking",
            "list",
            "definition"
        }:
            continue

        for kw in keywords:

            if kw in q:
                return query_type

    yesno_starters = (
        "is",
        "are",
        "does",
        "do",
        "can",
        "could",
        "should",
        "will",
        "would",
        "has",
        "have"
    )

    if q.startswith(yesno_starters):
        return "yesno"

    return "general"

# =========================================================
# 🔹 SIMPLE QUERY DETECTION
# =========================================================
def is_simple_query(query):

    tokens = tokenize(query)

    if len(tokens) > 12:
        return False

    simple_patterns = {
        "what is",
        "define",
        "meaning of",
        "list",
        "types of",
        "kinds of"
    }

    q = query.lower().strip()

    for pattern in simple_patterns:

        if q.startswith(pattern):
            return True

    complex_keywords = {

        "compare",
        "versus",
        "vs",

        "difference",

        "treatment",

        "therapy",

        "diagnosis",

        "staging",

        "prognosis",

        "survival",

        "metastasis",

        "mechanism",

        "pathogenesis",

        "side effects",

        "adverse effects",

        "clinical trial",

        "risk factors"
    }

    for kw in complex_keywords:

        if kw in q:
            return False

    return len(tokens) <= 5


# =========================================================
# 🔹 DETERMINISTIC EXPANSION
# =========================================================
import re

def deterministic_expansion(
    query,
    query_type
):

    query = query.lower().strip()

    # -----------------------------
    # LIST
    # -----------------------------
    if query_type == "list":

        return (
            query
            + " classification categories subtypes"
        )

    # -----------------------------
    # RANKING
    # -----------------------------
    if query_type == "ranking":

        return (
            query
            + " prevalence incidence frequency ranking"
        )

    # -----------------------------
    # DEFINITION
    # -----------------------------
    if query_type == "definition":

        return (
            query
            + " definition overview description"
        )

    # -----------------------------
    # SYMPTOMS
    # -----------------------------
    if query_type == "symptoms":

        return (
            query
            + " symptoms signs manifestations clinical presentation"
        )

    # -----------------------------
    # DIAGNOSIS
    # -----------------------------
    if query_type == "diagnosis":

        return (
            query
            + " diagnosis diagnostic criteria biopsy imaging screening evaluation"
        )

    # -----------------------------
    # TREATMENT
    # -----------------------------
    if query_type == "treatment":

        return (
            query
            + " treatment therapy management chemotherapy immunotherapy radiotherapy surgery"
        )

    # -----------------------------
    # SIDE EFFECTS
    # -----------------------------
    if query_type == "side_effects":

        return (
            query
            + " adverse effects toxicity complications treatment toxicity"
        )

    # -----------------------------
    # PROGNOSIS
    # -----------------------------
    if query_type == "prognosis":

        return (
            query
            + " prognosis survival outcome overall survival disease free survival"
        )

    # -----------------------------
    # STAGING
    # -----------------------------
    if query_type == "staging":

        return (
            query
            + " cancer stage tnm staging classification"
        )

    # -----------------------------
    # RISK FACTORS
    # -----------------------------
    if query_type == "risk_factors":

        return (
            query
            + " risk factors causes etiology predisposition"
        )

    # -----------------------------
    # PREVENTION
    # -----------------------------
    if query_type == "prevention":

        return (
            query
            + " prevention screening risk reduction protective factors"
        )

    # -----------------------------
    # CLINICAL TRIALS
    # -----------------------------
    if query_type == "clinical_trials":

        return (
            query
            + " clinical trial study evidence phase i phase ii phase iii"
        )

    # -----------------------------
    # COMPARISON
    # -----------------------------
    if query_type == "comparison":

        return (
            query
            + " comparison differences advantages disadvantages"
        )

    # -----------------------------
    # YESNO
    # -----------------------------
    if query_type == "yesno":

        return (
            query
            + " evidence recommendation guideline"
        )

    # -----------------------------
    # GENERAL
    # -----------------------------
    return (
        query
        + " oncology cancer information"
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

    k_map = {

        "definition": 4,

        "yesno": 4,

        "symptoms": 6,

        "diagnosis": 6,

        "treatment": 7,

        "side_effects": 7,

        "comparison": 8,

        "staging": 7,

        "prognosis": 7,

        "risk_factors": 6,

        "prevention": 6,

        "clinical_trials": 8,

        "list": 8,

        "ranking": 10,

        "epidemiology": 7
    }

    return k_map.get(query_type, 5)


# =========================================================
# 🔹 INTENT DETECTION
# =========================================================
def detect_intent(query_type):

    intent_map = {

        # Direct factual retrieval
        "definition": "factual",
        "symptoms": "factual",
        "diagnosis": "factual",
        "staging": "factual",
        "epidemiology": "factual",
        "yesno": "factual",

        # Enumerative answers
        "list": "enumeration",
        "ranking": "enumeration",

        # Analytical reasoning
        "comparison": "analytical",

        # Treatment guidance
        "treatment": "clinical_guidance",
        "side_effects": "clinical_guidance",
        "prevention": "clinical_guidance",
        "risk_factors": "clinical_guidance",

        # Outcome-focused
        "prognosis": "outcome_analysis",

        # Research-oriented
        "clinical_trials": "research"
    }

    return intent_map.get(
        query_type,
        "exploratory"
    )
# =========================================================
# 🔹 KEYWORD EXTRACTION
# =========================================================
def extract_keywords(query):

    stopwords = {

        "what", "is", "the", "of", "a", "an",
        "does", "can", "are", "and", "to",
        "in", "on", "for", "with", "by",
        "how", "why", "when", "where",
        "which", "who", "whom",

        "explain",
        "describe",
        "define",
        "list",
        "compare",
        "difference",
        "between",
        "versus",
        "vs",

        "tell",
        "show",
        "give"
    }

    tokens = tokenize(query)

    keywords = []

    seen = set()

    for token in tokens:

        if token in stopwords:
            continue

        if len(token) < 3:
            continue

        if token not in seen:
            keywords.append(token)
            seen.add(token)

    return keywords[:10]


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

    if len(tokenize(query)) < 6:
        return False

    if is_simple_query(query):
        return False

    if query_type in {
        "definition",
        "list",
        "ranking",
        "yesno",
        "symptoms",
        "diagnosis",
        "staging"
    }:
        return False

    return intent in {
        "analytical",
        "clinical_guidance",
        "outcome_analysis",
        "research",
        "exploratory"
    }

# =========================================================
# 🔹 DOMAIN SAFE EXPANSION
# =========================================================
import re

def domain_safe_expansion(
    original_query,
    expanded_query,
    query_type
):

    # --------------------------------------------------
    # Remove obviously irrelevant domain drift
    # --------------------------------------------------
    banned_patterns = [

        r"\blegal\b",
        r"\blaw\b",

        r"\bfinancial\b",
        r"\bfinance\b",
        r"\beconomic\b",

        r"\bstudy design\b",
        r"\bretrieval query\b",

        r"\bacademic\b",
        r"\bliterature review\b",

        r"\bmarketing\b",
        r"\bbusiness\b",

        r"\bpolitical\b",
        r"\bgovernment policy\b"
    ]

    safe = expanded_query.lower().strip()

    for pattern in banned_patterns:

        safe = re.sub(
            pattern,
            " ",
            safe,
            flags=re.IGNORECASE
        )

    safe = re.sub(
        r"\s+",
        " ",
        safe
    ).strip()

    # --------------------------------------------------
    # Ensure original query remains intact
    # --------------------------------------------------
    original = original_query.lower().strip()

    if not safe.startswith(original):

        safe = (
            original
            + " "
            + safe
        )

    # --------------------------------------------------
    # Tokenize
    # --------------------------------------------------
    original_tokens = tokenize(original)

    safe_tokens = tokenize(safe)

    # --------------------------------------------------
    # Keep only newly-added tokens
    # --------------------------------------------------
    added = [

        token for token in safe_tokens

        if token not in original_tokens
    ]

    # --------------------------------------------------
    # Query-type-specific limits
    # --------------------------------------------------
    limits = {

        "definition": 6,

        "yesno": 6,

        "symptoms": 8,

        "diagnosis": 8,

        "staging": 8,

        "risk_factors": 8,

        "prevention": 8,

        "treatment": 10,

        "side_effects": 10,

        "prognosis": 10,

        "comparison": 12,

        "clinical_trials": 14,

        "ranking": 8,

        "list": 8,

        "general": 8
    }

    max_added_words = limits.get(
        query_type,
        8
    )

    added = added[:max_added_words]

    # --------------------------------------------------
    # Remove duplicates while preserving order
    # --------------------------------------------------
    seen = set()

    final_tokens = []

    for token in original_tokens + added:

        if token not in seen:

            final_tokens.append(token)

            seen.add(token)

    # --------------------------------------------------
    # Oncology safety check
    # Prevent weird LLM expansions
    # --------------------------------------------------
    oncology_terms = {

        "cancer",
        "tumor",
        "tumour",
        "oncology",

        "chemotherapy",
        "immunotherapy",
        "radiotherapy",

        "diagnosis",
        "symptoms",

        "treatment",
        "therapy",

        "prognosis",
        "survival",

        "metastasis",
        "metastatic",

        "staging",

        "carcinoma",
        "sarcoma",
        "lymphoma",
        "leukemia",
        "melanoma",

        "clinical",
        "trial",

        "toxicity",
        "adverse",
        "effects",

        "screening",
        "prevention",

        "egfr",
        "alk",
        "her2",
        "braf",
        "pd-l1",
        "car-t"
    }

    has_medical_signal = any(
        token in oncology_terms
        for token in final_tokens
    )

    if not has_medical_signal:

        return original

    return " ".join(final_tokens)
# =========================================================
# 🔹 MAIN PROCESSOR
# =========================================================
def process_query(query):

    if not settings.is_laqa_enabled():

        return settings.build_raw_query_payload(
            query
        )

    # ------------------------------------
    # CLEAN QUERY
    # ------------------------------------
    query = clean_query(query)

    # ------------------------------------
    # DETECT QUERY TYPE
    # ------------------------------------
    query_type = detect_query_type(
        query
    )

    intent = detect_intent(
        query_type
    )

    query_complexity = (
        "simple"
        if is_simple_query(query)
        else "complex"
    )

    # ------------------------------------
    # DETERMINISTIC EXPANSION
    # ------------------------------------
    expanded_query = deterministic_expansion(
        query,
        query_type
    )

    expansion_source = "deterministic"

    # ------------------------------------
    # OPTIONAL LLM EXPANSION
    # ------------------------------------
    if should_use_llm_expansion(
        query,
        intent,
        query_type
    ):

        expanded_query = lightweight_llm_expand(
            expanded_query
        )

        expansion_source = "llm"

    # ------------------------------------
    # DOMAIN SAFETY
    # ------------------------------------
    expanded_query = domain_safe_expansion(
        query,
        expanded_query,
        query_type
    )

    # ------------------------------------
    # FINAL ENRICHMENT
    # ------------------------------------
    expanded_query = enrich_query(
        expanded_query
    )

    # ------------------------------------
    # KEYWORDS
    # ------------------------------------
    keywords = list(dict.fromkeys(

        extract_keywords(query)

        +

        extract_keywords(expanded_query)

    ))

    # ------------------------------------
    # QUERY METADATA
    # ------------------------------------
    query_metadata = classify_query_metadata(

        query=query,

        keywords=keywords,

        query_type=query_type,

        expanded_query=expanded_query
    )

    # ------------------------------------
    # FINAL OUTPUT
    # ------------------------------------
    final_output = {

        "intent": intent,

        "query_type": query_type,

        "query_complexity": query_complexity,

        "keywords": keywords,

        "query_metadata": query_metadata,

        "expanded_query": expanded_query,

        "expansion_source": expansion_source,

        "retrieval_k": choose_k(
            query_type
        ),

        "original_query": query
    }

    # ------------------------------------
    # DEBUG
    # ------------------------------------
    print("\n🧠 LAQA PARSED:")

    print(
        json.dumps(
            final_output,
            indent=2
        )
    )

    return final_output
