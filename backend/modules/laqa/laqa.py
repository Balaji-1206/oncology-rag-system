import json
import re
import requests
import settings
from utils.metadata_tools import classify_query_metadata

SESSION = requests.Session()
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"


def extract_json(output: str):
    """Extracts JSON object from model string output."""
    output = output.replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found")
    return json.loads(match.group(0))


def clean_query(text: str) -> str:
    """Normalizes query text, corrects typos, and standardizes medical terminology."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
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

        # Oncology abbreviations & synonyms
        "os": "overall survival",
        "dfs": "disease free survival",
        "pfs": "progression free survival",
        "orr": "objective response rate",
        "crc": "colorectal cancer",
        "rcc": "renal cell carcinoma",
        "hcc": "hepatocellular carcinoma",
        "tnbc": "triple negative breast cancer",
        "gist": "gastrointestinal stromal tumor",
        "pdl1": "pd-l1",
        "pd1": "pd-1",
        "ctla4": "ctla-4",
        "keytruda": "pembrolizumab",
        "opdivo": "nivolumab",
        "herceptin": "trastuzumab",

        # Diagnostic abbreviations
        "mri": "magnetic resonance imaging",
        "ct scan": "computed tomography scan",
        "pet ct": "positron emission tomography computed tomography"
    }

    for wrong, correct in fixes.items():
        pattern = r'\b' + re.escape(wrong) + r'\b'
        text = re.sub(pattern, correct, text, flags=re.IGNORECASE)

    return text.strip()


def tokenize(text: str) -> list:
    """Splits query text into clean token array."""
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9]+(?:[-+/][a-zA-Z0-9]+)*", text.lower().strip())


def detect_query_type(query: str) -> str:
    """Classifies query into clinical intent category based on keyword patterns."""
    q = query.lower().strip()

    patterns = {
        "comparison": ["difference between", "compare", "comparison", "vs", "versus"],
        "ranking": ["rank", "ranking", "top", "most common", "least common", "best", "highest", "lowest"],
        "list": ["list", "types", "type of", "kinds of", "categories", "classification"],
        "definition": ["what is", "define", "meaning of", "explain", "describe"],
        "symptoms": ["symptoms", "signs", "clinical presentation", "manifestations"],
        "diagnosis": ["diagnosis", "diagnostic", "detect", "identify", "screening", "biopsy", "test"],
        "treatment": ["treatment", "therapy", "drug", "medicine", "management", "intervention"],
        "side_effects": ["side effects", "adverse effects", "toxicity", "complications", "adverse events"],
        "prognosis": ["prognosis", "survival", "outcome", "life expectancy", "mortality"],
        "staging": ["staging", "stage", "tnm", "cancer stage"],
        "prevention": ["prevention", "prevent", "risk reduction"],
        "risk_factors": ["risk factors", "causes", "cause", "etiology", "predispose"],
        "clinical_trials": ["clinical trial", "phase i", "phase ii", "phase iii", "research study"],
        "epidemiology": ["incidence", "prevalence", "epidemiology", "frequency", "percentage", "percent", "proportion", "rate", "how many", "what fraction", "what proportion", "how common", "incidence of", "prevalence of"]
    }

    # Structural checks
    for query_type in ["comparison", "ranking", "list"]:
        for p in patterns[query_type]:
            if p in q:
                return query_type

    # Domain-specific clinical checks
    for query_type in ["symptoms", "diagnosis", "treatment", "side_effects", "prognosis", "staging", "prevention", "risk_factors", "clinical_trials", "epidemiology"]:
        for kw in patterns[query_type]:
            if kw in q:
                return query_type

    # Generic definition check
    for p in patterns["definition"]:
        if p in q:
            return "definition"

    yesno_starters = ("is", "are", "does", "do", "can", "could", "should", "will", "would", "has", "have")
    if q.startswith(yesno_starters):
        return "yesno"

    return "general"


def is_simple_query(query: str) -> bool:
    """Evaluates whether query is simple or complex."""
    tokens = tokenize(query)
    if len(tokens) > 12:
        return False

    simple_patterns = {"what is", "define", "meaning of", "list", "types of", "kinds of"}
    q = query.lower().strip()
    for pattern in simple_patterns:
        if q.startswith(pattern):
            return True

    complex_keywords = {
        "compare", "versus", "vs", "difference", "treatment", "therapy", "diagnosis",
        "staging", "prognosis", "survival", "metastasis", "mechanism", "pathogenesis",
        "side effects", "adverse effects", "clinical trial", "risk factors"
    }

    for kw in complex_keywords:
        if kw in q:
            return False

    return len(tokens) <= 5


def deterministic_expansion(query: str, query_type: str) -> str:
    """Applies rule-based clinical keyword expansion based on query type."""
    query = query.lower().strip()

    expansions = {
        "list": " classification categories subtypes",
        "ranking": " prevalence incidence frequency ranking",
        "definition": " definition overview description",
        "symptoms": " symptoms signs manifestations clinical presentation",
        "diagnosis": " diagnosis diagnostic criteria biopsy imaging screening evaluation",
        "treatment": " treatment therapy management chemotherapy immunotherapy radiotherapy surgery",
        "side_effects": " adverse effects toxicity complications treatment toxicity",
        "prognosis": " prognosis survival outcome overall survival disease free survival",
        "staging": " cancer stage tnm staging classification",
        "risk_factors": " risk factors causes etiology predisposition",
        "prevention": " prevention screening risk reduction protective factors",
        "clinical_trials": " clinical trial study evidence phase i phase ii phase iii",
        "comparison": " comparison differences advantages disadvantages",
        "yesno": " evidence recommendation guideline"
    }

    return query + expansions.get(query_type, " oncology cancer information")


def enrich_query(query: str) -> str:
    """Deduplicates tokens in query string."""
    tokens = query.split()
    unique = []
    seen = set()
    for t in tokens:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return " ".join(unique)[:300]


def choose_k(query_type: str) -> int:
    """Selects adaptive retrieval k value based on query complexity."""
    k_map = {
        "definition": 4, "yesno": 4, "symptoms": 6, "diagnosis": 6,
        "treatment": 7, "side_effects": 7, "comparison": 8, "staging": 7,
        "prognosis": 7, "risk_factors": 6, "prevention": 6, "clinical_trials": 8,
        "list": 8, "ranking": 10, "epidemiology": 7
    }
    return k_map.get(query_type, 5)


def detect_intent(query_type: str) -> str:
    """Maps query type to overall intent classification."""
    intent_map = {
        "definition": "factual", "symptoms": "factual", "diagnosis": "factual",
        "staging": "factual", "epidemiology": "factual", "yesno": "factual",
        "list": "enumeration", "ranking": "enumeration", "comparison": "analytical",
        "treatment": "clinical_guidance", "side_effects": "clinical_guidance",
        "prevention": "clinical_guidance", "risk_factors": "clinical_guidance",
        "prognosis": "outcome_analysis", "clinical_trials": "research"
    }
    return intent_map.get(query_type, "exploratory")


def extract_keywords(query: str) -> list:
    """Extracts non-stopword medical keywords from query."""
    stopwords = {
        "what", "is", "the", "of", "a", "an", "does", "can", "are", "and", "to",
        "in", "on", "for", "with", "by", "how", "why", "when", "where", "which",
        "who", "whom", "explain", "describe", "define", "list", "compare",
        "difference", "between", "versus", "vs", "tell", "show", "give"
    }

    tokens = tokenize(query)
    keywords = []
    seen = set()

    for token in tokens:
        if token in stopwords or len(token) < 3:
            continue
        if token not in seen:
            keywords.append(token)
            seen.add(token)

    return keywords[:10]


def lightweight_llm_expand(query: str) -> str:
    """Optional LLM query expansion prompt for complex analytical queries."""
    prompt = f"""
Expand this medical oncology retrieval query slightly.

STRICT RULES:
- Preserve the original user wording
- Keep concise
- Add ONLY medically useful retrieval keywords
- Maximum 8 added words
- NEVER add: legal, finance, prevalence, epidemiology, population studies, academic wording, literature review, retrieval query, study design
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
                    "temperature": 0, "top_p": 0.1, "top_k": 10,
                    "num_predict": 40, "keep_alive": "20m"
                }
            },
            timeout=20
        )
        text = response.json().get("response", "").strip()
        if len(text) > 5:
            return text
    except Exception:
        pass

    return query


def should_use_llm_expansion(query: str, intent: str, query_type: str) -> bool:
    """Determines whether LLM expansion is appropriate for query."""
    if len(tokenize(query)) < 6 or is_simple_query(query):
        return False

    if query_type in {"definition", "list", "ranking", "yesno", "symptoms", "diagnosis", "staging"}:
        return False

    return intent in {"analytical", "clinical_guidance", "outcome_analysis", "research", "exploratory"}


def domain_safe_expansion(original_query: str, expanded_query: str, query_type: str) -> str:
    """Filters domain-drift keywords from expanded query."""
    banned_patterns = [
        r"\blegal\b", r"\blaw\b", r"\bfinancial\b", r"\bfinance\b", r"\beconomic\b",
        r"\bstudy design\b", r"\bretrieval query\b", r"\bacademic\b", r"\bliterature review\b",
        r"\bmarketing\b", r"\bbusiness\b", r"\bpolitical\b", r"\bgovernment policy\b"
    ]

    safe = expanded_query.lower().strip()
    for pattern in banned_patterns:
        safe = re.sub(pattern, " ", safe, flags=re.IGNORECASE)

    safe = re.sub(r"\s+", " ", safe).strip()
    original = original_query.lower().strip()

    if not safe.startswith(original):
        safe = original + " " + safe

    original_tokens = tokenize(original)
    safe_tokens = tokenize(safe)

    added = [token for token in safe_tokens if token not in original_tokens]

    limits = {
        "definition": 6, "yesno": 6, "symptoms": 8, "diagnosis": 8,
        "staging": 8, "risk_factors": 8, "prevention": 8, "treatment": 10,
        "side_effects": 10, "prognosis": 10, "comparison": 12,
        "clinical_trials": 14, "ranking": 8, "list": 8, "general": 8
    }

    max_added_words = limits.get(query_type, 8)
    added = added[:max_added_words]

    seen = set()
    final_tokens = []
    for token in original_tokens + added:
        if token not in seen:
            final_tokens.append(token)
            seen.add(token)

    oncology_terms = {
        "cancer", "tumor", "tumour", "oncology", "breast", "lung", "colon",
        "rectal", "colorectal", "prostate", "ovarian", "cervical", "renal",
        "kidney", "pancreatic", "liver", "gastric", "stomach", "thyroid",
        "brain", "bladder", "esophageal", "chemotherapy", "immunotherapy",
        "radiotherapy", "radiation", "surgery", "resection", "hormone",
        "endocrine", "targeted", "diagnosis", "symptoms", "biopsy", "imaging",
        "screening", "treatment", "therapy", "management", "prognosis",
        "survival", "outcome", "mortality", "recurrence", "metastasis",
        "metastatic", "staging", "stage", "tnm", "carcinoma", "sarcoma",
        "lymphoma", "leukemia", "melanoma", "myeloma", "gist", "clinical",
        "trial", "toxicity", "adverse", "effects", "complications",
        "prevention", "egfr", "alk", "her2", "braf", "pd-l1", "pd-1", "car-t", "brca"
    }

    has_medical_signal = any(token in oncology_terms for token in final_tokens)
    if not has_medical_signal:
        return original

    return " ".join(final_tokens)


def process_query(query: str) -> dict:
    """Main LAQA processing entry point."""
    if not settings.is_laqa_enabled():
        return settings.build_raw_query_payload(query)

    query = clean_query(query)
    query_type = detect_query_type(query)
    intent = detect_intent(query_type)
    query_complexity = "simple" if is_simple_query(query) else "complex"

    expanded_query = deterministic_expansion(query, query_type)
    expansion_source = "deterministic"

    if should_use_llm_expansion(query, intent, query_type):
        expanded_query = lightweight_llm_expand(expanded_query)
        expansion_source = "llm"

    expanded_query = domain_safe_expansion(query, expanded_query, query_type)
    expanded_query = enrich_query(expanded_query)

    keywords = list(dict.fromkeys(extract_keywords(query) + extract_keywords(expanded_query)))
    query_metadata = classify_query_metadata(
        query=query,
        keywords=keywords,
        query_type=query_type,
        expanded_query=expanded_query
    )

    final_output = {
        "intent": intent,
        "query_type": query_type,
        "query_complexity": query_complexity,
        "keywords": keywords,
        "query_metadata": query_metadata,
        "expanded_query": expanded_query,
        "expansion_source": expansion_source,
        "retrieval_k": choose_k(query_type),
        "original_query": query
    }

    print(
        f"  🔍 Intent: {final_output.get('intent', 'N/A')}  |  "
        f"Type: {final_output.get('query_type', 'N/A')}  |  "
        f"Complexity: {final_output.get('query_complexity', 'N/A')}"
    )

    return final_output
