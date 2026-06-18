import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


CATEGORY_HINTS = {
    "symptoms": [
        "symptom", "symptoms", "sign", "signs", "presentation", "clinical presentation",
        "pain", "fatigue", "bleeding", "weight loss", "jaundice", "dyspnea"
    ],
    "diagnosis": [
        "diagnosis", "diagnostic", "biopsy", "imaging", "evaluation", "workup",
        "ct scan", "mri", "pet", "ultrasound", "laboratory", "tumor marker"
    ],
    "screening": [
        "screening", "early detection", "mammography", "colonoscopy", "pap smear",
        "psa", "low-dose ct", "surveillance"
    ],
    "pathology": [
        "pathology", "histology", "histopathology", "cytology", "grade",
        "biomarker", "immunohistochemistry", "molecular", "mutation"
    ],
    "treatment": [
        "treatment", "therapy", "management", "regimen", "intervention",
        "standard of care", "first-line", "second-line"
    ],
    "chemotherapy": [
        "chemotherapy", "chemo", "cytotoxic", "systemic therapy", "cisplatin",
        "carboplatin", "oxaliplatin", "paclitaxel", "docetaxel", "doxorubicin",
        "cyclophosphamide"
    ],
    "immunotherapy": [
        "immunotherapy", "checkpoint", "immune checkpoint", "pd-1", "pd-l1",
        "pdl1", "ctla-4", "nivolumab", "pembrolizumab", "atezolizumab", "durvalumab"
    ],
    "radiotherapy": [
        "radiotherapy", "radiation therapy", "radiation", "irradiation",
        "external beam", "brachytherapy", "stereotactic radiotherapy"
    ],
    "targeted_therapy": [
        "targeted therapy", "targeted treatment", "tyrosine kinase inhibitor",
        "egfr", "alk", "her2", "braf", "mek", "trastuzumab", "rituximab",
        "bevacizumab", "cetuximab"
    ],
    "surgery": [
        "surgery", "surgical", "resection", "operative", "operation",
        "mastectomy", "lumpectomy", "excision", "debulking"
    ],
    "staging": [
        "staging", "stage i", "stage ii", "stage iii", "stage iv", "tnm",
        "metastatic", "localized", "regional", "node-positive"
    ],
    "prognosis": [
        "prognosis", "survival", "overall survival", "progression-free survival",
        "disease-free survival", "outcome", "recurrence", "mortality", "remission"
    ],
    "epidemiology": [
        "epidemiology", "incidence", "prevalence", "risk factor", "risk factors",
        "population", "age-adjusted", "familial"
    ],
    "mechanism": [
        "mechanism", "pathogenesis", "oncogene", "tumor suppressor", "genetic",
        "molecular pathway", "dna repair", "carcinogenesis"
    ],
    "clinical_trials": [
        "clinical trial", "trial", "phase i", "phase ii", "phase iii",
        "randomized", "enrolled", "endpoint", "hazard ratio"
    ],
    "prevention": [
        "prevention", "preventive", "risk reduction", "vaccination", "hpv vaccine",
        "smoking cessation", "chemoprevention"
    ],
    "side_effects": [
        "side effect", "side effects", "adverse effect", "adverse event",
        "toxicity", "complication", "neutropenia", "neuropathy", "nausea"
    ],
    "palliative_care": [
        "palliative", "palliative care", "hospice", "symptom control",
        "end-of-life", "pain management"
    ],
    "survivorship": [
        "survivorship", "survivor", "follow-up care", "long-term follow-up",
        "quality of life", "rehabilitation", "late effects"
    ],
    "general": []
}

CANCER_HINTS = {
    "breast cancer": ["breast cancer", "breast carcinoma", "mammary"],
    "lung cancer": ["lung cancer", "nsclc", "sclc", "pulmonary"],
    "colorectal cancer": ["colorectal cancer", "colon cancer", "rectal cancer", "crc"],
    "prostate cancer": ["prostate cancer"],
    "ovarian cancer": ["ovarian cancer"],
    "cervical cancer": ["cervical cancer"],
    "renal cancer": ["renal cancer", "kidney cancer", "renal cell carcinoma", "rcc"],
    "bladder cancer": ["bladder cancer", "urothelial carcinoma"],
    "esophageal cancer": ["esophageal cancer", "oesophageal cancer"],
    "thyroid cancer": ["thyroid cancer", "papillary thyroid carcinoma"],
    "multiple myeloma": ["multiple myeloma", "myeloma"],
    "sarcoma": ["sarcoma", "osteosarcoma", "ewing sarcoma"],
    "neuroblastoma": ["neuroblastoma"],
    "retinoblastoma": ["retinoblastoma"],
    "wilms tumor": ["wilms tumor", "wilms tumour", "nephroblastoma"],
    "glioblastoma": ["glioblastoma"],
    "meningioma": ["meningioma"],
    "pancreatic cancer": ["pancreatic cancer", "pancreatic adenocarcinoma"],
    "gastric cancer": ["gastric cancer", "stomach cancer"],
    "liver cancer": ["liver cancer", "hepatocellular carcinoma", "hepatocellular", "hcc"],
    "cholangiocarcinoma": ["cholangiocarcinoma", "bile duct cancer"],
    "leukemia": ["leukemia", "leukaemia", "aml", "all", "cll", "cml"],
    "lymphoma": ["lymphoma", "hodgkin", "non-hodgkin"],
    "melanoma": ["melanoma"],
    "brain cancer": ["brain cancer", "glioma", "astrocytoma"],
    "head and neck cancer": ["head and neck", "oral cancer", "throat cancer", "laryngeal"],
    "general": []
}

TREATMENT_HINTS = {
    "chemotherapy": [
        "chemotherapy", "chemo", "cytotoxic", "platinum", "antineoplastic",
        "cisplatin", "carboplatin", "oxaliplatin", "paclitaxel", "docetaxel",
        "doxorubicin", "cyclophosphamide"
    ],
    "immunotherapy": [
        "immunotherapy", "checkpoint", "immune checkpoint", "pd-1", "pd-l1",
        "pdl1", "ctla-4", "nivolumab", "pembrolizumab", "atezolizumab", "durvalumab"
    ],
    "radiation therapy": [
        "radiotherapy", "radiation therapy", "radiation", "irradiation",
        "external beam radiation", "brachytherapy"
    ],
    "surgery": [
        "surgery", "surgical", "resection", "operative", "mastectomy",
        "lumpectomy", "excision", "debulking"
    ],
    "targeted therapy": [
        "targeted therapy", "targeted", "egfr", "alk", "her2", "braf", "mek",
        "trastuzumab", "rituximab", "bevacizumab", "cetuximab"
    ],
    "hormone therapy": [
        "hormone therapy", "hormonal therapy", "endocrine therapy", "hormone",
        "endocrine", "tamoxifen", "aromatase inhibitor", "androgen deprivation"
    ],
    "supportive care": [
        "supportive care", "supportive", "palliative", "symptom control",
        "pain management", "antiemetic", "nutrition", "psychosocial"
    ],
    "general": []
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "with", "without",
    "which", "who", "what", "when", "where", "why", "how", "does", "do",
    "can", "could", "may", "might", "should", "would", "will", "patient",
    "patients", "study", "studies", "report", "reports", "data", "analysis",
    "case", "cases", "group", "groups", "using", "used", "use", "based",
    "including", "include", "includes"
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9\-]+\b", (text or "").lower())


def _normalize_label(value: Any, default: str = "general") -> str:
    if value is None:
        return default
    if isinstance(value, list):
        if not value:
            return default
        value = str(value[0])
    if isinstance(value, str):
        value = value.strip().lower()
        return value or default
    return default


def _normalize_keywords(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = re.split(r"[,;|/]", value)
    if not isinstance(value, list):
        return []

    normalized = []
    for item in value:
        token = str(item).strip().lower()
        if token and token not in normalized:
            normalized.append(token)
    return normalized[:12]


def _phrase_count(text_lower: str, phrase: str) -> int:
    phrase = (phrase or "").strip().lower()
    if not phrase:
        return 0
    if re.fullmatch(r"[a-z0-9\-]+", phrase):
        return len(re.findall(rf"\b{re.escape(phrase)}\b", text_lower))
    return len(re.findall(rf"(?<!\w){re.escape(phrase)}(?!\w)", text_lower))


def _weighted_scores(text: str, hint_map: Dict[str, List[str]]) -> Dict[str, int]:
    text_lower = (text or "").lower()
    scores: Dict[str, int] = {}

    for label, hints in hint_map.items():
        if label == "general":
            continue

        score = 0
        for hint in hints:
            count = _phrase_count(text_lower, hint)
            if count:
                score += count * (3 if " " in hint else 1)

        if score > 0:
            scores[label] = score

    return scores


def _best_weighted_label(
    text: str,
    hint_map: Dict[str, List[str]],
    default: str = "general",
    boost: Optional[Tuple[str, int]] = None,
) -> str:
    scores = _weighted_scores(text, hint_map)
    if boost:
        boost_label, boost_score = boost
        if boost_label in hint_map and boost_label != "general":
            scores[boost_label] = scores.get(boost_label, 0) + boost_score

    if not scores:
        return default
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def _first_hint(text: str, hint_map: Dict[str, List[str]], default: str = "general") -> str:
    return _best_weighted_label(text, hint_map, default=default)


def detect_category(text: str, section: Optional[str] = None) -> str:
    section = _normalize_label(section, "general")
    boost = (section, 2) if section in CATEGORY_HINTS else None
    return _best_weighted_label(text, CATEGORY_HINTS, boost=boost)


def detect_cancer_type(text: str, document_metadata: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(document_metadata, dict):
        candidate = document_metadata.get("cancer_type")
        if isinstance(candidate, str) and candidate.strip().lower() != "general":
            return candidate.strip().lower()

        cancer_types = document_metadata.get("cancer_types") or []
        if isinstance(cancer_types, list) and cancer_types:
            first = str(cancer_types[0]).strip().lower()
            if first and first != "general":
                return first

    return _best_weighted_label(text, CANCER_HINTS)


def detect_treatment_type(text: str, document_metadata: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(document_metadata, dict):
        candidate = document_metadata.get("treatment_type")
        if isinstance(candidate, str) and candidate.strip().lower() != "general":
            return candidate.strip().lower()

    return _best_weighted_label(text, TREATMENT_HINTS)


def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    text_lower = (text or "").lower()
    phrase_scores: Dict[str, int] = {}
    phrase_sources = []

    for hint_map in (CATEGORY_HINTS, CANCER_HINTS, TREATMENT_HINTS):
        for phrases in hint_map.values():
            phrase_sources.extend(phrases)

    for phrase in set(phrase_sources):
        if not phrase or len(phrase) < 3:
            continue
        count = _phrase_count(text_lower, phrase)
        if count:
            phrase_scores[phrase] = count * (3 if " " in phrase else 2)

    token_counts = Counter()
    for token in tokenize(text):
        if len(token) < 3 or token in STOPWORDS or token.isdigit():
            continue
        token_counts[token] += 1

    oncology_terms = set()
    for phrases in list(CANCER_HINTS.values()) + list(TREATMENT_HINTS.values()):
        for phrase in phrases:
            oncology_terms.update(tokenize(phrase))

    token_scores = {
        token: count + (3 if token in oncology_terms else 0) + (1 if len(token) >= 8 else 0)
        for token, count in token_counts.items()
    }

    ranked_phrases = sorted(
        phrase_scores.items(),
        key=lambda item: (item[1], len(item[0])),
        reverse=True
    )
    ranked_tokens = sorted(
        token_scores.items(),
        key=lambda item: (item[1], len(item[0])),
        reverse=True
    )

    keywords: List[str] = []
    for phrase, _ in ranked_phrases:
        if phrase not in keywords:
            keywords.append(phrase)
        if len(keywords) >= max_keywords:
            return keywords[:max_keywords]

    for token, _ in ranked_tokens:
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= max_keywords:
            break

    return keywords[:max_keywords]


def heuristic_chunk_metadata(
    text: str,
    section: Optional[str] = None,
    source_document: Optional[str] = None,
    document_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    section = _normalize_label(section, "general")
    category = detect_category(text, section=section)
    cancer_type = detect_cancer_type(text, document_metadata=document_metadata)
    treatment_type = detect_treatment_type(text, document_metadata=document_metadata)
    keywords = extract_keywords(text, max_keywords=8)

    sub_category = section
    if sub_category == "general":
        if treatment_type != "general":
            sub_category = treatment_type
        elif cancer_type != "general":
            sub_category = cancer_type
        elif category != "general":
            sub_category = category

    return {
        "category": category,
        "sub_category": sub_category,
        "keywords": keywords,
        "cancer_type": cancer_type,
        "treatment_type": treatment_type,
        "source_document": source_document or "unknown",
        "section": section,
    }


def _heuristic_chunk_metadata(
    text: str,
    section: Optional[str] = None,
    source_document: Optional[str] = None,
) -> Dict[str, Any]:
    return heuristic_chunk_metadata(text, section=section, source_document=source_document)


def _heuristic_query_metadata(
    query: str,
    keywords: Optional[List[str]] = None,
    query_type: Optional[str] = None,
) -> Dict[str, Any]:
    query_text = query or ""
    category = query_type or _first_hint(query_text, CATEGORY_HINTS)
    cancer_type = _first_hint(query_text, CANCER_HINTS)
    treatment_type = _first_hint(query_text, TREATMENT_HINTS)
    merged_keywords = _normalize_keywords(keywords)

    for token in extract_keywords(query_text, max_keywords=8):
        if token not in merged_keywords:
            merged_keywords.append(token)

    return {
        "category": category or "general",
        "keywords": merged_keywords[:8],
        "cancer_type": cancer_type,
        "treatment_type": treatment_type
    }


def classify_chunk_metadata(
    text: str,
    section: Optional[str] = None,
    source_document: Optional[str] = None,
    document_metadata: Optional[Dict[str, Any]] = None,
    heuristic_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return heuristic_chunk_metadata(
        text=text,
        section=section,
        source_document=source_document,
        document_metadata=document_metadata,
    )


def classify_query_metadata(
    query: str,
    keywords: Optional[List[str]] = None,
    query_type: Optional[str] = None,
    expanded_query: Optional[str] = None,
) -> Dict[str, Any]:
    base_query = expanded_query or query or ""
    return _heuristic_query_metadata(base_query, keywords=keywords, query_type=query_type)


def metadata_overlap_score(left_keywords: List[str], right_keywords: List[str]) -> float:
    left = {token.strip().lower() for token in left_keywords if str(token).strip()}
    right = {token.strip().lower() for token in right_keywords if str(token).strip()}
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    denominator = max(len(left), len(right), 1)
    return round(min(overlap / denominator, 1.0), 3)


def metadata_match_score(query_metadata: Dict[str, Any], chunk_metadata: Dict[str, Any]) -> float:
    query_metadata = query_metadata or {}
    chunk_metadata = chunk_metadata or {}

    category_match = 1.0 if _normalize_label(query_metadata.get("category")) == _normalize_label(chunk_metadata.get("category")) else 0.0
    cancer_type_match = 1.0 if _normalize_label(query_metadata.get("cancer_type")) != "general" and _normalize_label(query_metadata.get("cancer_type")) == _normalize_label(chunk_metadata.get("cancer_type")) else 0.0
    treatment_type_match = 1.0 if _normalize_label(query_metadata.get("treatment_type")) != "general" and _normalize_label(query_metadata.get("treatment_type")) == _normalize_label(chunk_metadata.get("treatment_type")) else 0.0
    keyword_overlap = metadata_overlap_score(
        _normalize_keywords(query_metadata.get("keywords")),
        _normalize_keywords(chunk_metadata.get("keywords"))
    )

    score = (
        0.4 * category_match
        + 0.3 * cancer_type_match
        + 0.2 * treatment_type_match
        + 0.1 * keyword_overlap
    )
    return round(max(0.0, min(score, 1.0)), 3)


def normalize_metadata_record(record: Dict[str, Any], fallback_id: Optional[str] = None) -> Dict[str, Any]:
    record = dict(record or {})
    metadata = record.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {
            "category": _normalize_label(record.get("category"), record.get("section", "general")),
            "sub_category": _normalize_label(record.get("sub_category"), record.get("section", "general")),
            "keywords": _normalize_keywords(record.get("keywords")),
            "cancer_type": _normalize_label(record.get("cancer_type")),
            "treatment_type": _normalize_label(record.get("treatment_type")),
            "source_document": str(record.get("source_document") or record.get("doc_id") or fallback_id or "unknown"),
            "section": _normalize_label(record.get("section"), "general")
        }
    else:
        metadata = {
            "category": _normalize_label(metadata.get("category"), record.get("section", "general")),
            "sub_category": _normalize_label(metadata.get("sub_category"), record.get("section", "general")),
            "keywords": _normalize_keywords(metadata.get("keywords")),
            "cancer_type": _normalize_label(metadata.get("cancer_type")),
            "treatment_type": _normalize_label(metadata.get("treatment_type")),
            "source_document": str(metadata.get("source_document") or record.get("doc_id") or fallback_id or "unknown"),
            "section": _normalize_label(metadata.get("section"), record.get("section", "general"))
        }

    record["metadata"] = metadata
    record.setdefault("id", fallback_id)
    record.setdefault("section", metadata["section"])
    record.setdefault("source_document", metadata["source_document"])
    record.setdefault("category", metadata["category"])
    record.setdefault("sub_category", metadata["sub_category"])
    record.setdefault("keywords", metadata["keywords"])
    record.setdefault("cancer_type", metadata["cancer_type"])
    record.setdefault("treatment_type", metadata["treatment_type"])
    return record
