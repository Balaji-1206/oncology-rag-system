import os
import re
from PyPDF2 import PdfReader

from utils.metadata_tools import (
    classify_chunk_metadata
)


# =========================================================
# 🔹 LOAD PDFS
# =========================================================
def load_pdfs(folder_path, limit=None):

    documents = []

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if limit is not None:
        pdf_files = pdf_files[:limit]

    print(f"📄 PDFs selected: {len(pdf_files)}")

    for file in pdf_files:

        path = os.path.join(folder_path, file)
        print(f"📥 Loading: {file}")

        try:
            reader = PdfReader(path)
            text = ""

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            text = text.strip()

            if len(text) < 500:
                print(f"⚠️ Skipping weak PDF: {file}")
                continue

            documents.append({"id": file, "text": text})

        except Exception as e:
            print(f"❌ Failed loading {file}:", e)

    return documents


# =========================================================
# 🔹 CLEAN TEXT
# =========================================================
def clean_text(text):

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[-=_]{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# 🔹 SECTION DETECTION
# =========================================================
def detect_section(text):

    text_lower = text.lower()

    sections = {
        "symptoms": [
            "symptoms",
            "clinical presentation",
            "signs",
            "manifestations",
            "complaints",
            "presenting symptoms"
        ],
        "diagnosis": [
            "diagnosis",
            "diagnostic",
            "evaluation",
            "screening",
            "biopsy",
            "imaging",
            "pathology",
            "laboratory findings",
            "diagnostic criteria"
        ],
        "treatment": [
            "treatment",
            "therapy",
            "management",
            "treatment options",
            "therapeutic approach",
            "intervention"
        ],
        "chemotherapy": [
            "chemotherapy",
            "cytotoxic",
            "adjuvant chemotherapy",
            "neoadjuvant chemotherapy",
            "chemotherapeutic"
        ],
        "radiotherapy": [
            "radiotherapy",
            "radiation therapy",
            "irradiation",
            "external beam radiation",
            "brachytherapy"
        ],
        "immunotherapy": [
            "immunotherapy",
            "immune checkpoint",
            "pembrolizumab",
            "nivolumab",
            "atezolizumab",
            "car-t"
        ],
        "targeted_therapy": [
            "targeted therapy",
            "targeted treatment",
            "egfr",
            "alk",
            "her2",
            "braf",
            "tyrosine kinase inhibitor"
        ],
        "surgery": [
            "surgery",
            "surgical",
            "resection",
            "operation",
            "mastectomy",
            "lumpectomy"
        ],
        "staging": [
            "staging",
            "tnm",
            "stage i",
            "stage ii",
            "stage iii",
            "stage iv",
            "tumor stage"
        ],
        "prognosis": [
            "prognosis",
            "survival",
            "overall survival",
            "disease-free survival",
            "outcome",
            "mortality"
        ],
        "side_effects": [
            "side effects",
            "adverse effects",
            "toxicity",
            "complications",
            "treatment toxicity",
            "adverse events"
        ],
        "prevention": [
            "prevention",
            "risk reduction",
            "screening recommendations",
            "preventive measures"
        ],
        "epidemiology": [
            "incidence",
            "prevalence",
            "epidemiology",
            "risk factors",
            "population"
        ],
        "mechanism": [
            "mechanism",
            "pathogenesis",
            "molecular",
            "genetic",
            "mutation",
            "oncogene",
            "tumor suppressor"
        ],
        "clinical_trials": [
            "clinical trial",
            "phase i",
            "phase ii",
            "phase iii",
            "study results",
            "trial"
        ]
    }

    scores = {}

    for section, keywords in sections.items():
        score = 0
        for kw in keywords:
            if kw in text_lower:
                score += 1
        if score > 0:
            scores[section] = score

    if scores:
        ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return {
            "primary": ordered[0],
            "secondary": ordered[1:3]
        }

    return {
        "primary": "general",
        "secondary": []
    }


# =========================================================
# 🔹 QUALITY FILTER
# =========================================================
def is_good_chunk(text):

    text_lower = text.lower()
    words = text.split()

    if len(words) < 40:
        return False

    if len(words) > 700:
        return False

    bad_patterns = [
        "table of contents",
        "copyright",
        "all rights reserved",
        "isbn",
        "doi.org",
        "www.",
        "http://",
        "https://"
    ]

    for p in bad_patterns:
        if p in text_lower:
            return False

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.55:
        return False

    return True


# =========================================================
# 🔹 SENTENCE SPLITTER
# =========================================================
def split_sentences(text):

    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) > 20:
            cleaned.append(s)

    return cleaned


# =========================================================
# 🔹 SEMANTIC CHUNKER (OPTIMIZED)
# =========================================================
def fallback_chunk(text, max_chunk_words=320, overlap_sentences=2):

    print("🧩 Creating semantic chunks...")

    sentences = split_sentences(text)
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())

        if sent_words < 5:
            continue

        if current_words + sent_words > max_chunk_words:

            chunk = " ".join(current)

            if chunk.strip():
                chunks.append(chunk)

            overlap = current[-overlap_sentences:]
            current = overlap + [sent]
            current_words = sum(len(s.split()) for s in current)

        else:
            current.append(sent)
            current_words += sent_words

    if current:
        chunks.append(" ".join(current))

    print(f"✅ Semantic chunks created: {len(chunks)}")
    return chunks


# =========================================================
# 🔹 FAST DEDUPLICATION
# =========================================================
def deduplicate_chunks(chunks):

    print("🔹 Deduplicating chunks...")

    unique = []
    seen_hashes = set()

    for chunk in chunks:
        text = chunk["text"]
        key = hash(text[:300].lower())

        if key in seen_hashes:
            continue

        seen_hashes.add(key)
        unique.append(chunk)

    print(f"✅ Chunks after deduplication: {len(unique)}")
    return unique


# =========================================================
# 🔹 MAIN PIPELINE
# =========================================================
def chunk_text(documents):

    final_chunks = []
    chunk_counter = 0
    total_chunks = 0

    for i, doc in enumerate(documents):

        file_id = doc["id"]
        text = clean_text(doc["text"])

        print(f"\n📄 Processing {i+1}/{len(documents)}: {file_id}")

        chunks = fallback_chunk(text)
        print(f"📦 Candidate chunks: {len(chunks)}")

        kept = 0

        for c in chunks:

            c = c.strip()

            if not is_good_chunk(c):
                continue

            section_info = detect_section(c)
            primary_section = section_info["primary"]
            secondary_sections = section_info["secondary"]

            total_chunks += 1

            chunk_metadata = classify_chunk_metadata(
                c,
                section=primary_section,
                source_document=file_id
            )

            final_chunks.append({
                "id": f"doc_{chunk_counter}",
                "text": c,
                "doc_id": file_id,
                "source_document": file_id,
                "section": primary_section,
                "metadata": chunk_metadata,
                "category": chunk_metadata.get("category", primary_section),
                "sub_category": chunk_metadata.get("sub_category", primary_section),
                "categories": [primary_section] + secondary_sections,
                "keywords": chunk_metadata.get("keywords", []),
                "cancer_type": chunk_metadata.get("cancer_type", "general"),
                "treatment_type": chunk_metadata.get("treatment_type", "general"),
                "length": len(c.split())
            })

            chunk_counter += 1
            kept += 1

        print(f"✅ Valid chunks kept: {kept}")

    final_chunks = deduplicate_chunks(final_chunks)

    print("\n🔍 SAMPLE CHUNKS:\n")
    for c in final_chunks[:3]:
        print(c)
        print()

    print(f"Chunks processed: {total_chunks}")
    print(f"Heuristic metadata generated: {total_chunks}")
    print("LLM metadata calls: 0")
    print("LLM usage rate: 0.00%")

    print(f"\n✅ Total cleaned chunks: {len(final_chunks)}")

    return final_chunks
