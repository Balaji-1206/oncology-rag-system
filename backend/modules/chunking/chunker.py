import re
import requests
import os

from PyPDF2 import PdfReader

# =========================================================
# 🔹 CONFIG
# =========================================================
OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "phi3:mini"


# =========================================================
# 🔹 LOAD PDFS
# =========================================================
# =========================================================
# 🔹 LOAD PDFS
# =========================================================
def load_pdfs(
    folder_path,
    limit=None
):

    documents = []

    pdf_files = [

        f for f in os.listdir(folder_path)

        if f.endswith(".pdf")
    ]

    # =====================================================
    # 🔹 LIMIT PDF COUNT
    # =====================================================
    if limit is not None:

        pdf_files = pdf_files[:limit]

    print(
        f"📄 PDFs selected: {len(pdf_files)}"
    )

    for file in pdf_files:

        path = os.path.join(
            folder_path,
            file
        )

        print(f"📥 Loading: {file}")

        try:

            reader = PdfReader(path)

            text = ""

            for page in reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

            text = text.strip()

            # Skip tiny/broken PDFs
            if len(text) < 500:

                print(
                    f"⚠️ Skipping weak PDF: {file}"
                )

                continue

            documents.append({

                "id": file,

                "text": text
            })

        except Exception as e:

            print(
                f"❌ Failed loading {file}:",
                e
            )

    return documents


# =========================================================
# 🔹 CLEAN TEXT
# =========================================================
def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[-=_]{2,}",
        " ",
        text
    )

    text = re.sub(
        r"[^\x00-\x7F]+",
        " ",
        text
    )

    text = re.sub(
        r"\.{2,}",
        ".",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

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
            "signs"
        ],

        "diagnosis": [
            "diagnosis",
            "diagnostic",
            "evaluation"
        ],

        "treatment": [
            "treatment",
            "therapy",
            "management"
        ],

        "staging": [
            "staging",
            "tnm"
        ],

        "prognosis": [
            "prognosis",
            "survival"
        ],

        "mechanism": [
            "mechanism",
            "pathogenesis",
            "molecular"
        ]
    }

    for section, kws in sections.items():

        for kw in kws:

            if kw in text_lower:
                return section

    return "general"


# =========================================================
# 🔹 QUALITY FILTER
# =========================================================
# =========================================================
# 🔹 QUALITY FILTER
# =========================================================
def is_good_chunk(text):

    text_lower = text.lower()

    words = text.split()

    # =====================================================
    # 🔹 TOO SHORT
    # =====================================================
    if len(words) < 40:
        return False

    # =====================================================
    # 🔹 TOO LONG
    # =====================================================
    if len(words) > 700:
        return False

    # =====================================================
    # 🔹 HARD GARBAGE FILTERS
    # =====================================================
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

    # =====================================================
    # 🔹 CHARACTER QUALITY
    # =====================================================
    alpha_ratio = sum(

        c.isalpha()

        for c in text
    ) / max(len(text), 1)

    # Too much OCR junk
    if alpha_ratio < 0.55:
        return False

    # =====================================================
    # 🔹 MEDICAL TERM BOOST
    # =====================================================
    medical_terms = [

        "cancer",
        "tumor",
        "tumour",
        "therapy",
        "treatment",
        "diagnosis",
        "metastasis",
        "oncology",
        "patient",
        "survival",
        "immune",
        "clinical",
        "carcinoma",
        "chemotherapy",
        "radiotherapy",
        "biopsy",
        "immunotherapy"
    ]

    if any(
        term in text_lower
        for term in medical_terms
    ):
        return True

    # =====================================================
    # 🔹 OTHERWISE KEEP
    # =====================================================
    return True


# =========================================================
# 🔹 SENTENCE SPLITTER
# =========================================================
def split_sentences(text):

    text = re.sub(
        r"([a-z])([A-Z])",
        r"\1. \2",
        text
    )

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    cleaned = []

    for s in sentences:

        s = s.strip()

        if len(s) > 20:
            cleaned.append(s)

    return cleaned

# =========================================================
# 🔹 SEMANTIC CHUNKER (OPTIMIZED)
# =========================================================
def fallback_chunk(
    text,
    max_chunk_words=320,
    overlap_sentences=2
):

    print("🧩 Creating semantic chunks...")

    sentences = split_sentences(text)

    chunks = []

    current = []

    current_words = 0

    for sent in sentences:

        sent_words = len(
            sent.split()
        )

        # =====================================================
        # 🔹 SKIP VERY SHORT SENTENCES
        # =====================================================
        if sent_words < 5:
            continue

        # =====================================================
        # 🔹 CREATE CHUNK
        # =====================================================
        if (
            current_words + sent_words
            > max_chunk_words
        ):

            chunk = " ".join(current)

            if chunk.strip():

                chunks.append(chunk)

            # =================================================
            # 🔹 SMART OVERLAP
            # =================================================
            overlap = current[
                -overlap_sentences:
            ]

            current = overlap + [sent]

            current_words = sum(

                len(s.split())

                for s in current
            )

        else:

            current.append(sent)

            current_words += sent_words

    # =========================================================
    # 🔹 LAST CHUNK
    # =========================================================
    if current:

        chunks.append(
            " ".join(current)
        )

    print(
        f"✅ Semantic chunks created: "
        f"{len(chunks)}"
    )

    return chunks
# =========================================================
# =========================================================
# 🔹 AGENTIC ENRICHMENT
# =========================================================
# =========================================================
# 🔹 DOCUMENT-LEVEL AGENTIC ENRICHMENT
# =========================================================
def llm_chunk(text):

    print("🤖 Generating document metadata...")

    short_text = text[:4000]

    prompt = f"""
Analyze this oncology medical document.

Return ONLY JSON.

FORMAT:
{{
  "document_type": "...",
  "main_topics": ["...", "..."],
  "cancer_types": ["...", "..."],
  "summary": "..."
}}

TEXT:
{short_text}
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {

                    "temperature": 0,

                    "top_p": 0.1,

                    "num_predict": 180,

                    "keep_alive": "10m"
                }
            },
            timeout=40
        )

        raw = response.json().get(
            "response",
            ""
        )

        print("✅ Metadata generated")

        return raw

    except Exception as e:

        print(
            "⚠️ Metadata generation failed:",
            e
        )

        return ""
# =========================================================
# 🔹 FAST DEDUPLICATION
# =========================================================
def deduplicate_chunks(chunks):

    print("🔹 Deduplicating chunks...")

    unique = []

    seen_hashes = set()

    for chunk in chunks:

        text = chunk["text"]

        key = hash(
            text[:300].lower()
        )

        if key in seen_hashes:
            continue

        seen_hashes.add(key)

        unique.append(chunk)

    print(
        f"✅ Chunks after deduplication: "
        f"{len(unique)}"
    )

    return unique


# =========================================================
# 🔹 MAIN PIPELINE
# =========================================================
# =========================================================
# 🔹 MAIN PIPELINE
# =========================================================
# =========================================================
# 🔹 MAIN PIPELINE
# =========================================================
def chunk_text(documents):

    final_chunks = []

    chunk_counter = 0

    for i, doc in enumerate(documents):

        file_id = doc["id"]

        text = clean_text(
            doc["text"]
        )

        print(
            f"\n📄 Processing "
            f"{i+1}/{len(documents)}: "
            f"{file_id}"
        )

        # =====================================================
        # 🔹 STEP 1: SEMANTIC CHUNKING
        # =====================================================
        chunks = fallback_chunk(text)

        # =====================================================
        # 🔹 STEP 2: DOCUMENT METADATA
        # =====================================================
        metadata = llm_chunk(text)

        print(
            f"📦 Candidate chunks: "
            f"{len(chunks)}"
        )

        kept = 0

        for c in chunks:

            c = c.strip()

            if not is_good_chunk(c):
                continue

            section = detect_section(c)

            final_chunks.append({

                "id": f"doc_{chunk_counter}",

                "text": c,

                "doc_id": file_id,

                "section": section,

                "metadata": metadata,

                "length": len(
                    c.split()
                )
            })

            chunk_counter += 1

            kept += 1

        print(
            f"✅ Valid chunks kept: "
            f"{kept}"
        )

    # =========================================================
    # 🔹 DEDUPLICATION
    # =========================================================
    final_chunks = deduplicate_chunks(
        final_chunks
    )

    print("\n🔍 SAMPLE CHUNKS:\n")

    for c in final_chunks[:3]:

        print(c)

        print()

    print(
        f"\n✅ Total cleaned chunks: "
        f"{len(final_chunks)}"
    )

    return final_chunks