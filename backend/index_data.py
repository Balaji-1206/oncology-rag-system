import faiss
import numpy as np
import pickle
import os
import re
import json
from datetime import datetime

from rank_bm25 import BM25Okapi

import settings

from modules.embeddings.mrl_embeddings import (
    get_mrl_embedding
)

from modules.chunking.chunker import (
    load_pdfs,
    chunk_text
)

# =========================================================
# 🔹 PATHS
# =========================================================
DATA_PATH = "backend/data/oncology_docs"

# Determine target database based on MRL setting
target_db = "mrl" if settings.is_mrl_enabled() else "full"
SAVE_PATH = f"backend/database/{target_db}"

# Ensure base database directory exists
os.makedirs("backend/database", exist_ok=True)

# Create target database directory
os.makedirs(SAVE_PATH, exist_ok=True)

print(f"📦 Target database: {SAVE_PATH}")
print(f"📦 MRL mode: {'ENABLED' if settings.is_mrl_enabled() else 'DISABLED'}")


# =========================================================
# 🔹 TOKENIZER
# =========================================================
def tokenize(text):

    text = text.lower()

    return re.findall(
        r"\b[a-zA-Z0-9\-]+\b",
        text
    )


# =========================================================
# 🔹 DEDUPLICATION
# =========================================================
def deduplicate_documents(documents):

    unique = []

    seen = set()

    for doc in documents:

        text = doc["text"].strip()

        key = text[:300].lower()

        if key in seen:
            continue

        seen.add(key)

        unique.append(doc)

    return unique


# =========================================================
# 🔹 LOAD PDFS
# =========================================================
print("🔹 Loading PDFs...")
# =========================================================
# 🔹 NUMBER OF PDFs TO INDEX
# =========================================================
PDF_LIMIT = 25

raw_docs = load_pdfs(
    DATA_PATH,
    limit=PDF_LIMIT
)

print(f"📄 PDFs loaded: {len(raw_docs)}")


# =========================================================
# 🔹 CHUNKING
# =========================================================
print("🔹 Chunking...")

documents = chunk_text(raw_docs)

# =========================================================
# 🔹 DEDUPLICATION
# =========================================================
documents = deduplicate_documents(
    documents
)

print(
    f"✅ Final cleaned chunks: {len(documents)}"
)

# =========================================================
# 🔹 EXTRACT FIELDS
# =========================================================
texts = []

ids = []

sections = []

chunk_metadata = {}

for doc in documents:

    text = doc["text"]

    doc_id = doc["id"]

    section = doc.get(
        "section",
        "general"
    )

    texts.append(text)

    ids.append(doc_id)

    sections.append(section)

    # 🔥 metadata map
    chunk_metadata[doc_id] = {

        "section": section,

        "source_doc": doc.get(
            "doc_id",
            "unknown"
        ),

        "metadata": doc.get(
            "metadata",
            ""
        ),

        "length": doc.get(
            "length",
            len(text.split())
        )
    }

print(f"📚 Total chunks: {len(texts)}")

# =========================================================
# 🔹 EMBEDDINGS
# =========================================================
print("🔹 Creating MRL embeddings...")

embedding_dimension = settings.effective_embedding_dimension()

if settings.is_mrl_enabled():

    print(
        f"MRL enabled: indexing with dimension {embedding_dimension}"
    )

else:

    print(
        f"MRL disabled: indexing with full dimension {embedding_dimension}"
    )

doc_embeddings = get_mrl_embedding(
    texts,
    dim=embedding_dimension
)

# =========================================================
# 🔹 NORMALIZATION
# =========================================================
doc_embeddings = np.array(
    doc_embeddings,
    dtype=np.float32
)

faiss.normalize_L2(
    doc_embeddings
)

dimension = doc_embeddings.shape[1]

print(
    f"📐 Embedding dimension: {dimension}"
)

print(
    f"📐 Active embedding dimension: {embedding_dimension}"
)

if dimension != embedding_dimension:

    raise ValueError(
        f"Embedding dimension mismatch during indexing: "
        f"got {dimension}, expected {embedding_dimension}"
    )

# =========================================================
# 🔹 FAISS
# =========================================================
print("🔹 Building FAISS index...")

index = faiss.IndexFlatIP(
    dimension
)

index.add(doc_embeddings)

print(
    f"✅ Indexed vectors: {index.ntotal}"
)

# =========================================================
# 🔹 BM25
# =========================================================
print("🔹 Building BM25...")

tokenized_docs = [

    tokenize(text)

    for text in texts
]

bm25 = BM25Okapi(
    tokenized_docs
)

# =========================================================
# 🔹 SAVE FAISS
# =========================================================
print("💾 Saving FAISS index...")

faiss_index_path = f"{SAVE_PATH}/faiss.index"

if os.path.exists(faiss_index_path):

    old_index = faiss.read_index(
        faiss_index_path
    )

    print(
        f"🧹 Removing old FAISS index "
        f"(dim={old_index.d}, vectors={old_index.ntotal})"
    )

    os.remove(
        faiss_index_path
    )

faiss.write_index(
    index,
    faiss_index_path
)

# =========================================================
# 🔹 SAVE FULL DOCS
# =========================================================
with open(
    f"{SAVE_PATH}/docs.pkl",
    "wb"
) as f:

    pickle.dump(
        documents,
        f
    )

# =========================================================
# 🔹 SAVE BM25
# =========================================================
with open(
    f"{SAVE_PATH}/bm25.pkl",
    "wb"
) as f:

    pickle.dump(
        bm25,
        f
    )

# =========================================================
# 🔹 SAVE IDS
# =========================================================
with open(
    f"{SAVE_PATH}/ids.pkl",
    "wb"
) as f:

    pickle.dump(
        ids,
        f
    )

# =========================================================
# 🔹 SAVE TEXT MAP
# =========================================================
id_to_text = {

    doc["id"]: doc["text"]

    for doc in documents
}

with open(
    f"{SAVE_PATH}/id_to_text.pkl",
    "wb"
) as f:

    pickle.dump(
        id_to_text,
        f
    )

# =========================================================
# 🔹 SAVE CHUNK METADATA
# =========================================================
with open(
    f"{SAVE_PATH}/chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        chunk_metadata,
        f
    )

# =========================================================
# 🔹 SAVE SECTION MAP
# =========================================================
section_map = {

    doc["id"]: doc.get(
        "section",
        "general"
    )

    for doc in documents
}

with open(
    f"{SAVE_PATH}/section_map.pkl",
    "wb"
) as f:

    pickle.dump(
        section_map,
        f
    )

# =========================================================
# 🔹 STATS
# =========================================================
with open(
    f"{SAVE_PATH}/index_settings.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "enable_mrl": settings.is_mrl_enabled(),
            "embedding_dimension": dimension,
            "mrl_dimension": settings.MRL_DIMENSION,
            "full_embedding_dimension": settings.FULL_EMBEDDING_DIMENSION
        },
        f,
        indent=2,
        sort_keys=True
    )

# =========================================================
# 🔹 SAVE METADATA.JSON
# =========================================================
metadata = {
    "mrl_enabled": settings.is_mrl_enabled(),
    "embedding_dimension": dimension,
    "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
    "created_at": datetime.utcnow().isoformat() + "Z",
    "version": 2,
    "chunks_count": len(texts),
    "documents_count": len(documents)
}

with open(
    f"{SAVE_PATH}/metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        sort_keys=True
    )

print(f"✅ Saved metadata: {SAVE_PATH}/metadata.json")

section_counts = {}

for s in sections:

    section_counts[s] = (
        section_counts.get(s, 0)
        + 1
    )

print("\n📊 SECTION DISTRIBUTION:")

for k, v in section_counts.items():

    print(f"{k}: {v}")

# =========================================================
# 🔹 COMPLETE
# =========================================================
print("\n✅ Indexing complete!")

print(f"📁 Saved to: {SAVE_PATH}")
