# ============================================================
# Build RAG index from a Hugging Face medical dataset.
# Run once: creates chroma_db/ and fills it with embedded chunks.
# ============================================================

import os
from datasets import load_dataset
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------
# PubMed QA: medical Q&A with abstract context + long answer (good for RAG)
DATASET_NAME = "pubmed_qa"
DATASET_SUBSET = "pqa_labeled"  # or "pqa_artificial"; use None for base
SPLIT = "train"
MAX_SAMPLES = 2000  # cap for quick build; set None for full
CHUNK_SIZE = 400     # chars per chunk (simple split)
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "med_rag"
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, no GPU required


def chunk_text(text: str, size: int) -> list[str]:
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # break at last space to avoid cutting words
            last_space = text.rfind(" ", start, end + 1)
            if last_space > start:
                end = last_space + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def docs_from_pubmed_qa(records) -> list[tuple[str, str]]:
    """(id, text) for each document. PubMed QA: context (list of strings) + long_answer."""
    out = []
    for i, r in enumerate(records):
        ctx = r.get("context") or r.get("contexts")
        if isinstance(ctx, list):
            ctx = " ".join(c for c in ctx if isinstance(c, str))
        elif not isinstance(ctx, str):
            ctx = ""
        long_answer = r.get("long_answer") or r.get("final_decision") or ""
        if isinstance(long_answer, list):
            long_answer = " ".join(str(x) for x in long_answer)
        question = r.get("question") or ""
        text = f"Question: {question}\n\nContext: {ctx}\n\nAnswer: {long_answer}"
        text = " ".join(text.split())  # normalize whitespace
        if len(text) < 50:
            continue
        out.append((str(i), text))
    return out


def docs_from_generic(records, text_keys=None) -> list[tuple[str, str]]:
    """Fallback: concat values from text_keys (e.g. ['question','answer']) per row."""
    text_keys = text_keys or ["question", "answer", "context", "text"]
    out = []
    for i, r in enumerate(records):
        parts = []
        for k in text_keys:
            if k in r and r[k]:
                v = r[k]
                if isinstance(v, list):
                    v = " ".join(str(x) for x in v)
                parts.append(str(v))
        text = " ".join(parts).strip()
        if len(text) < 30:
            continue
        out.append((str(i), text))
    return out


def main():
    print("Loading dataset:", DATASET_NAME, SPLIT)
    ds = load_dataset(DATASET_NAME, DATASET_SUBSET, split=SPLIT, trust_remote_code=True)
    if MAX_SAMPLES:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

    # Build (id, text) docs - try PubMed QA shape first
    if "long_answer" in ds.column_names or "context" in ds.column_names:
        docs = docs_from_pubmed_qa(ds)
    else:
        docs = docs_from_generic(ds)

    if not docs:
        raise ValueError("No documents produced. Check dataset columns.")

    # Chunk
    all_chunks = []
    chunk_metas = []
    for doc_id, text in docs:
        for c in chunk_text(text, CHUNK_SIZE):
            if len(c) < 30:
                continue
            all_chunks.append(c)
            chunk_metas.append({"doc_id": doc_id})

    print("Chunks to embed:", len(all_chunks))
    print("Loading embedder:", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(COLLECTION_NAME, metadata={"description": "Medical RAG"})

    # Chroma wants list of ids; we use index
    ids = [str(i) for i in range(len(all_chunks))]
    coll.add(ids=ids, embeddings=embeddings.tolist(), documents=all_chunks, metadatas=chunk_metas)
    print("Index saved to", os.path.abspath(CHROMA_DIR), "collection:", COLLECTION_NAME)


if __name__ == "__main__":
    main()
