import fitz
import docx
import nltk
import numpy as np
import faiss
import pickle, os
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE    = 3
CHUNK_OVERLAP = 1
INDEX_DIR     = "faiss_index"


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


def extract_pdf(path: str) -> list[dict]:
    doc      = fitz.open(path)
    doc_name = Path(path).stem
    pages    = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"text": text, "page": i, "doc_name": doc_name})
    return pages


def extract_docx(path: str) -> list[dict]:
    doc      = docx.Document(path)
    doc_name = Path(path).stem
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": full_text, "page": 1, "doc_name": doc_name}]


def semantic_chunk(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        sentences = nltk.sent_tokenize(page["text"])
        for i in range(0, len(sentences), CHUNK_SIZE - CHUNK_OVERLAP):
            window = sentences[i: i + CHUNK_SIZE]
            if not window:
                continue
            chunks.append({
                "text":     " ".join(window),
                "page":     page["page"],
                "doc_name": page["doc_name"],
            })
    return chunks


def build_index(chunks: list[dict]) -> tuple:
    embedder = get_embedder()
    texts    = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks…")
    embeddings = embedder.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    from rank_bm25 import BM25Okapi
    tokenised = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenised)

    return index, bm25, chunks


def save_index(faiss_index, bm25, chunks, directory=INDEX_DIR):
    os.makedirs(directory, exist_ok=True)
    faiss.write_index(faiss_index, f"{directory}/faiss.index")
    with open(f"{directory}/bm25.pkl",   "wb") as f: pickle.dump(bm25,   f)
    with open(f"{directory}/chunks.pkl", "wb") as f: pickle.dump(chunks, f)


def load_index(directory=INDEX_DIR):
    faiss_index = faiss.read_index(f"{directory}/faiss.index")
    with open(f"{directory}/bm25.pkl",   "rb") as f: bm25   = pickle.load(f)
    with open(f"{directory}/chunks.pkl", "rb") as f: chunks = pickle.load(f)
    return faiss_index, bm25, chunks


def index_exists(directory=INDEX_DIR):
    return os.path.exists(f"{directory}/faiss.index")


def ingest_files(file_paths: list[str]) -> int:
    all_chunks = []
    for path in file_paths:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            pages = extract_pdf(path)
        elif ext in (".docx", ".doc"):
            pages = extract_docx(path)
        else:
            continue
        all_chunks.extend(semantic_chunk(pages))

    faiss_index, bm25, chunks = build_index(all_chunks)
    save_index(faiss_index, bm25, chunks)
    return len(chunks)