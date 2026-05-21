import numpy as np
from .ingestor import get_embedder

TOP_K = 3
RRF_K = 60


def reciprocal_rank_fusion(faiss_ranks: list[int],
                            bm25_ranks:  list[int],
                            n_chunks:    int) -> list[int]:
    scores = np.zeros(n_chunks)
    for rank, idx in enumerate(faiss_ranks, start=1):
        scores[idx] += 1 / (RRF_K + rank)
    for rank, idx in enumerate(bm25_ranks, start=1):
        scores[idx] += 1 / (RRF_K + rank)
    return np.argsort(-scores)[:TOP_K].tolist()


def retrieve(query: str,
             faiss_index,
             bm25,
             chunks: list[dict]) -> list[dict]:
    embedder = get_embedder()
    n        = len(chunks)

    q_emb = embedder.encode(
        [query], normalize_embeddings=True
    ).astype("float32")
    _, faiss_top = faiss_index.search(q_emb, min(TOP_K * 3, n))
    faiss_ranks  = faiss_top[0].tolist()

    tokenised   = query.lower().split()
    bm25_scores = bm25.get_scores(tokenised)
    bm25_ranks  = np.argsort(-bm25_scores)[:TOP_K * 3].tolist()

    merged = reciprocal_rank_fusion(faiss_ranks, bm25_ranks, n)
    return [chunks[i] for i in merged]