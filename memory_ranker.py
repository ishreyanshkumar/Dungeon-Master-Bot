"""
memory_ranker.py — ML-based re-ranking of retrieved memory documents.

The problem with pure vector similarity (ChromaDB default):
  Embedding similarity finds documents SEMANTICALLY CLOSE to the query,
  but "semantically close" doesn't always mean "most narratively useful".
  If the player typed "I attack the guard", the top-k retrieved memories
  might all be variations of "player attacked a guard" (high cosine sim),
  giving the DM redundant context instead of diverse, useful history.

Solution — two-stage retrieval:
  Stage 1: ChromaDB fetches a wide candidate pool (k=10).
  Stage 2: This module re-ranks those candidates using a hybrid score:
    - TF-IDF keyword overlap  (lexical relevance, catches exact names/places)
    - Embedding cosine sim    (already computed by Chroma, reused from metadata)
    - Recency bonus           (more recent memories get a small boost)
    - Diversity penalty       (MMR — penalise docs too similar to already-chosen ones)
  Final top-k is returned for the LLM prompt.
"""

import math
import re
from collections import Counter
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── TF-IDF helpers ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer, strips punctuation."""
    return re.findall(r"[a-z']+", text.lower())


def tfidf_scores(query: str, docs: List[str]) -> np.ndarray:
    """
    Return a (len(docs),) array of TF-IDF cosine similarities
    between the query and each document.
    Falls back to zeros if sklearn fails (empty corpus etc.).
    """
    if not docs:
        return np.array([])
    try:
        corpus = [query] + docs
        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf = vec.fit_transform(corpus)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        return sims
    except Exception:
        return np.zeros(len(docs))


# ── Embedding cosine (numpy) ───────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── MMR (Maximal Marginal Relevance) ──────────────────────────────────────

def mmr_rerank(
    query_emb: np.ndarray,
    doc_embs: List[np.ndarray],
    docs: List[str],
    relevance_scores: np.ndarray,
    top_k: int = 3,
    lambda_mmr: float = 0.6,
) -> List[int]:
    """
    MMR selection: iteratively pick the document that maximises:
        λ · relevance(doc, query) − (1−λ) · max_similarity(doc, already_selected)

    lambda_mmr=1.0  → pure relevance (same as top-k by score)
    lambda_mmr=0.0  → pure diversity
    lambda_mmr=0.6  → balanced (default)

    Returns a list of selected indices into docs/doc_embs.
    """
    if not docs:
        return []

    top_k = min(top_k, len(docs))
    selected: List[int] = []
    remaining = list(range(len(docs)))

    for _ in range(top_k):
        best_idx = None
        best_score = -float("inf")

        for idx in remaining:
            rel = relevance_scores[idx]

            # Max similarity to already-selected docs
            if selected:
                max_sim = max(
                    _cosine(doc_embs[idx], doc_embs[s]) for s in selected
                )
            else:
                max_sim = 0.0

            mmr = lambda_mmr * rel - (1 - lambda_mmr) * max_sim

            if mmr > best_score:
                best_score = mmr
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# ── Main reranker ──────────────────────────────────────────────────────────

def rerank_memories(
    query: str,
    docs: List[str],
    embs_model,           # HuggingFaceEmbeddings instance from config
    top_k: int = 3,
    recency_boost: bool = True,
    lambda_mmr: float = 0.6,
) -> List[str]:
    """
    Full re-ranking pipeline:
      1. TF-IDF keyword score
      2. Dense embedding cosine score  
      3. Recency bonus (earlier in list = older)
      4. MMR diversity selection

    Args:
        query:         The player's current action text.
        docs:          Candidate documents from ChromaDB (wide k, e.g. k=10).
        embs_model:    The shared HuggingFaceEmbeddings singleton.
        top_k:         How many to return for the LLM prompt.
        recency_boost: Give newer memories (later indices) a small bonus.
        lambda_mmr:    MMR trade-off. 0.6 = balanced relevance/diversity.

    Returns:
        List of top_k re-ranked document strings.
    """
    if not docs:
        return []
    if len(docs) <= top_k:
        return docs

    # ── Stage 1: TF-IDF keyword scores ──
    tfidf = tfidf_scores(query, docs)

    # ── Stage 2: Dense embedding scores ──
    try:
        all_texts = [query] + docs
        all_embs = np.array(embs_model.embed_documents(all_texts))
        query_emb = all_embs[0]
        doc_embs  = list(all_embs[1:])
        dense = np.array([_cosine(query_emb, d) for d in doc_embs])
    except Exception:
        query_emb = np.zeros(384)
        doc_embs  = [np.zeros(384)] * len(docs)
        dense = np.zeros(len(docs))

    # ── Stage 3: Recency bonus ──
    n = len(docs)
    recency = np.linspace(0.0, 0.15, n) if recency_boost else np.zeros(n)

    # ── Combine into relevance score ──
    # Weighted sum: TF-IDF is valuable for exact name/place matches,
    # dense catches paraphrase/semantic matches. Recency is a small nudge.
    relevance = 0.35 * tfidf + 0.50 * dense + 0.15 * recency

    # ── Stage 4: MMR diverse selection ──
    selected_indices = mmr_rerank(
        query_emb=query_emb,
        doc_embs=doc_embs,
        docs=docs,
        relevance_scores=relevance,
        top_k=top_k,
        lambda_mmr=lambda_mmr,
    )

    return [docs[i] for i in selected_indices]
