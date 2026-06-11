"""
memory_consolidator.py — Automatic memory consolidation using clustering.

The problem:
  After many turns, the long-term memory DB fills with very similar summaries.
  Example after a long tavern scene:
    "Player spoke to the innkeeper about the missing shipment."
    "Player asked the innkeeper whether he'd seen strangers."
    "Player questioned the innkeeper about the back room."
  These 3 occupy 3 retrieval slots but only convey 1 meaningful context chunk.
  The LLM is wasting its context window on near-duplicate information.

Solution — periodic KMeans consolidation:
  1. Embed all memories in the DB.
  2. Run KMeans with k = max(1, n // CLUSTER_RATIO).
  3. For each cluster, ask the LLM to merge all memories in that cluster
     into ONE canonical summary sentence.
  4. Delete the originals, store the merged summary.
  5. Net effect: a DB of 30 memories with ~6 distinct themes becomes 6 sharp,
     unique, information-dense summaries.

This runs automatically when the memory DB exceeds CONSOLIDATION_THRESHOLD.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm, embs

# ── Tuning constants ────────────────────────────────────────────────────────
CONSOLIDATION_THRESHOLD = 20   # Start consolidating after this many memories
CLUSTER_RATIO = 4              # Merge every ~4 memories into 1
MIN_CLUSTER_SIZE = 2           # Skip clusters with fewer docs (already unique)

# ── Merge prompt ────────────────────────────────────────────────────────────
_MERGE_PROMPT = ChatPromptTemplate.from_template(
    "You are maintaining a Dungeon Master's world history log. "
    "Merge the following related events into ONE concise summary sentence "
    "that preserves all important names, locations, and outcomes. "
    "Be specific and factual. Do not add new information.\n\n"
    "Events:\n{events}\n\nMerged summary:"
)
_merge_chain = _MERGE_PROMPT | llm | StrOutputParser()


def _get_all_docs(collection) -> tuple[list[str], list[str]]:
    """Return (ids, documents) from a ChromaDB collection."""
    raw = collection.get()
    ids  = raw.get("ids", [])
    docs = raw.get("documents", [])
    return ids, docs


def should_consolidate(vectorstore) -> bool:
    """Return True if the memory DB is large enough to consolidate."""
    try:
        ids, _ = _get_all_docs(vectorstore._collection)
        return len(ids) >= CONSOLIDATION_THRESHOLD
    except Exception:
        return False


def consolidate_memories(vectorstore, verbose: bool = False) -> int:
    """
    Run one consolidation pass on a ChromaDB vectorstore.

    Returns the number of memories reduced (original_count - new_count).
    Returns 0 if consolidation was skipped or failed.
    """
    try:
        ids, docs = _get_all_docs(vectorstore._collection)
        n = len(ids)
        if n < CONSOLIDATION_THRESHOLD:
            return 0

        # ── Embed all memories ──
        raw_embs = np.array(embs.embed_documents(docs))
        normed   = normalize(raw_embs)          # L2-norm for cosine-friendly KMeans

        # ── Cluster ──
        k = max(1, n // CLUSTER_RATIO)
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(normed)

        # ── Build cluster → doc groups ──
        clusters: dict[int, list[tuple[str, str]]] = {}  # label → [(id, doc)]
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append((ids[i], docs[i]))

        ids_to_delete: list[str] = []
        merged_texts:  list[str] = []

        for label, members in clusters.items():
            if len(members) < MIN_CLUSTER_SIZE:
                continue   # Already unique, leave it alone

            # Merge the cluster with LLM
            events_block = "\n".join(f"- {doc}" for _, doc in members)
            try:
                merged = _merge_chain.invoke({"events": events_block}).strip()
            except Exception:
                # If merge fails, keep the first doc as representative
                merged = members[0][1]

            if verbose:
                print(f"[Consolidator] Cluster {label}: {len(members)} → 1")
                print(f"  Merged: {merged[:80]}…")

            # Mark originals for deletion
            ids_to_delete.extend(mid for mid, _ in members)
            merged_texts.append(merged)

        if not ids_to_delete:
            return 0

        # ── Delete originals ──
        vectorstore._collection.delete(ids=ids_to_delete)

        # ── Store merged summaries ──
        if merged_texts:
            vectorstore.add_texts(merged_texts)

        reduction = len(ids_to_delete) - len(merged_texts)
        return max(0, reduction)

    except Exception as e:
        if verbose:
            print(f"[Consolidator] Error: {e}")
        return 0


def get_memory_count(vectorstore) -> int:
    """Return the current number of memories in a vectorstore."""
    try:
        ids, _ = _get_all_docs(vectorstore._collection)
        return len(ids)
    except Exception:
        return 0
