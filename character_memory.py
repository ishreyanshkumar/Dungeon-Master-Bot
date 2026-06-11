"""
character_memory.py — Per-NPC persistent memory using ChromaDB.
Uses metadata filtering to ensure each character's memories are isolated.
"""
from langchain_community.vectorstores import Chroma
from config import embs


def init_chars(campaign: str = "default"):
    """Initialise the character memory vector store for a given campaign."""
    char_db = Chroma(
        embedding_function=embs,
        persist_directory=f"./campaigns/{campaign}/characters"
    )
    return char_db.as_retriever(search_kwargs={"k": 3})


def add_char_mem(char_ret, name: str, memory: str):
    """Store a new memory snippet, tagged with the character's name."""
    char_ret.vectorstore.add_texts(
        [memory],
        metadatas=[{"character": name}]
    )


def get_char_mem(char_ret, name: str) -> str:
    """
    Retrieve memories for a specific character using a metadata WHERE filter
    so we never accidentally return another NPC's memories.
    """
    try:
        results = char_ret.vectorstore._collection.query(
            query_texts=[name],
            n_results=3,
            where={"character": name}
        )
        docs = results["documents"][0] if results.get("documents") else []
        return "\n".join(docs) if docs else "No significant memory of this character."
    except Exception:
        # Graceful fallback if the collection is empty
        return "No significant memory of this character."
