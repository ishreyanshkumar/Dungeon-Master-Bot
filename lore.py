"""
lore.py — World Lore Journal.
Automatically extracts bestiary entries, place descriptions, and lore facts
from DM responses and stores them in a persistent ChromaDB collection.
"""
import json
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm, embs

_LORE_PROMPT = ChatPromptTemplate.from_template(
    "Analyse the DM's response. Extract any NEW lore-worthy information worth recording in a world encyclopedia.\n"
    "Return ONLY valid JSON with these keys (null if nothing new):\n"
    "  entry_type: one of 'creature', 'place', 'faction', 'item', 'event', or null\n"
    "  name: the proper name of the subject (string), or null\n"
    "  description: 1-2 sentence encyclopedia-style description, or null\n\n"
    "Only extract something if it's specific, named, and meaningfully described.\n"
    "DM Response: {dm_response}\n\nJSON only:"
)
_lore_chain = _LORE_PROMPT | llm | StrOutputParser()

LORE_ICONS = {
    "creature": "🐉",
    "place":    "🏰",
    "faction":  "⚜️",
    "item":     "🗡️",
    "event":    "📜",
}


def init_lore(campaign: str = "default"):
    """Initialise the lore vector store for a campaign."""
    db = Chroma(
        embedding_function=embs,
        persist_directory=f"./campaigns/{campaign}/lore"
    )
    return db


def extract_and_store_lore(lore_db, dm_response: str) -> dict | None:
    """
    Extract a lore entry from the DM response and store it.
    Returns the entry dict if something was found, else None.
    """
    try:
        raw = _lore_chain.invoke({"dm_response": dm_response}).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        if not data.get("name") or not data.get("description") or not data.get("entry_type"):
            return None

        entry_type = data["entry_type"]
        name = data["name"].strip()
        desc = data["description"].strip()

        # Dedup: check if this name is already in lore
        existing = lore_db._collection.get(where={"name": name})
        if existing and existing.get("documents"):
            return None  # Already recorded

        lore_db.add_texts(
            [f"{name}: {desc}"],
            metadatas=[{"name": name, "type": entry_type}]
        )
        return {"type": entry_type, "name": name, "description": desc}

    except Exception:
        return None


def get_all_lore(lore_db) -> list[dict]:
    """Return all lore entries sorted by type."""
    try:
        raw = lore_db._collection.get()
        if not raw or not raw.get("documents"):
            return []
        entries = []
        for doc, meta in zip(raw["documents"], raw["metadatas"]):
            name = meta.get("name", "Unknown")
            entry_type = meta.get("type", "event")
            # doc is "Name: description"
            desc = doc.split(": ", 1)[1] if ": " in doc else doc
            entries.append({"type": entry_type, "name": name, "description": desc})
        # Sort by type then name
        entries.sort(key=lambda e: (e["type"], e["name"]))
        return entries
    except Exception:
        return []
