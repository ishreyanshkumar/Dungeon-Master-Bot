"""
quest.py — Quest and achievement tracking using ChromaDB.
"""
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import embs, llm


def init_quests(campaign: str = "default"):
    """Initialise the quest memory vector store for a given campaign."""
    q_db = Chroma(
        embedding_function=embs,
        persist_directory=f"./campaigns/{campaign}/quests"
    )
    return q_db.as_retriever(search_kwargs={"k": 5})


def add_quest(q_ret, user_text: str, dm_resp: str) -> str:
    """
    Ask the LLM to extract a quest/achievement from the turn.
    Stores it if found; returns the summary string (or empty string).
    """
    prompt = ChatPromptTemplate.from_template(
        "Summarize the main quest or achievement in this turn in 5 words or less. "
        "Examples: 'Defeated the goblin king', 'Found the hidden artifact', 'Escaped the prison'. "
        "If there is no clear quest or achievement, respond with exactly 'None'.\n\n"
        "Player: {user_input}\nDM: {ai_response}"
    )
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"user_input": user_text, "ai_response": dm_resp}).strip()

    if summary and summary.lower() != "none":
        q_ret.vectorstore.add_texts([summary])
        return summary
    return ""


def get_quests(q_ret) -> list[str]:
    """Return all stored quest/achievement strings for display."""
    try:
        docs = q_ret.vectorstore._collection.get()
        return docs["documents"] if docs and docs.get("documents") else []
    except Exception:
        return []
