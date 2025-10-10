# character_memory.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def init_character_memory():
    """Initialize a Chroma DB for NPC-specific memories."""
    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory="./chroma_db_characters"
    )
    return vectorstore.as_retriever(search_kwargs={'k': 3})

def add_character_memory(retriever, npc_name, memory_text):
    """Store memory linked to an NPC."""
    retriever.vectorstore.add_texts(
        [memory_text],
        metadatas=[{"character": npc_name}]
    )

def get_character_memories(retriever, npc_name):
    """Retrieve memories relevant to an NPC."""
    docs = retriever.get_relevant_documents(npc_name)
    if not docs:
        return "No significant memory of this character."
    return "\n".join([d.page_content for d in docs])