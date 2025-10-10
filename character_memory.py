from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embs = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initializes the vector DB for character memories
def init_chars():
    char_db = Chroma(
        embedding_function=embs,
        persist_directory="./chroma_db_characters"
    )
    return char_db.as_retriever(search_kwargs={'k': 3})

# Stores a new memory for a character
def add_char_mem(char_ret, name, mem):
    char_ret.vectorstore.add_texts(
        [mem],
        metadatas=[{"character": name}]
    )

# Retrieves relevant memories for a character
def get_char_mem(char_ret, name):
    docs = char_ret.get_relevant_documents(name)
    if not docs:
        return "No significant memory of this character."
    return "\n".join([doc.page_content for doc in docs])