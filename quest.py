from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

embs = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)

def init_quests():
    q_db = Chroma(
        embedding_function=embs,
        persist_directory="./chroma_db_quests"
    )
    return q_db.as_retriever(search_kwargs={'k': 5})

# Identifies and stores a new quest
def add_quest(q_ret, text, resp):

    prompt = ChatPromptTemplate.from_template(
        "Summarize the main quest or achievement in this turn in 5 words or less. "
        "Examples: 'Defeated the goblin king', 'Found the hidden artifact', 'Escaped the prison'. "
        "If there is no quest, respond with 'None'.\n\n"
        "Player: {user_input}\nDM: {ai_response}"
    )
    
    # Chain to extract a quest summary from the turn.
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"user_input": text, "ai_response": resp})

    # If a quest is found, add it to the vector store.
    if summary and "none" not in summary.lower():
        q_ret.vectorstore.add_texts([summary])
    return summary

# Retrieves all stored quests.
def get_quests(q_ret):
    docs = q_ret.vectorstore._collection.get()
    return [doc for doc in docs["documents"]] if docs and docs["documents"] else []