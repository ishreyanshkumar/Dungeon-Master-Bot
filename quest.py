# quest.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)

def init_quest_memory():
    """Initializes the vector store for quests and achievements."""
    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory="./chroma_db_quests"
    )
    return vectorstore.as_retriever(search_kwargs={'k': 5})

def add_quest(retriever, user_input, ai_response):
    """Checks for a new quest or achievement and adds it to memory."""
    quest_prompt = ChatPromptTemplate.from_template(
        "Summarize the main quest or achievement in this turn in 5 words or less. "
        "Examples: 'Defeated the goblin king', 'Found the hidden artifact', 'Escaped the prison'. "
        "If there is no quest, respond with 'None'.\n\n"
        "Player: {user_input}\nDM: {ai_response}"
    )
    chain = quest_prompt | llm | StrOutputParser()
    quest_summary = chain.invoke({"user_input": user_input, "ai_response": ai_response})

    if quest_summary and "none" not in quest_summary.lower():
        retriever.vectorstore.add_texts([quest_summary])
    return quest_summary

def get_all_quests(retriever):
    """Retrieves all stored quests and achievements."""
    docs = retriever.vectorstore._collection.get()
    return [d for d in docs["documents"]] if docs and docs["documents"] else []