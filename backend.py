import os
from dotenv import load_dotenv
import chromadb
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from character_memory import init_character_memory, add_character_memory, get_character_memories  


load_dotenv()

# Initialize LLM and embeddings
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_memory(st):
    """Initialize session state for game and memory."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "You awaken to the smell of brine and damp wood in the dimly lit hold of a creaking ship. You have no memory of how you got here. A single, barred door stands before you. What do you do?"
        }]

        st.session_state.short_term_memory = []

        vectorstore = Chroma(
            embedding_function=embedding_function,
            persist_directory="./chroma_db_dm"
        )
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        st.session_state.char_retriever = init_character_memory()

def generate_narrative(user_input, short_term_history, retriever):
    """Generate the next part of the story using the LLM."""
    narrative_prompt = ChatPromptTemplate.from_template("""
You are a creative Dungeon Master. Your world is persistent.
- Use 'Relevant Past Events' for long-term consistency.
- Use 'Recent History' for immediate context.
- Use 'Character Memory' to inform NPC interactions and story progress.
- Weave an engaging narrative based on the player's action.
- Your response must be concise (1-2 sentences) and only be the story continuation. Do not break character.
- Always give me suggestions for what to do next.
- Continue the story no matter what the player says.

--- Character Memory ({character_name}) ---
{character_memory}                                                        

--- Relevant Past Events (Long-Term Memory) ---
{relevant_docs}

--- Recent History (Last 4 Turns) ---
{short_term_history}

--- Player's Action ---
{user_input}
""")

    narrative_chain = (
        {"relevant_docs": retriever, "user_input": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            short_term_history=lambda x: short_term_history,
            character_memory=lambda x: character_memory,
            character_name=lambda x: character_name or "Unknown"
        )        
        | narrative_prompt
        | llm
        | StrOutputParser()
    )

    return narrative_chain.invoke(user_input)


def summarize_turn(user_input, ai_response):
    """Create a one-line summary of this turn for long-term memory."""
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the key event from this turn in one concise sentence. Player: {user_input}\nDM: {ai_response}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({"user_input": user_input, "ai_response": ai_response})

def summarize_character_turn(character_name, user_input, ai_response):
    """summarize character development for this turn."""
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize how {character_name} acted or changed this turn. "
        "Player: {user_input}\nDM: {ai_response}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({
        "character_name": character_name,
        "user_input": user_input,
        "ai_response": ai_response
    })