# backend.py
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
from quest import init_quest_memory

load_dotenv()

# Initialize LLM and embeddings
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def extract_npc_name(text):
    """Extracts the most prominent NPC name from a block of text."""
    prompt = ChatPromptTemplate.from_template(
        "Read the following text. Identify the single most important proper name of a character or creature being interacted with. "
        "Do not identify the player. If no specific character is mentioned, respond with 'None'.\n\n"
        "Text: {text}\n\n"
        "Character Name:"
    )
    chain = prompt | llm | StrOutputParser()
    name = chain.invoke({"text": text})
    if "none" in name.lower() or len(name.split()) > 3:
        return None
    return name.strip()

def initialize_memory(st):
    """Initialize all memory types in Streamlit's session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "You awaken to the smell of brine and damp wood in the dimly lit hold of a creaking ship. You have no memory of how you got here. A single, barred door stands before you. What do you do?"
        }]
        st.session_state.short_term_memory = []
        dm_vectorstore = Chroma(embedding_function=embedding_function, persist_directory="./chroma_db_dm")
        st.session_state.retriever = dm_vectorstore.as_retriever(search_kwargs={'k': 3})
        st.session_state.char_retriever = init_character_memory()
        st.session_state.quest_retriever = init_quest_memory()

def generate_narrative(user_input, short_term_history, retriever, char_retriever):
    """Generate the next part of the story, dynamically fetching NPC memory if relevant."""
    npc_name = extract_npc_name(user_input)
    character_memory = ""
    if npc_name:
        character_memory = get_character_memories(char_retriever, npc_name)

    narrative_prompt = ChatPromptTemplate.from_template("""
You are a creative Dungeon Master. Your world is persistent.
- Use 'Relevant Past Events' for long-term consistency.
- Use 'Recent History' for immediate context.
- If present, use 'NPC Memory' to inform that NPC's actions and dialogue.
- Weave an engaging narrative based on the player's action.
- Your response must be concise (1-2 sentences) and only be the story continuation. Do not break character.
- Always give me suggestions for what to do next.
- Continue the story no matter what the player says.

--- NPC Memory for "{npc_name}" ---
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
            npc_name=lambda x: npc_name or "N/A"
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

def summarize_npc_turn(npc_name, user_input, ai_response):
    """Summarize an NPC's development for this turn."""
    summary_prompt = ChatPromptTemplate.from_template(
        "In one concise sentence, summarize what this turn reveals about the NPC {npc_name}'s personality, knowledge, or relationship to the player. "
        "Player: {user_input}\nDM: {ai_response}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({
        "npc_name": npc_name,
        "user_input": user_input,
        "ai_response": ai_response
    })