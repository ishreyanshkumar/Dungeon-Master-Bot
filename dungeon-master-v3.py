import streamlit as st
import chromadb
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from collections import deque
import json
import os
from dotenv import load_dotenv

# --- 1. SETUP AND CONFIGURATION ---

# Load environment variables from .env file for the Groq API key
load_dotenv()

# Initialize the core components
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- 2. INITIALIZE SESSION STATE (IF IT DOESN'T EXIST) ---
# We use st.session_state to store all game data so it persists between actions.
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.turn_count = 0

    # Short-Term "Working" Memory: Stores the last 5 user/AI message pairs
    st.session_state.short_term_memory = deque(maxlen=10)

    # Long-Term "Persistent" Memory: Chroma vector store for RAG
    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory="./chroma_db_dm"
    )
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # Bonus Feature: Dynamic Quest Log
    st.session_state.quest_log = {"Main Quest: Uncover your past": "Find a way out of the ship's hold."}

    # Bonus Feature: Character Memory
    st.session_state.character_memory = {}
    
    # Chat history for UI display with an engaging start
    st.session_state.messages = [{"role": "assistant", "content": "You awaken to the smell of brine and damp wood. You're in the dimly lit hold of a creaking ship with no memory of how you arrived. A single, barred door stands before you. Next to it, a rusty key hangs on a hook. What do you do?"}]


# --- 3. STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("AI Dungeon Master 🧠🎲")
st.caption("A narrative AI with persistent memory, built for the Inter IIT Tech Meet.")

# Sidebar for displaying persistent game state
with st.sidebar:
    st.header("Game State")
    st.metric("Turn Count", st.session_state.get('turn_count', 0))
    
    with st.expander("**Quest Log**", expanded=True):
        st.json(st.session_state.get('quest_log', {}))
    
    with st.expander("**Character Memory**", expanded=True):
        if not st.session_state.get('character_memory', {}):
            st.info("You haven't met anyone memorable yet.")
        else:
            st.json(st.session_state.get('character_memory', {}))

# Display the chat history
for message in st.session_state.get('messages', []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. MAIN GAME LOGIC (RUNS WHEN USER ENTERS INPUT) ---

if user_input := st.chat_input("What is your next action?"):
    st.session_state.turn_count += 1
    
    # Add user's message to the UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # A. GATHER ALL CONTEXT FOR THE AI
    short_term_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.short_term_memory])
    quest_log_str = json.dumps(st.session_state.quest_log, indent=2)
    character_memory_str = json.dumps(st.session_state.character_memory, indent=2)

    # B. GENERATE THE NARRATIVE (LLM Call 1)
    narrative_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a master Dungeon Master. Your world is persistent.
- Use 'Relevant Past Events' for long-term consistency.
- Use 'Recent History' for immediate context.
- Use the 'Quest Log' and 'Character Memory' to inform the story.
- Weave a creative, engaging, and coherent narrative.
- **Keep your response concise (2-4 sentences).**
- Your response should ONLY be the story continuation. Do not break character.
- Remember to give me options like what I can do next. Do not repeat my words.
"""),
        ("user", """
--- Relevant Past Events (Long-Term Memory) ---
{relevant_docs}

--- Quest Log ---
{quest_log}

--- Character Memory ---
{character_memory}

--- Recent History (Last 5 Turns) ---
{short_term_history}

--- Player's Action ---
{user_input}
""")
    ])
    
    narrative_chain = (
        {
            "relevant_docs": st.session_state.retriever, 
            "user_input": RunnablePassthrough(),
            "short_term_history": lambda x: short_term_history,
            "quest_log": lambda x: quest_log_str,
            "character_memory": lambda x: character_memory_str
        }
        | narrative_prompt
        | llm
        | StrOutputParser()
    )

    with st.chat_message("assistant"):
        with st.spinner("The DM ponders your fate..."):
            ai_response = narrative_chain.invoke(user_input)
            st.markdown(ai_response)

    # C. UPDATE MEMORY AND STATE
    # 1. Update chat history and short-term memory
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.session_state.short_term_memory.append({"role": "user", "content": user_input})
    st.session_state.short_term_memory.append({"role": "assistant", "content": ai_response})

    # 2. Create and store a summary for long-term memory (LLM Call 2)
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the key events of this turn in one sentence."),
        ("user", f"Player: {user_input}\nDM: {ai_response}")
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()
    turn_summary = summary_chain.invoke({})
    st.session_state.retriever.vectorstore.add_texts([turn_summary])
    
    # 3. Update Quest Log and Character Memory (LLM Call 3)
    update_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a state management AI. Based on the turn summary, update the JSON for the Quest Log and Character Memory.
- If an NPC is mentioned, add or update their entry.
- If a quest objective is met, update its status.
- Output ONLY the two updated JSON objects, separated by '---'.

Current Quest Log JSON:
{quest_log}

Current Character Memory JSON:
{character_memory}"""),
        ("user", f"Turn Summary: {turn_summary}")
    ])
    update_chain = update_prompt | llm | StrOutputParser()
    updated_state_str = update_chain.invoke({
        "quest_log": quest_log_str,
        "character_memory": character_memory_str
    })
    
    try:
        # This part is less robust but simple: we split the text and parse JSON
        quest_str, char_str = updated_state_str.split('---')
        st.session_state.quest_log = json.loads(quest_str.strip())
        st.session_state.character_memory = json.loads(char_str.strip())
    except Exception as e:
        # If the AI gives a bad format, we just skip the state update for this turn
        print(f"Could not update game state: {e}")

