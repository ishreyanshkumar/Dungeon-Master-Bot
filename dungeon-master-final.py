import streamlit as st
import chromadb
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from collections import deque
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# --- 1. CORE APPLICATION SETUP ---
# Initialize the LLM for high-speed generation
# This fulfills the requirement to use an external free API like Groq [cite: 25]
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Initialize the embedding model for turning text into vectors
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 2. SESSION STATE & MEMORY INITIALIZATION ---
# Use Streamlit's session_state to hold memory across reruns
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    
    # Short-Term "Working" Memory: Stores the last 5 turns 
    st.session_state.short_term_memory = deque(maxlen=5)

    # Long-Term "Persistent" Memory: A vector store for RAG [cite: 21, 23]
    # This uses a local ChromaDB instance that persists in a directory
    vectorstore = Chroma(
        embedding_function=embedding_function, 
        persist_directory="./chroma_db_dm"
    )
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # Bonus: Dynamic Quest Log 
    st.session_state.quest_log = {"Main Quest: Find the Sunstone": "Not Started"}

    # Bonus: Character Memory 
    st.session_state.character_memory = {}

    # Chat history for displaying in the UI
    st.session_state.messages = [{"role": "assistant", "content": "Your adventure begins! What do you do?"}]

# --- 3. UI CONFIGURATION ---
st.title("AI Dungeon Master 🧠🎲")
st.caption("A narrative AI with persistent memory.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. CORE LOGIC ---
if user_input := st.chat_input("What is your next action?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # A. Format all memory components into strings for the prompt
    short_term_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.short_term_memory])
    quest_log_str = json.dumps(st.session_state.quest_log, indent=2)
    character_memory_str = json.dumps(st.session_state.character_memory, indent=2)

    # B. Define the prompt template for the main narrative generation
    # This template assembles all memory types to provide context to the LLM
    main_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a master Dungeon Master. Your world is persistent and your NPCs have memory.
        - Use the 'Relevant Past Events' to maintain long-term consistency.
        - Use the 'Recent History' for immediate context.
        - Use the 'Quest Log' and 'Character Memory' to inform NPC interactions and story progress.
        - Weave a creative, engaging, and coherent narrative and dont repeat my words.
        - Your response should ONLY be the story continuation. Do not break character.
        - Keep responses concise, around 50 words.
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

    # C. Create the RAG chain for narrative generation
    rag_chain = (
        {
            "relevant_docs": st.session_state.retriever, 
            "user_input": RunnablePassthrough(),
            "short_term_history": lambda x: short_term_history,
            "quest_log": lambda x: quest_log_str,
            "character_memory": lambda x: character_memory_str
        }
        | main_prompt_template
        | llm
        | StrOutputParser()
    )

    # D. Invoke the chain to get the AI's response
    with st.chat_message("assistant"):
        with st.spinner("The DM is thinking..."):
            ai_response = rag_chain.invoke(user_input)
            st.markdown(ai_response)

    # E. Update memories after the turn is complete
    # 1. Add user and AI messages to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    # 2. Update short-term memory
    st.session_state.short_term_memory.append({"role": "user", "content": user_input})
    st.session_state.short_term_memory.append({"role": "assistant", "content": ai_response})
    
    # 3. Create a summary of the turn for long-term memory
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a summarization expert. Create a concise, one-sentence summary of the key events from the following game turn. Example: 'The player entered the tavern and spoke to the mysterious cloaked figure.'"),
        ("user", f"Player: {user_input}\nDM: {ai_response}")
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()
    turn_summary = summary_chain.invoke({})
    
    # 4. Add the summary to the long-term vector store
    st.session_state.retriever.vectorstore.add_texts([turn_summary])

    # 5. Update the Quest Log and Character Memory dynamically
    update_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a state management AI. Based on the turn summary, update the JSON for the Quest Log and Character Memory.
        - If an NPC is mentioned, add or update their entry in Character Memory with a key fact from the interaction.
        - If a quest objective is met, update its status in the Quest Log.
        - Output ONLY the two updated JSON objects, separated by '---'.
        Quest Log JSON:
        {quest_log}
        
        Character Memory JSON:
        {character_memory}
        """),
        ("user", f"Turn Summary: {turn_summary}")
    ])

    update_chain = (
        {
            "quest_log": lambda _: quest_log_str,
            "character_memory": lambda _: character_memory_str
        }
        | update_prompt
        | llm
        | StrOutputParser()
    )
    updated_state_str = update_chain.invoke({})

    
    try:
        quest_str, char_str = updated_state_str.split('---')
        st.session_state.quest_log = json.loads(quest_str.strip())
        st.session_state.character_memory = json.loads(char_str.strip())
        # The line below is for debugging; you can uncomment to see the memory being updated
        st.sidebar.write("Updated State:", st.session_state.quest_log, st.session_state.character_memory)
    except Exception as e:
        # If the LLM fails to return valid JSON, we just skip the state update for this turn
        print(f"Could not update state: {e}")