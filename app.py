# app.py
import streamlit as st
from backend import initialize_memory, generate_narrative, summarize_turn

# --- 1. INITIALIZE SESSION STATE ---
initialize_memory(st)

# --- 2. STREAMLIT UI ---
st.title("AI Dungeon Master")

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. USER INPUT AND MAIN LOGIC ---
if user_input := st.chat_input("What is your next action?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Gather short-term memory context
    short_term_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.short_term_memory])

    # Generate narrative
    with st.chat_message("assistant"):
        with st.spinner("The DM ponders your fate..."):
            ai_response = generate_narrative(user_input, short_term_history, st.session_state.retriever)
            st.markdown(ai_response)

    # Update chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # Update short-term memory (keep 8 entries max)
    st.session_state.short_term_memory.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": ai_response},
    ])
    if len(st.session_state.short_term_memory) > 8:
        st.session_state.short_term_memory = st.session_state.short_term_memory[-8:]

    # Summarize this turn and add to vector store
    turn_summary = summarize_turn(user_input, ai_response)
    st.session_state.retriever.vectorstore.add_texts([turn_summary])

    # Rerun app to show updated chat
    st.rerun()
