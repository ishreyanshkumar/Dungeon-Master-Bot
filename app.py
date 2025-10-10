# app.py
import streamlit as st
from backend import initialize_memory, generate_narrative, summarize_turn, extract_npc_name, summarize_npc_turn
from character_memory import add_character_memory
from quest import add_quest, get_all_quests

# --- 1. INITIALIZE SESSION STATE ---
initialize_memory(st)

# --- 2. STREAMLIT UI ---
st.title("AI Dungeon Master ⚔️")

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. USER INPUT AND MAIN LOGIC ---
if user_input := st.chat_input("What is your next action? (Type 'q' to see quests)"):

    # Handle Quest Display
    if user_input.lower().strip() == 'q':
        all_quests = get_all_quests(st.session_state.quest_retriever)
        if all_quests:
            quest_list = "\n".join(f"- {q}" for q in all_quests)
            st.info(f"**Your Quests & Achievements:**\n{quest_list}")
        else:
            st.info("You have no active quests or major achievements yet.")
        st.stop()

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Generate Narrative ---
    with st.chat_message("assistant"):
        with st.spinner("The DM ponders your fate..."):
            short_term_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.short_term_memory])
            ai_response = generate_narrative(
                user_input=user_input,
                short_term_history=short_term_history,
                retriever=st.session_state.retriever,
                char_retriever=st.session_state.char_retriever
            )
            st.markdown(ai_response)

    # --- Update Memories ---
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.session_state.short_term_memory.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": ai_response},
    ])
    if len(st.session_state.short_term_memory) > 8:
        st.session_state.short_term_memory = st.session_state.short_term_memory[-8:]

    # Summarize turn for DM's long-term memory
    turn_summary = summarize_turn(user_input, ai_response)
    if turn_summary:
        st.session_state.retriever.vectorstore.add_texts([turn_summary])

    # Dynamic NPC Memory Update
    turn_text = f"Player: {user_input}\nDM: {ai_response}"
    npc_name = extract_npc_name(turn_text)
    if npc_name:
        # If an NPC is found, summarize the interaction and save it to their memory
        npc_summary = summarize_npc_turn(npc_name, user_input, ai_response)
        if npc_summary:
            add_character_memory(st.session_state.char_retriever, npc_name, npc_summary)

    # Check for and store new quests
    add_quest(st.session_state.quest_retriever, user_input, ai_response)

    st.rerun()