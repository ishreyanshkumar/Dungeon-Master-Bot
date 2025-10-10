import streamlit as st
from backend import init_memory, gen_story, sum_turn, find_npc, sum_npc_turn
from character_memory import add_char_mem
from quest import add_quest, get_quests

# Initialize all memory systems
init_memory(st)
st.title("AI Dungeon Master")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process new user input
if user_text := st.chat_input("What is your next action? (Type 'q' to see quests)"):

    # Handle special 'q' command to show quests
    if user_text.lower().strip() == 'q':
        quests = get_quests(st.session_state.quest_retriever)
        if quests:
            q_list = "\n".join(f"- {q}" for q in quests)
            st.info(f"**Your Quests & Achievements:**\n{q_list}")
        else:
            st.info("You have no active quests or major achievements yet.")
        st.stop() 

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Get the AI's response
    with st.chat_message("assistant"):
        with st.spinner("The DM ponders your fate..."):
            hist = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.short_term_memory])

            # Call the backend
            resp = gen_story(
                user_text,
                hist,
                st.session_state.retriever,
                st.session_state.char_retriever
            )
            st.markdown(resp)

    # MEMORY UPDATES

    st.session_state.messages.append({"role": "assistant", "content": resp})
    
    # Update short-term memory
    st.session_state.short_term_memory.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": resp},
    ])
    
    # Keep short-term memory to the last 4 turns
    if len(st.session_state.short_term_memory) > 8:
        st.session_state.short_term_memory = st.session_state.short_term_memory[-8:]

    # Summarize turn for long-term memory
    turn_sum = sum_turn(user_text, resp)
    if turn_sum:
        st.session_state.retriever.vectorstore.add_texts([turn_sum])

    # Update character-specific memory if an NPC was found
    turn_text = f"Player: {user_text}\nDM: {resp}"
    npc_name = find_npc(turn_text)
    if npc_name:
        npc_sum = sum_npc_turn(npc_name, user_text, resp)
        if npc_sum:
            add_char_mem(st.session_state.char_retriever, npc_name, npc_sum)

    # Check for and store any new quests
    add_quest(st.session_state.quest_retriever, user_text, resp)
    
    # Rerun the app to update the UI
    st.rerun()