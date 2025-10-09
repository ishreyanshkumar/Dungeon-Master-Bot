import re
import json

# -------------------------------
# 1️⃣ Helper Function: Summarize Turn
# -------------------------------
def summarize_turn(user_action, dm_response, llm_model):
    """
    Create a concise summary of the turn.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a summarization expert. Condense the following turn into one clear sentence."),
        ("user", f"Player action: {user_action}\nDM response: {dm_response}")
    ])
    summary_chain = prompt | llm_model | StrOutputParser()
    summary = summary_chain.invoke({}).strip()
    
    # Ensure it ends with a period
    if not summary.endswith('.'):
        summary += '.'
    return summary

# -------------------------------
# 2️⃣ Helper Function: Parse JSON Safely
# -------------------------------
def parse_json_blocks(llm_output):
    """
    Extract two JSON objects safely from LLM output.
    Returns (quest_log, character_memory) or (None, None) if parsing fails.
    """
    try:
        json_matches = re.findall(r"\{.*?\}", llm_output, re.DOTALL)
        if len(json_matches) >= 2:
            quest_log = json.loads(json_matches[0])
            character_memory = json.loads(json_matches[1])
            return quest_log, character_memory
        else:
            return None, None
    except Exception as error:
        print(f"[Error] Could not parse JSON: {error}")
        return None, None

# -------------------------------
# 3️⃣ Main Function: Update Game State
# -------------------------------
def update_game_state(user_action, dm_response, llm_model, session_state):
    """
    Update short-term memory, long-term memory, quest log, and character memory.
    """
    # 1. Add messages to short-term memory
    session_state.short_term_memory.append({"role": "user", "content": user_action})
    session_state.short_term_memory.append({"role": "assistant", "content": dm_response})

    # 2. Generate a concise summary for long-term memory
    turn_summary = summarize_turn(user_action, dm_response, llm_model)
    session_state.retriever.vectorstore.add_texts([turn_summary])

    # 3. Convert current state to JSON strings
    quest_log_json = json.dumps(session_state.quest_log, indent=2)
    character_memory_json = json.dumps(session_state.character_memory, indent=2)

    # 4. Prompt LLM to update state
    update_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a game assistant. Update the Quest Log and Character Memory based on the turn summary.
Return only two JSON objects separated by '---'.
"""),
        ("user", f"Turn Summary: {turn_summary}\nQuest Log: {quest_log_json}\nCharacter Memory: {character_memory_json}")
    ])

    update_chain = (
        {
            "quest_log": lambda _: quest_log_json,
            "character_memory": lambda _: character_memory_json
        }
        | update_prompt
        | llm_model
        | StrOutputParser()
    )

    llm_result = update_chain.invoke({})

    # 5. Parse JSON safely
    quest_log, character_memory = parse_json_blocks(llm_result)
    if quest_log and character_memory:
        session_state.quest_log = quest_log
        session_state.character_memory = character_memory
    else:
        print("[Warning] Memory update skipped due to invalid JSON output.")
