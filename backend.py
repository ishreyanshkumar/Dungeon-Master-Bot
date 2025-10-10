import os
from dotenv import load_dotenv
import chromadb
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from character_memory import init_chars, get_char_mem
from quest import init_quests

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)
embs = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extracts an NPC from text using an LLM.
def find_npc(text):

    prompt = ChatPromptTemplate.from_template(
        "Read the following text. Identify the single most important proper name of a character or creature being interacted with. "
        "Do not identify the player. If no specific character is mentioned, respond with 'None'.\n\n"
        "Text: {text}\n\n"
        "Character Name:"
    )
    chain = prompt | llm | StrOutputParser()
    name = chain.invoke({"text": text})

    # Filter out invalid names.
    if "none" in name.lower() or len(name.split()) > 3:
        return None
    return name.strip()

# Initializes all memory
def init_memory(sess):

    # Run only once per session.
    if "messages" not in sess.session_state:
        sess.session_state.messages = [{
            "role": "assistant",
            "content": "You awaken to the smell of brine and damp wood in the dimly lit hold of a creaking ship. You have no memory of how you got here. A single, barred door stands before you. What do you do?"
        }]
        # Initialize short-term memory window.
        sess.session_state.short_term_memory = []
        # Initialize long-term, character, and quest memories.
        dm_db = Chroma(embedding_function=embs, persist_directory="./chroma_db_dm")
        sess.session_state.retriever = dm_db.as_retriever(search_kwargs={'k': 3})
        sess.session_state.char_retriever = init_chars()
        sess.session_state.quest_retriever = init_quests()

# Generates story.
def gen_story(text, hist, ret, char_ret):
    # Fetch NPC-specific memories if an NPC is there
    npc = find_npc(text)
    char_mem = ""
    if npc:
        char_mem = get_char_mem(char_ret, npc)

    # Main prompt for the Dungeon Master AI.
    prompt = ChatPromptTemplate.from_template("""
You are a creative Dungeon Master. Your world is persistent.
- Use 'Relevant Past Events' for long-term consistency.
- Use 'Recent History' for immediate context.
- If present, use 'NPC Memory' to build the NPC's character and its dialogs & actions along with the storyline. 
- Weave an engaging narrative based on the player's action.
- Your response must be concise (1-2 sentences) and only be the story continuation. Do not break character.
- Always give me suggestions for what to do next.
- Continue the story no matter what the player says.
- Dont display anything internal else.

--- NPC Memory for "{npc_name}" ---
{character_memory}

--- Relevant Past Events (Long-Term Memory) ---
{relevant_docs}

--- Recent History (Last 4 Turns) ---
{short_term_history}

--- Player's Action ---
{user_input}
""")

    # LCEL chain to assemble context and generate the story
    chain = (
        {"relevant_docs": ret, "user_input": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            short_term_history=lambda x: hist,
            character_memory=lambda x: char_mem,
            npc_name=lambda x: npc or "N/A"
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(text)

# Creates a one-sentence summary of a turn for long-term memory.
def sum_turn(text, resp):
    prompt = ChatPromptTemplate.from_template(
        "Summarize the key event from this turn in one concise sentence. Player: {user_input}\nDM: {ai_response}"
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"user_input": text, "ai_response": resp})

# Summarizes what a turn reveals about an NPC
def sum_npc_turn(name, text, resp):
    prompt = ChatPromptTemplate.from_template(
        "In one concise sentence, summarize what this turn reveals about the NPC {npc_name}'s personality, knowledge, or relationship to the player. "
        "Player: {user_input}\nDM: {ai_response}"
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "npc_name": name,
        "user_input": text,
        "ai_response": resp
    })