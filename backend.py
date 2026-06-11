"""
backend.py — Core orchestration with ML-enhanced memory retrieval pipeline.

Retrieval pipeline (per turn):
  1. Classify player intent  (intent_classifier.py)
  2. Expand retrieval query  (intent-aware query augmentation)
  3. Wide candidate fetch    (ChromaDB k=10-15)
  4. ML re-rank + MMR        (memory_ranker.py)
  5. Graph context injection (narrative_graph.py — 1-2 hop entity connections)
  6. Emotion arc context     (emotion_tracker.py — arc guidance for DM)
  7. Assemble full prompt    (story generation)
  8. Post-turn: consolidate  (memory_consolidator.py — runs every 20 turns)
"""

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import llm, embs
from character_memory import init_chars, get_char_mem
from quest import init_quests
from character_sheet import default_sheet
from lore import init_lore
from stats import default_stats
from combat import default_combat_state
from relationships import get_tier

# ── ML memory modules ────────────────────────────────────────────────────────
from memory_ranker import rerank_memories
from memory_consolidator import should_consolidate, consolidate_memories, get_memory_count
from emotion_tracker import EmotionState, update_emotion, emotion_context_string
from intent_classifier import classify_intent, expand_query, intent_to_retrieval_params
from narrative_graph import (
    NarrativeGraph, load_graph, save_graph,
    extract_and_update_graph, graph_context_for_query, graph_stats,
)

# ─────────────────────────────────────────────────────────────────────────────
# NPC detection
# ─────────────────────────────────────────────────────────────────────────────

_NPC_PROMPT = ChatPromptTemplate.from_template(
    "Read the following text. Identify the single most important proper name of a character or creature "
    "being interacted with. Do not identify the player. "
    "If no specific character is mentioned, respond with exactly 'None'.\n\n"
    "Text: {text}\n\nCharacter Name:"
)
_npc_chain = _NPC_PROMPT | llm | StrOutputParser()


def find_npc(text: str) -> str | None:
    name = _npc_chain.invoke({"text": text}).strip()
    if "none" in name.lower() or len(name.split()) > 4:
        return None
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Genre openings
# ─────────────────────────────────────────────────────────────────────────────

GENRE_OPENINGS = {
    "High Fantasy": (
        "You stand at the gates of Arathorn, the ancient city of mages, as a red moon rises over the spires. "
        "Your name is whispered in the prophecy carved above the gate. A hooded figure beckons. What do you do?"
    ),
    "Dark Sci-Fi": (
        "You wake inside a cryo-pod aboard the derelict station Erebus-9, oxygen at 14%. Emergency lights "
        "strobe red. The evacuation logs are corrupted. Through the viewport, something massive drifts closer. What do you do?"
    ),
    "Lovecraftian Horror": (
        "You awaken in the study of Professor Aldous Crane, who has been missing for three weeks. "
        "His journal lies open to a page covered in spiralling, incoherent script — except for the last line, "
        "written in your own handwriting. What do you do?"
    ),
    "Pirates": (
        "You awaken to the smell of brine and damp wood in the dimly lit hold of a creaking ship. "
        "You have no memory of how you got here. A single barred door stands before you. "
        "Through the planks you hear cannon fire. What do you do?"
    ),
    "Dungeon Crawler": (
        "Torch in hand, you descend the last step into the First Vault beneath Molder Keep. "
        "Bones crunch underfoot. Somewhere ahead, something breathes in the dark. "
        "Your map ends here. What do you do?"
    ),
    "Post-Apocalyptic": (
        "The Geiger counter on your wrist clicks steadily as you emerge from Vault 7 for the first time. "
        "The skyline juts like broken teeth against an orange sky. A figure watches from the rubble. What do you do?"
    ),
    "Noir Mystery": (
        "Rain hammers the windows of your office. The envelope on your desk holds a photograph, a key, "
        "and a name: Elara Voss. Found dead this morning. Police are calling it a suicide. You know better. What do you do?"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Memory initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_memory(sess, campaign: str = "default", genre: str = "Pirates", player_name: str = "Adventurer"):
    """Initialise all session state. Runs once per session."""
    if "messages" not in sess.session_state:
        opening = GENRE_OPENINGS.get(genre, GENRE_OPENINGS["Pirates"])
        sess.session_state.campaign = campaign
        sess.session_state.genre    = genre

        sess.session_state.messages          = [{"role": "assistant", "content": opening}]
        sess.session_state.short_term_memory = []

        # Core memory stores
        dm_db = Chroma(embedding_function=embs, persist_directory=f"./campaigns/{campaign}/world")
        sess.session_state.retriever      = dm_db.as_retriever(search_kwargs={"k": 10})
        sess.session_state.char_retriever = init_chars(campaign)
        sess.session_state.quest_retriever= init_quests(campaign)
        sess.session_state.lore_db        = init_lore(campaign)

        # Character
        sheet = default_sheet()
        sheet["name"] = player_name
        sess.session_state.character = sheet

        # World state
        sess.session_state.location_history = ["Unknown"]
        sess.session_state.current_mood     = "mysterious"
        sess.session_state.combat           = default_combat_state()
        sess.session_state.relationships    = {}
        sess.session_state.stats            = default_stats()

        # ── ML memory state ──────────────────────────────────────────────
        # Emotion arc tracker
        sess.session_state.emotion_state    = EmotionState()
        # Narrative knowledge graph
        sess.session_state.narrative_graph  = load_graph(campaign)
        # Intent of last turn (for UI display)
        sess.session_state.last_intent      = None
        # Memory stats for display
        sess.session_state.memory_stats     = {"world": 0, "consolidated": 0}

        # Flash queues
        sess.session_state.new_quest_flash      = []
        sess.session_state.new_lore_flash       = None
        sess.session_state.combat_events        = []
        sess.session_state.relationship_events  = []


# ─────────────────────────────────────────────────────────────────────────────
# ML-enhanced memory retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_world_memories(
    user_text:        str,
    retriever,
    location_history: list,
    npc_names:        list,
    narrative_graph:  NarrativeGraph,
) -> tuple[str, object]:
    """
    Full ML retrieval pipeline:
      1. Classify intent
      2. Expand query
      3. Fetch wide candidate pool
      4. Re-rank with TF-IDF + dense + MMR
      5. Inject graph context
    Returns (formatted context string, intent result).
    """
    # ── 1. Intent classification ──
    intent = classify_intent(user_text)

    # ── 2. Query expansion ──
    expanded = expand_query(
        user_text,
        intent,
        location_history=location_history,
        npc_names=npc_names,
    )

    # ── 3. Wide candidate fetch ──
    params = intent_to_retrieval_params(intent)
    retriever.search_kwargs = params
    try:
        candidate_docs = retriever.get_relevant_documents(expanded)
        raw_texts = [d.page_content for d in candidate_docs]
    except Exception:
        raw_texts = []

    # ── 4. ML re-rank (TF-IDF + dense + recency + MMR) ──
    reranked = rerank_memories(
        query=user_text,
        docs=raw_texts,
        embs_model=embs,
        top_k=3,
        recency_boost=True,
        lambda_mmr=0.65,
    )

    # ── 5. Graph context injection ──
    graph_ctx = graph_context_for_query(narrative_graph, user_text)

    # Assemble final context
    mem_block = "\n".join(f"- {doc}" for doc in reranked) if reranked else "No relevant past events."
    if graph_ctx:
        mem_block = graph_ctx + "\n\n" + mem_block

    return mem_block, intent


# ─────────────────────────────────────────────────────────────────────────────
# Story generation — streaming
# ─────────────────────────────────────────────────────────────────────────────

_STORY_PROMPT = ChatPromptTemplate.from_template("""You are a master Dungeon Master running a {genre} adventure. Your world is vivid, consistent, and deeply alive.

RULES:
- Write 2-4 sentences of immersive, cinematic narrative continuing from the player's action.
- Be specific: name things, describe smells/sounds/textures, make the world tangible.
- If an NPC is present, use their memory — they have a personality, history, and feelings toward the player.
- If dice were rolled, let the result drive success/failure naturally without mentioning mechanics.
- Match tone to the scene — tense moments: short staccato sentences. Calm: let it breathe.
- End with exactly 3 numbered action suggestions in *italics*.
- Never break character. Never say "as an AI". Never expose your reasoning.

--- Emotional Arc Guidance ---
{emotion_context}

--- Relationship Context ---
{relationship_context}

--- NPC Memory for "{npc_name}" ---
{character_memory}

--- Relevant Past Events (ML-Retrieved, Re-Ranked) ---
{relevant_docs}

--- Recent History (Last 4 Turns) ---
{short_term_history}

--- Player's Action ---
{user_input}
""")


def gen_story_stream(
    user_text:        str,
    hist:             str,
    retriever,
    char_ret,
    genre:            str  = "Pirates",
    relationships:    dict = None,
    location_history: list = None,
    emotion_state:    EmotionState = None,
    narrative_graph:  NarrativeGraph = None,
    npc_names:        list = None,
):
    """
    Generator yielding story text chunks for streaming.
    Uses the full ML retrieval pipeline internally.
    """
    npc      = find_npc(user_text)
    char_mem = get_char_mem(char_ret, npc) if npc else "N/A"

    # ML retrieval
    rel_docs, intent = retrieve_world_memories(
        user_text, retriever,
        location_history or [],
        npc_names or [],
        narrative_graph or NarrativeGraph(),
    )

    # Relationship context
    rel_lines = []
    if relationships:
        for name, score in list(relationships.items())[:5]:
            emoji, label, _ = get_tier(score)
            rel_lines.append(f"  {name}: {emoji} {label} ({score:+d})")
    rel_context = "\n".join(rel_lines) if rel_lines else "No established relationships yet."

    # Emotion arc context
    emo_ctx = emotion_context_string(emotion_state) if emotion_state else "Story just beginning."

    # Build runnable (we bypass the retriever in the chain since we already fetched)
    prompt = _STORY_PROMPT.format_messages(
        genre=genre,
        emotion_context=emo_ctx,
        relationship_context=rel_context,
        npc_name=npc or "N/A",
        character_memory=char_mem,
        relevant_docs=rel_docs,
        short_term_history=hist,
        user_input=user_text,
    )

    yield from (llm | StrOutputParser()).stream(prompt)

    return intent  # caller can ignore, intent is also stored in session


def retrieve_intent_only(user_text: str) -> object:
    """Quick intent classification without full retrieval (for UI display)."""
    return classify_intent(user_text)


# ─────────────────────────────────────────────────────────────────────────────
# Post-turn memory maintenance
# ─────────────────────────────────────────────────────────────────────────────

def post_turn_updates(
    user_text:       str,
    full_resp:       str,
    retriever,
    narrative_graph: NarrativeGraph,
    emotion_state:   EmotionState,
    turn_count:      int,
    campaign:        str,
) -> tuple[EmotionState, NarrativeGraph, dict]:
    """
    Run all post-turn ML memory updates:
      - Summarise turn → long-term memory
      - Update emotion arc
      - Extract narrative graph edges
      - Periodic memory consolidation (every 20 turns)
    Returns updated (emotion_state, narrative_graph, consolidation_info).
    """
    consolidation_info = {}

    # ── 1. Long-term memory summary ──
    turn_sum = sum_turn(user_text, full_resp)
    if turn_sum:
        retriever.vectorstore.add_texts([turn_sum])

    # ── 2. Emotion arc update ──
    emotion_state = update_emotion(emotion_state, full_resp, turn=turn_count)

    # ── 3. Narrative graph extraction ──
    extract_and_update_graph(narrative_graph, full_resp, campaign)

    # ── 4. Memory consolidation (every 20 turns) ──
    if turn_count > 0 and turn_count % 20 == 0:
        if should_consolidate(retriever.vectorstore):
            reduced = consolidate_memories(retriever.vectorstore)
            mem_count = get_memory_count(retriever.vectorstore)
            consolidation_info = {
                "ran": True,
                "reduced": reduced,
                "remaining": mem_count,
            }

    return emotion_state, narrative_graph, consolidation_info


# ─────────────────────────────────────────────────────────────────────────────
# Summarisation chains
# ─────────────────────────────────────────────────────────────────────────────

_TURN_SUM_PROMPT = ChatPromptTemplate.from_template(
    "Summarise the key event from this turn in one concise sentence for a world-history log.\n"
    "Player: {user_input}\nDM: {ai_response}"
)
_turn_sum_chain = _TURN_SUM_PROMPT | llm | StrOutputParser()


def sum_turn(user_text: str, resp: str) -> str:
    return _turn_sum_chain.invoke({"user_input": user_text, "ai_response": resp})


_NPC_SUM_PROMPT = ChatPromptTemplate.from_template(
    "In one concise sentence, summarise what this turn reveals about {npc_name}'s personality, "
    "knowledge, or relationship with the player.\n"
    "Player: {user_input}\nDM: {ai_response}"
)
_npc_sum_chain = _NPC_SUM_PROMPT | llm | StrOutputParser()


def sum_npc_turn(name: str, user_text: str, resp: str) -> str:
    return _npc_sum_chain.invoke({"npc_name": name, "user_input": user_text, "ai_response": resp})
