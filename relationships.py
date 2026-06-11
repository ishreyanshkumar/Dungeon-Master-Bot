"""
relationships.py — NPC relationship tracker.
Tracks trust/hostility scores for each NPC the player interacts with.
Scores range from -100 (mortal enemy) to +100 (trusted ally).
"""
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

_REL_PROMPT = ChatPromptTemplate.from_template(
    "Analyse this interaction between the player and {npc_name}.\n"
    "How did the player's action affect their relationship?\n"
    "Return ONLY a JSON object: {{\"delta\": <integer from -30 to +30>}}\n"
    "Positive = relationship improved, Negative = relationship worsened, 0 = no change.\n\n"
    "Player: {user_input}\nDM: {dm_response}\n\nJSON only:"
)
_rel_chain = _REL_PROMPT | llm | StrOutputParser()

# Relationship tiers
TIERS = [
    (-100, -60, "💀", "Mortal Enemy",   "#8b1a1a"),
    (-59,  -30, "😠", "Hostile",         "#c83a2a"),
    (-29,   -1, "😒", "Distrustful",     "#8b5a1a"),
    (0,      0, "😐", "Neutral",         "#8a7a60"),
    (1,     29, "🙂", "Friendly",        "#4a8c5c"),
    (30,    59, "😊", "Trusted",         "#2d6a3f"),
    (60,   100, "🤝", "Sworn Ally",      "#c9a84c"),
]


def get_tier(score: int) -> tuple[str, str, str]:
    """Return (emoji, label, color) for a relationship score."""
    for lo, hi, emoji, label, color in TIERS:
        if lo <= score <= hi:
            return emoji, label, color
    return "😐", "Neutral", "#8a7a60"


def update_relationship(relationships: dict, npc_name: str, user_text: str, dm_resp: str) -> tuple[dict, str | None]:
    """
    Update the relationship score for an NPC.
    Returns updated dict and an event string if the tier changed.
    """
    try:
        raw = _rel_chain.invoke({
            "npc_name": npc_name,
            "user_input": user_text,
            "dm_response": dm_resp,
        }).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        delta = int(data.get("delta", 0))

        if delta == 0:
            return relationships, None

        old_score = relationships.get(npc_name, 0)
        new_score = max(-100, min(100, old_score + delta))

        old_emoji, old_label, _ = get_tier(old_score)
        new_emoji, new_label, _ = get_tier(new_score)

        relationships[npc_name] = new_score

        # Only announce if tier changed
        if old_label != new_label:
            arrow = "📈" if delta > 0 else "📉"
            event = f"{arrow} **{npc_name}**: {old_emoji} {old_label} → {new_emoji} {new_label}"
            return relationships, event

        return relationships, None

    except Exception:
        return relationships, None
