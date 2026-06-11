"""
combat.py — Combat state tracker.
Detects when combat starts/ends from DM narrative and tracks enemy HP.
"""
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

# ── Combat detection ───────────────────────────────────────────────────────

_COMBAT_DETECT_PROMPT = ChatPromptTemplate.from_template(
    "Analyse the DM's response and the player's action. Determine combat state changes.\n"
    "Return ONLY a valid JSON object with these keys (null if not applicable):\n"
    "  combat_started: true if combat/battle clearly begins this turn, else null\n"
    "  combat_ended: true if combat clearly ends (victory/fled/peace), else null\n"
    "  enemy_name: string name of the main enemy if combat started, else null\n"
    "  enemy_max_hp: estimated HP integer for the enemy (20-200 based on description), or null\n"
    "  damage_to_enemy: integer damage dealt to enemy this turn, or null\n"
    "  victory: true if player won, false if fled/failed, null if ongoing\n"
    "  loot: list of item strings gained from victory, or null\n\n"
    "Player: {user_input}\nDM: {dm_response}\n\nJSON only:"
)
_combat_chain = _COMBAT_DETECT_PROMPT | llm | StrOutputParser()


def default_combat_state() -> dict:
    return {
        "active": False,
        "enemy_name": None,
        "enemy_hp": 0,
        "enemy_max_hp": 0,
        "turn": 0,
        "player_actions": [],
    }


def update_combat(state: dict, user_text: str, dm_resp: str) -> tuple[dict, list[str]]:
    """
    Update combat state from the latest turn.
    Returns updated state and a list of combat event strings to display.
    """
    events = []
    try:
        raw = _combat_chain.invoke({"user_input": user_text, "dm_response": dm_resp}).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        # Combat starts
        if data.get("combat_started") and not state["active"]:
            enemy = data.get("enemy_name") or "Unknown Enemy"
            max_hp = int(data.get("enemy_max_hp") or 60)
            state.update({
                "active": True,
                "enemy_name": enemy,
                "enemy_hp": max_hp,
                "enemy_max_hp": max_hp,
                "turn": 1,
                "player_actions": [],
            })
            events.append(f"⚔️ **Combat begins!** You face **{enemy}** ({max_hp} HP)")

        # Combat ongoing — apply damage
        elif state["active"]:
            state["turn"] += 1
            dmg = data.get("damage_to_enemy")
            if dmg:
                dmg = int(dmg)
                state["enemy_hp"] = max(0, state["enemy_hp"] - dmg)
                events.append(f"💥 You deal **{dmg} damage** → {state['enemy_name']}: {state['enemy_hp']}/{state['enemy_max_hp']} HP")

        # Combat ends
        if data.get("combat_ended") and state["active"]:
            victory = data.get("victory")
            loot = data.get("loot") or []
            if victory:
                events.append(f"🏆 **Victory!** {state['enemy_name']} is defeated!")
                if loot:
                    events.append("💰 Loot: " + ", ".join(f"**{l}**" for l in loot))
            elif victory is False:
                events.append(f"🏃 You escaped from **{state['enemy_name']}**.")
            state = default_combat_state()

    except Exception:
        pass

    return state, events


def enemy_hp_bar_html(enemy_hp: int, enemy_max_hp: int, enemy_name: str) -> str:
    if enemy_max_hp == 0:
        return ""
    pct = max(0, min(100, int(enemy_hp / enemy_max_hp * 100)))
    color = "#2d6a3f" if pct > 60 else ("#b5770d" if pct > 30 else "#8b1a1a")
    return (
        f"<div style='margin:6px 0'>"
        f"<div style='font-size:0.8rem;color:#c9a84c;margin-bottom:3px'>⚔ {enemy_name}: {enemy_hp}/{enemy_max_hp} HP</div>"
        f"<div style='background:#1a0a0a;border:1px solid #2e2416;border-radius:4px;height:8px;overflow:hidden'>"
        f"<div style='width:{pct}%;height:100%;background:{color};border-radius:4px;transition:width 0.4s'></div>"
        f"</div></div>"
    )
