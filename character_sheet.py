"""
character_sheet.py — Player character sheet management.
Detects inventory changes and HP changes from DM narrative using an LLM.
"""
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

_SHEET_PROMPT = ChatPromptTemplate.from_template(
    "Analyse the DM's response and extract any changes to the player's character.\n"
    "Return ONLY a valid JSON object with these keys (use null if nothing changed):\n"
    "  items_gained: list of item names gained (strings), or null\n"
    "  items_lost: list of item names lost/used/consumed, or null\n"
    "  hp_change: integer (positive = healed, negative = damage), or null\n"
    "  gold_change: integer (positive = gained, negative = spent), or null\n\n"
    "DM Response: {dm_response}\n\n"
    "JSON only, no explanation, no markdown:"
)

_sheet_chain = _SHEET_PROMPT | llm | StrOutputParser()


def default_sheet() -> dict:
    return {
        "name": "Unnamed Adventurer",
        "hp": 100,
        "max_hp": 100,
        "gold": 10,
        "inventory": []
    }


def update_sheet(sheet: dict, dm_response: str) -> tuple[dict, list[str]]:
    """
    Parse DM response for character sheet changes.
    Returns updated sheet and a list of human-readable change strings for display.
    """
    changes = []
    try:
        raw = _sheet_chain.invoke({"dm_response": dm_response}).strip()
        # Strip markdown fences if the LLM added them
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        if data.get("items_gained"):
            for item in data["items_gained"]:
                item = item.strip()
                if item and item not in sheet["inventory"]:
                    sheet["inventory"].append(item)
                    changes.append(f"📦 Gained: **{item}**")

        if data.get("items_lost"):
            for item in data["items_lost"]:
                item = item.strip()
                if item in sheet["inventory"]:
                    sheet["inventory"].remove(item)
                    changes.append(f"💨 Lost: **{item}**")

        if data.get("hp_change"):
            delta = int(data["hp_change"])
            sheet["hp"] = max(0, min(sheet["max_hp"], sheet["hp"] + delta))
            emoji = "💚" if delta > 0 else "💔"
            changes.append(f"{emoji} HP: {'+' if delta > 0 else ''}{delta} → {sheet['hp']}/{sheet['max_hp']}")

        if data.get("gold_change"):
            delta = int(data["gold_change"])
            sheet["gold"] = max(0, sheet["gold"] + delta)
            emoji = "💰" if delta > 0 else "💸"
            changes.append(f"{emoji} Gold: {'+' if delta > 0 else ''}{delta} → {sheet['gold']}")

    except Exception:
        pass  # Silent fail — character sheet is non-critical

    return sheet, changes
