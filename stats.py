"""
stats.py — Session statistics and death handling.
Tracks turns played, enemies defeated, items found, distance travelled.
Also handles the death/game-over state.
"""


def default_stats() -> dict:
    return {
        "turns": 0,
        "enemies_defeated": 0,
        "items_found": 0,
        "locations_visited": 0,
        "quests_completed": 0,
        "dice_rolled": 0,
        "deaths": 0,
    }


def increment(stats: dict, key: str, amount: int = 1) -> dict:
    stats[key] = stats.get(key, 0) + amount
    return stats


def is_dead(character: dict) -> bool:
    return character.get("hp", 1) <= 0


def respawn(character: dict, stats: dict) -> tuple[dict, dict, str]:
    """
    Handle death: deduct gold, restore half HP, increment death counter.
    Returns updated character, updated stats, and a narrative string.
    """
    stats = increment(stats, "deaths")
    penalty_gold = max(0, character["gold"] // 4)
    character["gold"] = max(0, character["gold"] - penalty_gold)
    character["hp"] = character["max_hp"] // 2

    narrative = (
        f"💀 **You have fallen.**\n\n"
        f"*The darkness takes you... but fate is not yet done with you.*\n\n"
        f"You awaken, battered and disoriented, at the last safe place you remember. "
        f"Your wounds have been tended by unknown hands. "
        f"You find {penalty_gold} gold missing from your pouch — payment, perhaps, for your resurrection.\n\n"
        f"HP restored to {character['hp']}/{character['max_hp']}. Deaths: {stats['deaths']}."
    )
    return character, stats, narrative


def format_stats(stats: dict) -> str:
    """Return a markdown-formatted stats block."""
    lines = [
        f"🔄 **Turns Played:** {stats.get('turns', 0)}",
        f"⚔️ **Enemies Defeated:** {stats.get('enemies_defeated', 0)}",
        f"🎒 **Items Found:** {stats.get('items_found', 0)}",
        f"🗺️ **Locations Visited:** {stats.get('locations_visited', 0)}",
        f"📜 **Quests Completed:** {stats.get('quests_completed', 0)}",
        f"🎲 **Dice Rolled:** {stats.get('dice_rolled', 0)}",
        f"💀 **Deaths:** {stats.get('deaths', 0)}",
    ]
    return "\n\n".join(lines)
