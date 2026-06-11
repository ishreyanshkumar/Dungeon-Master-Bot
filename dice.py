"""
dice.py — Dice roll parsing and resolution.
Supports standard RPG notation: NdM (e.g. 1d20, 2d6, 3d8).
"""
import re
import random
from dataclasses import dataclass


@dataclass
class DiceResult:
    notation: str       # e.g. "1d20"
    rolls: list[int]    # individual die results
    total: int          # sum
    modifier: int = 0   # flat bonus/penalty

    @property
    def display(self) -> str:
        rolls_str = " + ".join(str(r) for r in self.rolls)
        mod_str = f" + {self.modifier}" if self.modifier > 0 else (f" - {abs(self.modifier)}" if self.modifier < 0 else "")
        return f"**{self.notation}** → [{rolls_str}]{mod_str} = **{self.total + self.modifier}**"


# Matches patterns like "1d20", "2d6+3", "d20", "3d8-1"
_DICE_PATTERN = re.compile(r'\b(\d*)d(\d+)([+-]\d+)?\b', re.IGNORECASE)


def parse_and_roll(text: str) -> list[DiceResult]:
    """
    Scan text for dice notation and return a list of DiceResult objects.
    Returns an empty list if no dice notation is found.
    """
    results = []
    for match in _DICE_PATTERN.finditer(text):
        num_str, sides_str, mod_str = match.groups()
        num = int(num_str) if num_str else 1
        sides = int(sides_str)
        modifier = int(mod_str) if mod_str else 0

        # Sanity clamp
        num = min(num, 20)
        sides = min(sides, 100)

        rolls = [random.randint(1, sides) for _ in range(num)]
        notation = f"{num}d{sides}"
        results.append(DiceResult(notation=notation, rolls=rolls, total=sum(rolls), modifier=modifier))
    return results


def format_rolls(results: list[DiceResult]) -> str:
    """Format a list of DiceResult objects into a readable string."""
    return "\n".join(r.display for r in results)
