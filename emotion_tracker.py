"""
emotion_tracker.py — Emotional arc tracking using VADER sentiment analysis.

What this does:
  Every DM response is scored by VADER (Valence Aware Dictionary and
  sEntiment Reasoner), a rule-based sentiment analyser tuned for informal
  text. Unlike simple positive/negative, we track the TRAJECTORY across
  turns — giving the DM emotional arc context:

    "The story has been building tension (declining positivity) over the
     last 5 turns. Recent tone: Tense → Dread → Horror."

  This feeds into the story prompt so the DM can:
    - Provide relief after extended tension  
    - Escalate naturally after calm scenes
    - Avoid emotional monotony (all-doom or all-jubilant)

VADER scores return a compound in [-1, +1]:
  ≥  0.5  →  Positive
  ≤ -0.5  →  Negative
  In between →  Neutral/Mixed

We also track MOMENTUM: the slope of compound scores over the last N turns.
Negative slope = story darkening. Positive slope = story brightening.
"""

import math
from dataclasses import dataclass, field
from typing import List

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _VADER_OK = True
except ImportError:
    _VADER_OK = False


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class EmotionPoint:
    turn:      int
    compound:  float   # VADER compound score [-1, 1]
    label:     str     # "Positive" / "Negative" / "Neutral"
    tone:      str     # Finer label: "Triumphant", "Dread", etc.


@dataclass
class EmotionState:
    history:   List[EmotionPoint] = field(default_factory=list)
    momentum:  float = 0.0          # Slope of last N compounds (−=darkening, +=brightening)
    arc_label: str   = "Neutral"    # Overall arc description


# ── Compound → tone label mapping ───────────────────────────────────────────

def _compound_to_tone(compound: float) -> tuple[str, str]:
    """Return (broad_label, fine_tone) from a compound score."""
    if compound >= 0.7:
        return "Positive", "Triumphant"
    elif compound >= 0.4:
        return "Positive", "Hopeful"
    elif compound >= 0.1:
        return "Positive", "Calm"
    elif compound >= -0.1:
        return "Neutral",  "Uncertain"
    elif compound >= -0.4:
        return "Negative", "Tense"
    elif compound >= -0.6:
        return "Negative", "Dread"
    else:
        return "Negative", "Horror"


# ── Momentum calculation ─────────────────────────────────────────────────────

def _linear_slope(values: List[float]) -> float:
    """
    Compute the slope of a linear regression through the given values.
    Used to quantify whether the emotional arc is rising or falling.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def _arc_description(momentum: float, latest_tone: str) -> str:
    """Human-readable arc description for the DM prompt."""
    if momentum < -0.05:
        direction = "darkening"
    elif momentum > 0.05:
        direction = "brightening"
    else:
        direction = "holding steady"
    return f"{direction.title()} ({latest_tone})"


# ── Public API ───────────────────────────────────────────────────────────────

def score_turn(text: str) -> float:
    """Return VADER compound score for a block of text, or 0.0 if unavailable."""
    if not _VADER_OK or not text:
        return 0.0
    try:
        return _vader.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def update_emotion(state: EmotionState, dm_response: str, turn: int) -> EmotionState:
    """
    Score the latest DM response and update the emotion state.
    Returns the updated EmotionState.
    """
    compound = score_turn(dm_response)
    label, tone = _compound_to_tone(compound)

    state.history.append(EmotionPoint(turn=turn, compound=compound, label=label, tone=tone))

    # Keep last 20 turns for momentum calculation
    if len(state.history) > 20:
        state.history = state.history[-20:]

    # Momentum: slope over last 5 turns
    recent = [p.compound for p in state.history[-5:]]
    state.momentum = _linear_slope(recent)
    state.arc_label = _arc_description(state.momentum, tone)

    return state


def emotion_context_string(state: EmotionState, n_recent: int = 3) -> str:
    """
    Build a concise string for the DM prompt describing the emotional arc.

    Example output:
        Emotional Arc: Darkening (Horror) over last 5 turns.
        Recent tones: Tense → Dread → Horror
        Suggestion: consider providing a moment of relief or revelation.
    """
    if not state.history:
        return "Emotional Arc: Story just beginning."

    recent_tones = " → ".join(p.tone for p in state.history[-n_recent:])
    arc = state.arc_label

    # Craft a suggestion for the DM
    m = state.momentum
    if m < -0.08:
        suggestion = "Consider a moment of relief, discovery, or dark humour to prevent emotional fatigue."
    elif m > 0.08:
        suggestion = "The story is brightening — a complication or twist could deepen engagement."
    else:
        suggestion = "Maintain current tension; a small escalation or quiet character beat would work well."

    return (
        f"Emotional Arc: {arc}.\n"
        f"Recent tones: {recent_tones}\n"
        f"DM guidance: {suggestion}"
    )


def get_emotion_history(state: EmotionState) -> List[dict]:
    """Return the full emotion history as a list of dicts (for charting)."""
    return [
        {"turn": p.turn, "compound": p.compound, "tone": p.tone, "label": p.label}
        for p in state.history
    ]
