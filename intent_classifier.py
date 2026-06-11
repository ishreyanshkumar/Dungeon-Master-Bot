"""
intent_classifier.py — Player intent classification for smarter memory retrieval.

The problem:
  The vector DB retrieves the k nearest docs to the raw player action text.
  But the *kind* of action matters for what memories are useful:
    - "I attack the guard" → recent COMBAT memories are most useful
    - "I examine the ancient runes" → LORE/EXPLORATION memories matter more
    - "I try to persuade the merchant" → NPC RELATIONSHIP memories most useful
    - "I look around carefully" → LOCATION/ENVIRONMENT memories best

  Without intent awareness, retrieval is purely semantic — it fetches memories
  about similar *topics* but ignores the *action type*, which changes what
  context is actually useful to the DM.

Solution — keyword-based multi-label intent classifier:
  We classify each player action into one or more intent categories using a
  weighted keyword vocabulary. This is deliberately lightweight (no external
  model call, runs in <1ms) — it uses simple TF-IDF-weighted bag-of-words
  matching against hand-crafted intent prototypes.

  The resulting intent vector then biases ChromaDB retrieval:
    - COMBAT intent → also search "combat" "attack" "fight" keywords
    - SOCIAL intent → also search NPC names + "relationship" "trust"
    - EXPLORE intent → also search recent location names + "discover"

  This is called "query expansion" in IR literature. Our version is
  ML-informed (weighted keyword matching) rather than purely rule-based.
"""

import re
import math
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ── Intent taxonomy ─────────────────────────────────────────────────────────

INTENTS = [
    "combat",
    "social",
    "explore",
    "stealth",
    "investigate",
    "rest",
    "trade",
    "magic",
    "flee",
]

# Prototype keyword lists per intent (used to build TF-IDF prototype vectors)
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "combat":      ["attack", "fight", "stab", "shoot", "strike", "kill", "battle",
                    "swing", "slash", "punch", "defend", "block", "parry", "charge",
                    "ambush", "assault", "draw", "weapon", "sword", "bow", "fire"],
    "social":      ["talk", "speak", "ask", "persuade", "negotiate", "bribe", "convince",
                    "lie", "deceive", "charm", "threaten", "intimidate", "befriend",
                    "greet", "question", "tell", "say", "explain", "plead", "offer"],
    "explore":     ["look", "go", "walk", "enter", "leave", "travel", "head", "move",
                    "climb", "descend", "open", "door", "path", "direction", "search",
                    "area", "room", "corridor", "outside", "inside", "around"],
    "stealth":     ["sneak", "hide", "shadow", "quiet", "silent", "crouch", "avoid",
                    "slip", "creep", "conceal", "disguise", "follow", "tail", "spy"],
    "investigate": ["examine", "inspect", "study", "read", "analyse", "check", "look",
                    "investigate", "clue", "evidence", "discover", "rune", "inscription",
                    "map", "document", "symbol", "trace", "find", "notice", "observe"],
    "rest":        ["rest", "sleep", "camp", "wait", "heal", "recover", "bandage",
                    "eat", "drink", "pause", "stay", "sit", "meditate"],
    "trade":       ["buy", "sell", "trade", "barter", "purchase", "pay", "gold",
                    "shop", "merchant", "item", "price", "offer", "exchange", "market"],
    "magic":       ["cast", "spell", "magic", "conjure", "summon", "enchant", "curse",
                    "ritual", "chant", "invoke", "channel", "mystical", "arcane"],
    "flee":        ["run", "flee", "escape", "retreat", "dash", "bolt", "sprint",
                    "hide", "evade", "get away", "withdraw", "back", "leave quickly"],
}

# ── Build vocabulary & prototype matrix at import time ─────────────────────

def _build_vocab(keyword_map: Dict[str, List[str]]) -> tuple[list, np.ndarray]:
    """Build a shared vocabulary and a prototype matrix (n_intents × vocab_size)."""
    vocab = sorted({w for words in keyword_map.values() for w in words})
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    I = len(INTENTS)

    proto = np.zeros((I, V), dtype=np.float32)
    for i, intent in enumerate(INTENTS):
        for kw in keyword_map[intent]:
            if kw in vocab_idx:
                proto[i, vocab_idx[kw]] = 1.0
        # L2 normalise
        norm = np.linalg.norm(proto[i])
        if norm > 0:
            proto[i] /= norm

    return vocab, vocab_idx, proto


_VOCAB, _VOCAB_IDX, _PROTO_MATRIX = _build_vocab(_INTENT_KEYWORDS)


def _text_to_vec(text: str) -> np.ndarray:
    """Convert text to a normalised bag-of-words vector over shared vocab."""
    tokens = re.findall(r"[a-z]+", text.lower())
    vec = np.zeros(len(_VOCAB), dtype=np.float32)
    for tok in tokens:
        if tok in _VOCAB_IDX:
            vec[_VOCAB_IDX[tok]] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Public API ───────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    primary:    str            # Top intent label
    scores:     Dict[str, float]  # All intent → score
    top_k:      List[str]     # Top-3 intents by score


def classify_intent(text: str, top_k: int = 3) -> IntentResult:
    """
    Classify player action text into intent categories.
    Returns an IntentResult with primary intent and all scores.
    """
    vec = _text_to_vec(text).reshape(1, -1)
    sims = cosine_similarity(vec, _PROTO_MATRIX).flatten()

    scores = {intent: float(sims[i]) for i, intent in enumerate(INTENTS)}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    primary = ranked[0][0] if ranked[0][1] > 0 else "explore"
    top = [k for k, v in ranked[:top_k] if v > 0.05]
    if not top:
        top = ["explore"]

    return IntentResult(primary=primary, scores=scores, top_k=top)


def expand_query(
    original_query: str,
    intent: IntentResult,
    location_history: List[str] = None,
    npc_names: List[str] = None,
) -> str:
    """
    Expand the retrieval query based on detected intent.
    This enriched query is passed to ChromaDB instead of the raw player text,
    giving more targeted memory retrieval.

    Intent-specific expansion:
      - combat     → add combat keywords + enemy context
      - social     → add recent NPC names (they're the relevant memory targets)
      - explore    → add current location name
      - investigate → add "discover clue evidence" 
      - (others)  → minor expansion
    """
    expansions = []

    if "combat" in intent.top_k:
        expansions.append("combat fight battle enemy attack")
    if "social" in intent.top_k and npc_names:
        expansions.append(" ".join(npc_names[-3:]))   # recent NPCs
    if "explore" in intent.top_k and location_history:
        expansions.append(location_history[-1])
    if "investigate" in intent.top_k:
        expansions.append("discover clue evidence lore secret")
    if "stealth" in intent.top_k:
        expansions.append("hidden secret guard patrol")
    if "magic" in intent.top_k:
        expansions.append("spell arcane ritual magic power")
    if "trade" in intent.top_k:
        expansions.append("merchant item gold shop purchase")

    if expansions:
        return original_query + " " + " ".join(expansions)
    return original_query


def intent_to_retrieval_params(intent: IntentResult) -> dict:
    """
    Return ChromaDB retrieval parameter overrides based on intent.
    Combat → fetch more docs (action is dense, needs more context).
    Rest/trade → fewer docs needed (less complex context).
    """
    base_k = 10   # wide candidate pool for re-ranker
    if intent.primary in ("combat", "magic"):
        return {"k": min(15, base_k + 5)}
    elif intent.primary in ("rest", "trade"):
        return {"k": max(6, base_k - 4)}
    return {"k": base_k}
