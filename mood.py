"""
mood.py — Scene atmosphere detection.
Detects the emotional tone of each DM response and provides
ambient styling cues to the UI.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

_MOOD_PROMPT = ChatPromptTemplate.from_template(
    "Read the DM's response. Classify the dominant atmosphere in ONE word from this list:\n"
    "tense, peaceful, mysterious, triumphant, dreadful, romantic, humorous, melancholic, dangerous, wondrous\n\n"
    "DM Response: {dm_response}\n\nOne word only:"
)
_mood_chain = _MOOD_PROMPT | llm | StrOutputParser()

VALID_MOODS = {
    "tense", "peaceful", "mysterious", "triumphant", "dreadful",
    "romantic", "humorous", "melancholic", "dangerous", "wondrous"
}

# Mood → (emoji, accent_color, border_color)
MOOD_STYLES = {
    "tense":      ("😤", "#c9a84c", "#8b5a1a"),
    "peaceful":   ("🌿", "#4a8c5c", "#2a5c3c"),
    "mysterious": ("🌑", "#7a5cab", "#4a3c7a"),
    "triumphant": ("🏆", "#e8c87a", "#c9a84c"),
    "dreadful":   ("💀", "#8b1a1a", "#5c0a0a"),
    "romantic":   ("🌹", "#c45c7a", "#8b3a5a"),
    "humorous":   ("😄", "#c9a84c", "#8b7a2a"),
    "melancholic":("🌧️", "#5a7a9a", "#3a5a7a"),
    "dangerous":  ("⚠️", "#c83a2a", "#8b1a1a"),
    "wondrous":   ("✨", "#7ac8e8", "#4a8aaa"),
}


def detect_mood(dm_response: str) -> str:
    """Return the mood string for the current scene."""
    try:
        result = _mood_chain.invoke({"dm_response": dm_response}).strip().lower()
        # Clean up — sometimes the model adds punctuation
        result = result.split()[0].strip(".,!?")
        return result if result in VALID_MOODS else "mysterious"
    except Exception:
        return "mysterious"


def get_mood_style(mood: str) -> tuple[str, str, str]:
    """Return (emoji, accent_color, border_color) for a mood."""
    return MOOD_STYLES.get(mood, MOOD_STYLES["mysterious"])
