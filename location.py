"""
location.py — Location tracking for the adventure.
Uses a small LLM call to detect location changes each turn.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm


_LOCATION_PROMPT = ChatPromptTemplate.from_template(
    "Read the DM's response and determine if the player has moved to a new, named location. "
    "If yes, respond with only the location name (e.g. 'The Rusty Anchor Tavern', 'Forest of Whispers'). "
    "If the player has NOT moved to a clearly new location, respond with exactly 'same'.\n\n"
    "DM Response: {dm_response}"
)

_location_chain = _LOCATION_PROMPT | llm | StrOutputParser()


def detect_location(dm_response: str, current_location: str) -> str:
    """
    Return the new location name if the player moved, otherwise return current_location.
    """
    try:
        result = _location_chain.invoke({"dm_response": dm_response}).strip()
        if result.lower() == "same" or not result:
            return current_location
        # Filter out overly long responses (means the LLM got confused)
        if len(result) > 60:
            return current_location
        return result
    except Exception:
        return current_location
