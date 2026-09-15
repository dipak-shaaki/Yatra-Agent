"""
Recent conversation history for the current session, for referential
follow-ups ("that place", "the one you mentioned").
"""

from app.components.redis.session_store import get_history

DEFAULT_CONTEXT_TURNS = 6  # ~3 user/assistant exchanges


def get_conversation_context(
    sender: str, num_turns: int = DEFAULT_CONTEXT_TURNS
) -> list[dict]:
    """Return the session's most recent messages, oldest first, as
    [{"role", "content"}, ...]; empty list if no history exists."""
    return get_history(sender, limit=num_turns)
