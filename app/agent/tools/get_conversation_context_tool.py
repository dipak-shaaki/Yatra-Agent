"""
Pulls recent conversation history for the current session, for referential
follow-up queries ("that place", "the one you mentioned", "compare it with
the second one"). The agent calls this when a query can't be resolved from
its own text alone.
"""
from app.components.redis.session_store import get_history

DEFAULT_CONTEXT_TURNS = 6  # last 6 messages (~3 user/assistant exchanges)


def get_conversation_context(sender: str, num_turns: int = DEFAULT_CONTEXT_TURNS) -> list[dict]:
    """
    Returns the most recent messages for this session, oldest first, as
    [{"role": ..., "content": ...}, ...]. Empty list if no history exists.
    """
    return get_history(sender, limit=num_turns)