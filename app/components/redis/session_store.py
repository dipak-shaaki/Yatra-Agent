"""
Conversation history per session (sender), stored as a Redis list of JSON
{"role", "content"} messages under "session:{sender}:messages".

The JSON shape matches what the agent loop builds, so history can be spliced
straight into its `messages` list. TTL is reset on every write, so an active
conversation never expires mid-session while an abandoned one cleans itself
up.
"""

import json

from app.components.redis.client import get_redis_client
from app.configs.agent_config import MAX_STORED_HISTORY_MESSAGES, SESSION_TTL_SECONDS


def _messages_key(sender: str) -> str:
    return f"session:{sender}:messages"


def append_message(sender: str, role: str, content: str) -> None:
    """Append a message, trimming to the cap and resetting the TTL."""
    client = get_redis_client()
    key = _messages_key(sender)

    message = json.dumps({"role": role, "content": content})
    client.rpush(key, message)
    client.ltrim(key, -MAX_STORED_HISTORY_MESSAGES, -1)  # keep most recent N
    client.expire(key, SESSION_TTL_SECONDS)


def get_history(sender: str, limit: int | None = None) -> list[dict]:
    """Return the session's stored messages as [{"role", "content"}, ...]."""
    client = get_redis_client()
    key = _messages_key(sender)

    raw_messages = client.lrange(key, 0, -1)
    messages = [json.loads(m) for m in raw_messages]
    return messages[-limit:] if limit else messages


def clear_history(sender: str) -> None:
    """Delete a session's history (e.g. when the user explicitly starts over)."""
    client = get_redis_client()
    client.delete(_messages_key(sender))
