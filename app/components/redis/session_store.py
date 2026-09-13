"""
Conversation history storage per session (sender), with TTL-based expiry.

Design notes:
- Each session's messages stored as a Redis list under key "session:{sender}:messages"
- Each message stored as a JSON string (role + content), matching the shape
  the agent loop already builds internally, so history can be spliced
  straight back into agent.py's `messages` list.
- TTL is reset on every write, so an active conversation never expires
  mid-session, but an abandoned one cleans itself up automatically.
"""
import json

from app.components.redis.client import get_redis_client
from app.configs.agent_config import SESSION_TTL_SECONDS, MAX_STORED_HISTORY_MESSAGES


def _messages_key(sender: str) -> str:
    return f"session:{sender}:messages"


def append_message(sender: str, role: str, content: str) -> None:
    """Append one message to a session's history, trimming to the cap and resetting TTL."""
    client = get_redis_client()
    key = _messages_key(sender)

    message = json.dumps({"role": role, "content": content})
    client.rpush(key, message)
    client.ltrim(key, -MAX_STORED_HISTORY_MESSAGES, -1)  # keep only the most recent N
    client.expire(key, SESSION_TTL_SECONDS)


def get_history(sender: str, limit: int | None = None) -> list[dict]:
    """Returns this session's stored messages as [{"role": ..., "content": ...}, ...]."""
    client = get_redis_client()
    key = _messages_key(sender)

    raw_messages = client.lrange(key, 0, -1)
    messages = [json.loads(m) for m in raw_messages]
    return messages[-limit:] if limit else messages


def clear_history(sender: str) -> None:
    """Wipe a session's history (e.g. if the user explicitly starts over)."""
    client = get_redis_client()
    client.delete(_messages_key(sender))