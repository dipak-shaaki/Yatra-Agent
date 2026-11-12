"""
Conversation history per session, stored as a Redis list of JSON
{"role", "content"} messages under "session:{sender}:messages".

Session identity: clients call POST /api/v1/session to mint a token, and
the token itself is the sender id — the client never picks its own, so one
user cannot read or corrupt another's history. TTL is reset on every write,
so an active conversation never expires mid-session while an abandoned one
cleans itself up.

Also owns the per-session fixed-window rate limiter for /chat.
"""

import json
import secrets
import time

from app.components.redis.client import get_redis_client
from app.core.config import get_settings


def _messages_key(sender: str) -> str:
    return f"session:{sender}:messages"


def _token_key(token: str) -> str:
    return f"session_token:{token}"


def _rate_key(scope: str, window: int) -> str:
    return f"rate:{scope}:{window}"


def _fixed_window(scope: str, limit_per_minute: int) -> tuple[bool, int]:
    """Fixed-window limiter; returns (allowed, retry_after_seconds)."""
    settings = get_settings()
    client = get_redis_client()
    window = int(time.time() // 60)
    key = _rate_key(scope, window)
    count = client.incr(key)
    if count == 1:
        client.expire(key, settings.rate_limit_window_seconds)
    retry_after = 60 - int(time.time() % 60)
    return count <= limit_per_minute, retry_after


def create_session_token() -> str:
    """Mint a new session token (the server-side sender identity)."""
    settings = get_settings()
    token = secrets.token_urlsafe(24)
    get_redis_client().set(_token_key(token), "1", ex=settings.session_ttl_seconds)
    return token


def validate_session_token(token: str) -> bool:
    """True if the token is known. Refreshes the sliding TTL on use."""
    settings = get_settings()
    client = get_redis_client()
    if not client.get(_token_key(token)):
        return False
    client.expire(_token_key(token), settings.session_ttl_seconds)
    return True


def check_rate_limit(sender: str, limit_per_minute: int) -> tuple[bool, int]:
    """Per-token /chat limiter; returns (allowed, retry_after_seconds)."""
    return _fixed_window(f"chat:{sender}", limit_per_minute)


def check_session_mint_limit(client_ip: str) -> tuple[bool, int]:
    """Per-IP /session limiter: one caller cannot mint endless tokens and
    sidestep the per-token /chat limit."""
    return _fixed_window(
        f"session:{client_ip}", get_settings().session_rate_limit_per_minute
    )


def append_message(sender: str, role: str, content: str) -> None:
    """Append a message, trimming to the cap and resetting the TTL."""
    settings = get_settings()
    client = get_redis_client()
    key = _messages_key(sender)

    message = json.dumps({"role": role, "content": content})
    client.rpush(key, message)
    client.ltrim(key, -settings.max_stored_history_messages, -1)  # keep most recent N
    client.expire(key, settings.session_ttl_seconds)


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
