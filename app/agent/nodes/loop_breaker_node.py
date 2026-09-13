"""
Loop-breaker — if a session hits N consecutive filler turns, proactively
break the pattern instead of silently continuing to short-circuit forever.
"""
from app.components.redis.client import get_redis_client
from app.configs.agent_config import AGENT_MODEL, FILLER_LOOP_THRESHOLD

LOOP_BREAK_MESSAGE = "Just checking — is there something specific about Nepal travel I can help with, or are we good for now?"


def _filler_counter_key(sender: str) -> str:
    return f"session:{sender}:filler_count"


def increment_filler_count(sender: str) -> int:
    client = get_redis_client()
    key = _filler_counter_key(sender)
    count = client.incr(key)
    client.expire(key, 60 * 60 * 2)  # same TTL as session history
    return count


def reset_filler_count(sender: str) -> None:
    client = get_redis_client()
    client.delete(_filler_counter_key(sender))


def check_loop_break(sender: str) -> str | None:
    """Returns the loop-break message if threshold hit, else None."""
    client = get_redis_client()
    count = int(client.get(_filler_counter_key(sender)) or 0)
    if count >= FILLER_LOOP_THRESHOLD:
        reset_filler_count(sender)
        return LOOP_BREAK_MESSAGE
    return None