"""
Shared pytest fixtures.

Redis: tests use a real server when one is reachable (CI runs a redis
service); otherwise they transparently fall back to an in-process fakeredis,
so `pytest` works on a laptop with nothing running.
"""

import fakeredis
import pytest

from app.components.redis.client import get_redis_client


def _real_redis_available() -> bool:
    try:
        get_redis_client().ping()
        return True
    except Exception:
        return False


def _clear_rate_limits(client) -> None:
    """Each test starts with an empty fixed window."""
    for key in client.scan_iter("rate:*"):
        client.delete(key)


@pytest.fixture(autouse=True)
def redis_backend(monkeypatch):
    """Provide Redis for every test, real if available else fakeredis.

    Modules import `get_redis_client` by value, so the fake is patched into
    each module's namespace that uses it.
    """
    if _real_redis_available():
        client = get_redis_client()
        _clear_rate_limits(client)
        yield
        _clear_rate_limits(client)
        return

    fake = fakeredis.FakeRedis(decode_responses=True)
    for module in (
        "app.components.redis.client",
        "app.components.redis.session_store",
        "app.agent.nodes.loop_breaker_node",
        "app.data_ingestion.corpus_version",
    ):
        monkeypatch.setattr(f"{module}.get_redis_client", lambda: fake)
    _clear_rate_limits(fake)
    yield
    fake.flushall()
