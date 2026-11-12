"""
Shared corpus-version counter: the single signal other workers poll to know
the markdown corpus changed. Lives in Redis because workers are separate
processes; when Redis is down the readers fall back to "0" (serve the cached
snapshot rather than thrash) and log the failed bump.
"""

from app.components.redis.client import get_redis_client
from app.utils.logger import log_event

CORPUS_VERSION_KEY = "yatra:corpus_version"


def current_corpus_version() -> str:
    try:
        version = get_redis_client().get(CORPUS_VERSION_KEY)
        if isinstance(version, bytes):
            version = version.decode()
        return str(version) if version is not None else "0"
    except Exception:
        return "0"


def bump_corpus_version() -> None:
    """Signal every worker that the corpus changed (called on upsert)."""
    try:
        get_redis_client().incr(CORPUS_VERSION_KEY)
    except Exception as e:
        # shortcut: same-host workers share the corpus dir but keep their
        # in-memory snapshot until restart when Redis is down; upgrade to a
        # filesystem mtime check if that ever matters.
        log_event("corpus_version_bump_failed", error=str(e))
