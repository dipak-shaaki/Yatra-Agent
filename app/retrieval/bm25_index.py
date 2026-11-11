"""
In-memory BM25 sparse index over the corpus chunks.

Unlike Chroma, BM25 has no natural persistence/upsert API — rebuilding it
from the full chunk set is cheap (just tokenization, no embedding calls),
so we rebuild in memory rather than persisting to disk.

Multi-worker safe: the index is cached against (corpus_dir, corpus_version),
where the version lives in Redis and is bumped by every document upsert. A
worker that sees a new version rebuilds automatically — no more silently
stale index in the processes that did not handle the upsert.
"""

import re
import threading

from rank_bm25 import BM25Okapi

from app.components.redis.client import get_redis_client
from app.core.config import get_settings
from app.data_ingestion.chunker import chunk_corpus_dir
from app.utils.logger import log_event

TOKEN_RE = re.compile(r"[a-z0-9]+")

COMMA_IN_NUMBER_RE = re.compile(r"(?<=\d),(?=\d)")

CORPUS_VERSION_KEY = "yatra:corpus_version"


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer, with comma-in-number normalization."""
    text = COMMA_IN_NUMBER_RE.sub("", text.lower())
    return TOKEN_RE.findall(text)


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.ids = [c["id"] for c in chunks]
        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, n_results: int = 5) -> list[tuple[str, float]]:
        """Returns [(chunk_id, score), ...] sorted by score descending."""
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:n_results]


def bump_corpus_version() -> None:
    """Signal every worker that the corpus changed (called on upsert)."""
    try:
        get_redis_client().incr(CORPUS_VERSION_KEY)
    except Exception as e:
        # shortcut: same-host workers share the corpus dir but keep their
        # in-memory index until restart when Redis is down; upgrade to a
        # filesystem mtime check if that ever matters.
        log_event("corpus_version_bump_failed", error=str(e))


def _current_corpus_version() -> str:
    try:
        version = get_redis_client().get(CORPUS_VERSION_KEY)
        return version.decode() if version else "0"
    except Exception:
        return "0"  # Redis unavailable — serve the cached index, don't thrash


_cache_lock = threading.Lock()
_cached: tuple[str, str, BM25Index] | None = None  # (corpus_dir, version, index)


def get_bm25_index() -> BM25Index:
    """Cached singleton, rebuilt when the corpus dir or shared version changes."""
    global _cached
    corpus_dir = get_settings().corpus_dir
    version = _current_corpus_version()
    with _cache_lock:
        if _cached is not None and _cached[:2] == (corpus_dir, version):
            return _cached[2]
        index = BM25Index(chunk_corpus_dir(corpus_dir))
        _cached = (corpus_dir, version, index)
        log_event("bm25_index_built", corpus_dir=corpus_dir, version=version)
        return index


def reset_bm25_index() -> None:
    """Drop this process's cached index (mainly for tests)."""
    global _cached
    with _cache_lock:
        _cached = None
