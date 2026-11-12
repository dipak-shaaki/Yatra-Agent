"""
In-memory BM25 sparse index over the corpus chunks.

Unlike Chroma, BM25 has no natural persistence/upsert API — rebuilding it
from the full chunk set is cheap (just tokenization, no embedding calls),
so we rebuild in memory rather than persisting to disk.

Multi-worker safe: the index is cached against (corpus_dir, corpus_version),
where the version lives in Redis and is bumped by every document upsert. A
worker that sees a new version rebuilds automatically — no more silently
stale index in the processes that did not handle the upsert. The chunks come
from `corpus_meta.get_corpus_chunks()`, so BM25 and the tools read the same
versioned snapshot of the markdown corpus.
"""

import re
import threading

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.data_ingestion.corpus_meta import get_corpus_chunks
from app.data_ingestion.corpus_version import (  # noqa: F401 — re-exported
    bump_corpus_version,
    current_corpus_version,
)
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
        # BM25Okapi divides by the corpus size (avgdl), so an empty corpus
        # raises ZeroDivisionError. Keep an empty index searchable instead.
        self.bm25 = (
            BM25Okapi([_tokenize(c["text"]) for c in chunks]) if chunks else None
        )

    def search(
        self, query: str, n_results: int | None = None
    ) -> list[tuple[str, float]]:
        """Returns [(chunk_id, score), ...] sorted by score descending."""
        if self.bm25 is None:
            return []
        n_results = n_results or get_settings().search_n_results
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:n_results]


_cache_lock = threading.Lock()
_cached: tuple[str, str, BM25Index] | None = None  # (corpus_dir, version, index)


def get_bm25_index() -> BM25Index:
    """Cached singleton, rebuilt when the corpus dir or shared version changes."""
    global _cached
    corpus_dir = get_settings().corpus_dir
    version = current_corpus_version()
    with _cache_lock:
        if _cached is not None and _cached[:2] == (corpus_dir, version):
            return _cached[2]
        index = BM25Index(list(get_corpus_chunks()))
        _cached = (corpus_dir, version, index)
        log_event("bm25_index_built", corpus_dir=corpus_dir, version=version)
        return index


def reset_bm25_index() -> None:
    """Drop this process's cached index (mainly for tests)."""
    global _cached
    with _cache_lock:
        _cached = None
