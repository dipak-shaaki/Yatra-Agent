"""
In-memory BM25 sparse index over the corpus chunks.

Unlike Chroma, BM25 has no natural persistence/upsert API — rebuilding it
from the full chunk set is cheap (just tokenization, no embedding calls),
so we rebuild in memory on load rather than persisting to disk.
"""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.data_ingestion.chunker import chunk_corpus_dir

TOKEN_RE = re.compile(r"[a-z0-9]+")

COMMA_IN_NUMBER_RE = re.compile(r"(?<=\d),(?=\d)")


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


@lru_cache
def get_bm25_index(corpus_dir: str | None = None) -> BM25Index:
    """Cached singleton — rebuild by clearing the cache (see reset below)."""
    if corpus_dir is None:
        corpus_dir = get_settings().corpus_dir
    chunks = chunk_corpus_dir(corpus_dir)
    return BM25Index(chunks)


def reset_bm25_index() -> None:
    """Call after corpus updates so the next get_bm25_index() rebuilds from disk."""
    get_bm25_index.cache_clear()
