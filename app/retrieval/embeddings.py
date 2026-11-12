"""
Wraps the self-hosted embedding model (bge-base-en-v1.5) so the rest of the
app never touches sentence-transformers directly.

The sentence-transformers import is deliberately lazy and only happens on
first embed — importing it eagerly would force a model-download check at
import time for code paths that never embed anything (tests, classifiers).
"""

from functools import lru_cache
from threading import BoundedSemaphore

from app.core.config import get_settings

# bge models recommend a query-side instruction prefix for retrieval tasks
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Caps concurrent encode() calls; the model is CPU-bound and shared.
_embed_semaphore: BoundedSemaphore | None = None


def _embedding_lock() -> BoundedSemaphore:
    global _embed_semaphore
    if _embed_semaphore is None:
        _embed_semaphore = BoundedSemaphore(get_settings().max_concurrent_embeddings)
    return _embed_semaphore


@lru_cache
def get_embedding_model():
    """Load once, reuse across the app (model load is slow, ~1-2s).

    HF_HUB_OFFLINE is only set when the deployment opts in: it must be in the
    environment before sentence-transformers reads it, and forcing it on by
    default breaks first-run model downloads.
    """
    import os

    settings = get_settings()
    if settings.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus chunks for storage. No prefix — bge only prefixes queries."""
    model = get_embedding_model()
    with _embedding_lock():
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user query with the bge retrieval prefix."""
    model = get_embedding_model()
    with _embedding_lock():
        embedding = model.encode(
            BGE_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
        )
    return embedding.tolist()
