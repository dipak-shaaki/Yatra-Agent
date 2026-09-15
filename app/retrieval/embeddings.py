"""
Wraps the self-hosted embedding model (bge-base-en-v1.5) so the rest of the
app never touches sentence-transformers directly.
"""

import os
from functools import lru_cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

# bge models recommend a query-side instruction prefix for retrieval tasks
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Load once, reuse across the app (model load is slow, ~1-2s)."""
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus chunks for storage. No prefix — bge only prefixes queries."""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user query for search — uses bge's recommended query prefix."""
    model = get_embedding_model()
    embedding = model.encode(
        BGE_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
    )
    return embedding.tolist()
