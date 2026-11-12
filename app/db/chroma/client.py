"""
Chroma persistent client + collection access. Pure connection/storage setup —
no business logic, no retrieval ranking (that lives in app/retrieval/).
"""

from functools import lru_cache
from typing import Any

import chromadb

from app.core.config import get_settings
from app.utils.logger import log_event


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection():
    settings = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": settings.embedding_model,
        },
    )


def verify_embedding_compat(dimension: int) -> None:
    """Fail fast when the configured embedding model does not match the one
    that built the collection — otherwise queries silently return garbage.

    Chroma stores whatever metadata the collection was created with, so old
    collections get theirs backfilled here after the first dimension check.
    """
    settings = get_settings()
    collection = get_collection()
    metadata = dict(collection.metadata or {})
    stored_model = metadata.get("embedding_model")
    stored_dim = metadata.get("embedding_dimension")

    if stored_model and stored_model != settings.embedding_model:
        raise RuntimeError(
            f"Embedding model mismatch: collection '{settings.collection_name}' "
            f"was built with '{stored_model}', settings say "
            f"'{settings.embedding_model}'. Rebuild the index or revert the setting."
        )

    if stored_dim is None and collection.count() > 0:
        # Older collections predate the metadata: read one stored vector.
        sample = collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        if embeddings is not None and len(embeddings):
            stored_dim = len(embeddings[0])

    if stored_dim is not None and int(stored_dim) != dimension:
        raise RuntimeError(
            f"Embedding dimension mismatch: collection has {stored_dim}-dim "
            f"vectors, '{settings.embedding_model}' produces {dimension}. "
            "Rebuild the index or revert the embedding model."
        )

    if (
        metadata.get("embedding_model") != settings.embedding_model
        or metadata.get("embedding_dimension") != dimension
    ):
        # Chroma rejects `hnsw:space` in modify() ("changing the distance
        # function... not supported") and the distance now lives in the
        # collection configuration, so drop legacy hnsw:* keys from the
        # metadata we write back.
        metadata = {k: v for k, v in metadata.items() if not k.startswith("hnsw:")}
        metadata.update(
            {
                "embedding_model": settings.embedding_model,
                "embedding_dimension": dimension,
            }
        )
        collection.modify(metadata=metadata)
        log_event(
            "chroma_embedding_metadata_updated",
            embedding_model=settings.embedding_model,
            embedding_dimension=dimension,
        )


def get_chunk_ids_by_source_file(source_file: str) -> list[str]:
    """Chunk ids currently stored for one source file."""
    collection = get_collection()
    existing = collection.get(where={"source_file": {"$eq": source_file}})
    return list(existing["ids"])


def delete_chunks_by_source_file(source_file: str) -> int:
    """Deletes all chunks belonging to one source file (e.g. 'bandipur.md')."""
    collection = get_collection()
    existing = collection.get(where={"source_file": {"$eq": source_file}})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    return len(existing["ids"])


def delete_chunks(ids: list[str]) -> int:
    """Delete specific chunk ids (used for the post-upsert stale-chunk swap)."""
    if not ids:
        return 0
    get_collection().delete(ids=ids)
    return len(ids)


def upsert_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    collection = get_collection()
    collection.upsert(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )


def query_collection(
    query_embedding: list[float],
    n_results: int | None = None,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results or get_settings().search_n_results,
        where=where,
    )


def reset_collection() -> None:
    """Drop and recreate — useful when re-running build_index.py during development."""
    client = get_chroma_client()
    try:
        client.delete_collection(get_settings().collection_name)
    except Exception as e:
        log_event("chroma_reset_collection_not_found", detail=str(e))
