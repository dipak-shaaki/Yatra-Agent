"""
Chroma persistent client + collection access. Pure connection/storage setup —
no business logic, no retrieval ranking (that lives in app/retrieval/).
"""

from functools import lru_cache
from typing import Any

import chromadb

from app.core.config import get_settings
from app.utils.logger import log_event

COLLECTION_NAME = "yatra_destinations"


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def delete_chunks_by_source_file(source_file: str) -> int:
    """Deletes all chunks belonging to one source file (e.g. 'bandipur.md'),
    used before re-upserting a destination so removed/renamed sections don't
    leave stale, orphaned chunks behind."""
    collection = get_collection()
    existing = collection.get(where={"source_file": {"$eq": source_file}})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    return len(existing["ids"])


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
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )


def reset_collection() -> None:
    """Drop and recreate — useful when re-running build_index.py during development."""
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        log_event("chroma_reset_collection_not_found", detail=str(e))
