"""Hybrid retrieval against a real (temporary) Chroma collection, using a
deterministic fake embedder so torch/sentence-transformers never load."""

import pytest

from app.core.config import get_settings
from app.data_ingestion.corpus_meta import get_corpus_chunks
from app.db.chroma.client import (
    get_chroma_client,
    get_collection,
    reset_collection,
    upsert_chunks,
    verify_embedding_compat,
)
from app.retrieval import hybrid_retriever
from app.retrieval.bm25_index import reset_bm25_index
from tests.fake_embeddings import fake_embed_documents, fake_embedding


@pytest.fixture
def temp_chroma(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    get_chroma_client.cache_clear()
    reset_bm25_index()
    yield
    get_chroma_client.cache_clear()
    get_settings.cache_clear()
    reset_bm25_index()


def test_hybrid_search_returns_fused_chunks_from_real_chroma(
    temp_chroma, monkeypatch
) -> None:
    chunks = list(get_corpus_chunks())
    assert chunks, "corpus should not be empty"

    reset_collection()
    upsert_chunks(
        ids=[c["id"] for c in chunks],
        embeddings=fake_embed_documents([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    monkeypatch.setattr(hybrid_retriever, "embed_query", fake_embedding)

    results = hybrid_retriever.hybrid_search(
        "Manaslu circuit permits altitude", n_results=5
    )

    assert len(results) == 5
    assert len({r["id"] for r in results}) == 5
    for result in results:
        assert {"id", "text", "metadata", "score"} <= set(result)
        assert result["text"]
        assert result["metadata"]["source_file"].endswith(".md")


def test_verify_embedding_compat_backfills_legacy_metadata(temp_chroma) -> None:
    """Regression: a collection created before dimension metadata (and with
    legacy hnsw:space metadata) must be backfilled without Chroma rejecting
    the modify() call — this crashed app startup."""
    reset_collection()
    get_collection()

    verify_embedding_compat(768)

    metadata = get_collection().metadata or {}
    assert metadata["embedding_dimension"] == 768
    assert metadata["embedding_model"] == get_settings().embedding_model
    assert metadata["embedding_revision"] == get_settings().embedding_revision

    verify_embedding_compat(768)  # already backfilled -> no modify, no raise


def test_verify_embedding_compat_rejects_revision_mismatch(
    temp_chroma, monkeypatch
) -> None:
    """Same model name but different weights must fail fast: name + dimension
    alone cannot detect an upstream retag, and the vectors would be garbage."""
    reset_collection()
    get_collection()

    monkeypatch.setenv("EMBEDDING_REVISION", "deadbeef")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="Embedding revision mismatch"):
        verify_embedding_compat(768)


def test_verify_embedding_compat_rejects_model_mismatch(
    temp_chroma, monkeypatch
) -> None:
    reset_collection()
    get_collection()

    monkeypatch.setenv("EMBEDDING_MODEL", "some/other-model")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="Embedding model mismatch"):
        verify_embedding_compat(768)


@pytest.mark.usefixtures("temp_chroma")
def test_hybrid_search_handles_empty_collection(monkeypatch) -> None:
    reset_collection()
    monkeypatch.setattr(hybrid_retriever, "embed_query", fake_embedding)
    monkeypatch.setattr(hybrid_retriever, "get_bm25_index", lambda: _EmptyIndex())

    assert hybrid_retriever.hybrid_search("anything", n_results=5) == []


class _EmptyIndex:
    def __init__(self):
        self.chunks: list = []

    def search(self, query, n_results=None):
        return []
