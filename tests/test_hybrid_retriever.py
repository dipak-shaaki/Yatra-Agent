"""Hybrid retrieval against a real (temporary) Chroma collection, using a
deterministic fake embedder so torch/sentence-transformers never load."""

import pytest

from app.core.config import get_settings
from app.data_ingestion.corpus_meta import get_corpus_chunks
from app.db.chroma.client import get_chroma_client, reset_collection, upsert_chunks
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
