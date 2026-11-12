"""Unit tests for the retrieval internals (fusion + BM25), no Chroma."""

import app.retrieval.bm25_index as bm25_module
from app.retrieval.bm25_index import BM25Index, _tokenize, get_bm25_index
from app.retrieval.fusion import reciprocal_rank_fusion

CHUNKS = [
    {"id": "a__permits", "text": "Permits cost NPR 3,000 for ACAP"},
    {"id": "a__budget", "text": "Budget around NPR 12,000"},
    {"id": "b__permits", "text": "TIMS card required"},
]


def test_tokenizer_normalizes_comma_numbers() -> None:
    assert _tokenize("NPR 3,000") == ["npr", "3000"]


def test_bm25_ranks_keyword_match_first() -> None:
    index = BM25Index(CHUNKS)
    ranked = index.search("ACAP permits NPR 3,000", n_results=2)
    assert ranked[0][0] == "a__permits"
    assert len(ranked) == 2


def test_bm25_exposes_chunks_for_rehydration() -> None:
    index = BM25Index(CHUNKS)
    assert {c["id"] for c in index.chunks} == {"a__permits", "a__budget", "b__permits"}


def test_rrf_rewards_agreement_between_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    ids = [chunk_id for chunk_id, _ in fused]
    assert ids[0] == "b"  # present in both lists -> summed reciprocal ranks
    assert set(ids) == {"a", "b", "c"}


def test_rrf_scores_early_ranks_higher() -> None:
    fused = reciprocal_rank_fusion([["a", "b"]], k=60)
    assert fused[0][0] == "a"
    assert fused[0][1] > fused[1][1]


def test_empty_corpus_index_is_searchable() -> None:
    # BM25Okapi([]) raises ZeroDivisionError; the wrapper must not.
    index = BM25Index([])
    assert index.search("anything") == []


def test_bm25_index_rebuilds_after_version_bump(monkeypatch) -> None:
    """Regression: a document upsert bumps the shared version, and the next
    query in every worker must see the new chunks (the old bug decoded an
    already-decoded version, so the cache never invalidated)."""
    state = {"version": "1", "chunks": [CHUNKS[0]]}
    monkeypatch.setattr(bm25_module, "current_corpus_version", lambda: state["version"])
    monkeypatch.setattr(
        bm25_module, "get_corpus_chunks", lambda: tuple(state["chunks"])
    )
    bm25_module.reset_bm25_index()

    first = get_bm25_index()
    assert {c["id"] for c in first.chunks} == {"a__permits"}

    # Another worker handled the upsert: version moves, chunks change.
    state["version"] = "2"
    state["chunks"] = [CHUNKS[2]]

    second = get_bm25_index()
    assert {c["id"] for c in second.chunks} == {"b__permits"}
    assert second is not first

    # Same version -> cached instance, no rebuild.
    assert get_bm25_index() is second
