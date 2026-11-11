"""Unit tests for the retrieval internals (fusion + BM25), no Chroma."""

from app.retrieval.bm25_index import BM25Index, _tokenize
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
