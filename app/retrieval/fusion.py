"""
Reciprocal Rank Fusion combines a dense ranking and a sparse ranking into
one ordered result set, without needing to normalize/compare raw scores
from two different scales (cosine distance vs. BM25 score).
"""

RRF_K = 60  # standard constant from the original RRF paper; dampens the impact of high ranks


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    ranked_lists: list of ranked id lists, e.g. [dense_ids, sparse_ids],
                  each already sorted best-to-worst.
    Returns [(chunk_id, fused_score), ...] sorted by fused_score descending.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)