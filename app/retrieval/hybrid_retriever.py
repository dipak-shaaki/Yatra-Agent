"""
Ties dense (Chroma) + sparse (BM25) search together via RRF.
"""

from app.core.config import get_settings
from app.db.chroma.client import query_collection
from app.retrieval.bm25_index import get_bm25_index
from app.retrieval.embeddings import embed_query
from app.retrieval.fusion import reciprocal_rank_fusion
from app.utils.logger import log_event


def hybrid_search(
    query: str, n_results: int | None = None, candidate_pool: int | None = None
) -> list[dict]:
    """
    Returns the top n_results fused chunks as:
      [{"id": ..., "text": ..., "metadata": ..., "score": ...}, ...]
    candidate_pool controls how many results each individual method retrieves
    before fusion — should be >= n_results to give RRF enough to work with.
    """
    if n_results is None:
        n_results = get_settings().search_n_results
    if candidate_pool is None:
        candidate_pool = get_settings().candidate_pool

    # Dense search
    query_embedding = embed_query(query)
    dense_results = query_collection(query_embedding, n_results=candidate_pool)
    dense_ids = dense_results["ids"][0]

    # Sparse search
    bm25_index = get_bm25_index()
    sparse_hits = bm25_index.search(query, n_results=candidate_pool)
    sparse_ids = [chunk_id for chunk_id, _ in sparse_hits]

    # Fuse
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])[:n_results]

    # Rehydrate full chunk data for the fused top results.
    # dense_results carries documents/metadata for dense hits; BM25's in-memory
    # chunks cover the rest. If an id is in neither (e.g. BM25 serves a stale
    # cache after a corpus edit), fall back to the next dense result instead of
    # silently returning fewer chunks than asked for — and log the drift.
    chunk_lookup = {c["id"]: c for c in bm25_index.chunks}
    dense_by_id = {
        chunk_id: {
            "id": chunk_id,
            "text": doc,
            "metadata": meta,
        }
        for chunk_id, doc, meta in zip(
            dense_ids, dense_results["documents"][0], dense_results["metadatas"][0]
        )
    }

    results = []
    for chunk_id, score in fused:
        chunk = chunk_lookup.get(chunk_id) or dense_by_id.get(chunk_id)
        if chunk is None:
            log_event("hybrid_search_missing_chunk", chunk_id=chunk_id, source="fusion")
            continue
        results.append(
            {
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": score,
            }
        )
    # Top up from dense results if fusion lost chunks to lookup misses.
    if len(results) < n_results:
        seen = {r["id"] for r in results}
        for chunk_id, chunk in dense_by_id.items():
            if len(results) >= n_results:
                break
            if chunk_id not in seen:
                log_event("hybrid_search_backfilled_from_dense", chunk_id=chunk_id)
                results.append({**chunk, "score": 0.0})
    return results
