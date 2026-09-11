"""
Ties dense (Chroma) + sparse (BM25) search together via RRF.
"""
from app.db.chroma.client import query_collection
from app.retrieval.bm25_index import get_bm25_index
from app.retrieval.embeddings import embed_query
from app.retrieval.fusion import reciprocal_rank_fusion


def hybrid_search(query: str, n_results: int = 5, candidate_pool: int = 10) -> list[dict]:
    """
    Returns the top n_results fused chunks as:
      [{"id": ..., "text": ..., "metadata": ..., "score": ...}, ...]
    candidate_pool controls how many results each individual method retrieves
    before fusion — should be >= n_results to give RRF enough to work with.
    """
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

    # Rehydrate full chunk data for the fused top results
    # (dense_results already has documents/metadatas for dense_ids; pull the rest from bm25_index.chunks)
    chunk_lookup = {c["id"]: c for c in bm25_index.chunks}

    results = []
    for chunk_id, score in fused:
        chunk = chunk_lookup.get(chunk_id)
        if chunk:
            results.append({
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": score,
            })
    return results