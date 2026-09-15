"""
Core lookup tool — wraps hybrid_search() for direct single-destination or
single-topic factual questions. This is the tool the agent calls most often.
"""

from app.retrieval.hybrid_retriever import hybrid_search


def search_destinations(query: str, n_results: int = 5) -> list[dict]:
    """
    Returns retrieved chunks as a list of dicts:
      [{"destination": ..., "section": ..., "text": ...}, ...]
    Simplified/flattened from hybrid_search()'s raw output, since this is
    what gets passed back into the LLM's context for answer generation —
    keep it lean, no internal scores/ids the LLM doesn't need.
    """
    results = hybrid_search(query, n_results=n_results)
    return [
        {
            "destination": r["metadata"].get(
                "name", r["metadata"].get("source_file", "unknown")
            ),
            "section": r["metadata"].get("section_title", ""),
            "text": r["text"],
        }
        for r in results
    ]
