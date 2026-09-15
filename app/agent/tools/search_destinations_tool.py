"""
Wraps hybrid_search() for single-destination or single-topic factual
questions; the tool the agent calls most often.
"""

from app.retrieval.hybrid_retriever import hybrid_search


def search_destinations(query: str, n_results: int = 5) -> list[dict]:
    """Return retrieved chunks as [{"destination", "section", "text"}] for
    the LLM's context — internal scores and ids are dropped on purpose."""
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
