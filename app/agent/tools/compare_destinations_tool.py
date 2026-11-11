"""
Side-by-side comparison across the schema-derived comparison sections
(see corpus_meta.comparison_sections). Reads full section text from the
corpus chunks directly, so it cannot drift from a stale vector index.
"""

from app.data_ingestion.corpus_meta import (
    comparison_sections,
    get_corpus_chunks,
    resolve_destination,
)


def compare_destinations(destination_names: list[str]) -> dict:
    sections_of_interest = set(comparison_sections())
    by_destination: dict[str, dict[str, str]] = {}
    for chunk in get_corpus_chunks():
        metadata = chunk["metadata"]
        name = metadata.get("name")
        section = metadata.get("section_title")
        if section in sections_of_interest:
            by_destination.setdefault(name, {})[section] = chunk["text"]

    comparison: dict = {}
    unresolved = []

    for raw_name in destination_names:
        resolved = resolve_destination(raw_name)
        if not resolved:
            unresolved.append(raw_name)
            continue
        sections = by_destination.get(resolved)
        if sections:
            comparison[resolved] = sections

    if unresolved:
        comparison["_unresolved"] = (
            unresolved  # lets the agent tell the user which names it couldn't find
        )

    return comparison
