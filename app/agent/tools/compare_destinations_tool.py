from app.agent.tools.destination_resolver import resolve_destination_name
from app.db.chroma.client import get_collection

COMPARISON_SECTIONS = [
    "Overview",
    "Difficulty",
    "Duration",
    "Estimated Budget",
    "Best Time to Visit",
    "Permits",
]


def compare_destinations(destination_names: list[str]) -> dict:
    collection = get_collection()
    comparison = {}
    unresolved = []

    for raw_name in destination_names:
        resolved = resolve_destination_name(raw_name)
        if not resolved:
            unresolved.append(raw_name)
            continue

        results = collection.get(
            where={
                "$and": [
                    {"name": {"$eq": resolved}},
                    {"section_title": {"$in": COMPARISON_SECTIONS}},
                ]
            }
        )
        if not results["ids"]:
            continue

        sections = {}
        for text, metadata in zip(results["documents"], results["metadatas"]):
            sections[metadata.get("section_title", "Unknown")] = text
        comparison[resolved] = sections

    if unresolved:
        comparison["_unresolved"] = (
            unresolved  # lets the agent tell the user which names it couldn't find
        )

    return comparison
