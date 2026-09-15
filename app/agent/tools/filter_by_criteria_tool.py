"""
Metadata-filtered discovery tool — for queries like "easy treks near Pokhara"
or "destinations in Gandaki province" where the user wants a *set* of matching
destinations, not a single factual lookup. Filters on frontmatter metadata
(province, difficulty_level, type) rather than relying on semantic search alone,
since these are categorical facts embeddings can miss or blur.
"""

from app.db.chroma.client import get_collection


def filter_by_criteria(
    province: str | None = None,
    difficulty_level: str | None = None,
    type: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    """
    Returns destinations matching the given metadata filters, deduplicated
    by destination name (since a match can come from multiple section chunks
    of the same destination).

    All params are optional — pass only the ones the user specified. At least
    one should be provided or this just returns arbitrary chunks.
    """
    collection = get_collection()

    where_conditions = []
    if province:
        where_conditions.append({"province": {"$eq": province}})
    if difficulty_level:
        where_conditions.append({"difficulty_level": {"$eq": difficulty_level}})
    if type:
        where_conditions.append({"type": {"$eq": type}})

    if not where_conditions:
        return []

    where = (
        where_conditions[0]
        if len(where_conditions) == 1
        else {"$and": where_conditions}
    )

    results = collection.get(
        where=where, limit=max_results * 5
    )  # over-fetch, we'll dedupe

    seen_destinations = {}
    for metadata in results["metadatas"]:
        name = metadata.get("name")
        if name and name not in seen_destinations:
            seen_destinations[name] = {
                "name": name,
                "province": metadata.get("province"),
                "difficulty_level": metadata.get("difficulty_level"),
                "type": metadata.get("type"),
                "time_required": metadata.get("time_required"),
                "estimated_budget_npr": metadata.get("estimated_budget_npr"),
            }
        if len(seen_destinations) >= max_results:
            break

    return list(seen_destinations.values())
