"""
Metadata-filtered discovery for queries like "easy treks near Pokhara" or
"destinations in Gandaki province" that expect a set of matches. Filters on
frontmatter metadata (province, difficulty_level, type), which embeddings
tend to miss or blur.
"""

from app.db.chroma.client import get_collection


def filter_by_criteria(
    province: str | None = None,
    difficulty_level: str | None = None,
    type: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Return destinations matching the given metadata filters, deduplicated
    by name (one destination can match via several section chunks).

    All params are optional; with none set there is nothing to filter on.
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
        where=where,
        limit=max_results * 5,  # over-fetch, then dedupe
    )

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
