"""
Metadata-filtered discovery for queries like "easy treks near Pokhara" or
"destinations in Gandaki province" that expect a set of matches. Filters on
validated frontmatter metadata (province, difficulty, tags), which
embeddings tend to miss or blur.

Matching is tolerant on purpose: difficulty matches adjacent levels (the
corpus's 1-5 scale is coarse), and tags come from the same controlled
vocabulary the tool schema advertises, so "trek" matches ["lake", "trek"].
"""

from app.data_ingestion.corpus_meta import Destination, get_destinations


def _destination_card(dest: Destination) -> dict:
    return {
        "name": dest.name,
        "province": dest.province,
        "difficulty": dest.difficulty,
        "tags": list(dest.tags),
        "typical_days": [dest.days_min, dest.days_max],
        "budget_npr": [dest.budget_min_npr, dest.budget_max_npr],
        "budget_per_day": dest.budget_per_day,
        "time_required": dest.time_required,
        "estimated_budget_npr": dest.estimated_budget_npr,
        "last_updated": dest.last_updated,
    }


def filter_by_criteria(
    province: str | None = None,
    difficulty_level: int | None = None,
    type: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Return destinations matching the given filters, one card per
    destination.

    All params are optional; with none set there is nothing to filter on.
    """
    if not any([province, difficulty_level is not None, type]):
        return []

    province_l = province.strip().lower() if province else None
    # Tolerate plurals the model may send despite the enum ("treks" -> "trek").
    type_l = type.strip().lower().rstrip("s") if type else None

    matches: list[dict] = []
    for dest in get_destinations():
        if province_l and dest.province.lower() != province_l:
            continue
        if difficulty_level is not None and abs(dest.difficulty - difficulty_level) > 1:
            continue
        if type_l and type_l not in {tag.lower() for tag in dest.tags}:
            continue
        matches.append(_destination_card(dest))
        if len(matches) >= max_results:
            break
    return matches
