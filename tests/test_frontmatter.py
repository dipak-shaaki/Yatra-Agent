"""Frontmatter schema validation — bad documents must fail at ingestion."""

import pytest

from app.data_ingestion.frontmatter import ALLOWED_TAGS, PROVINCES, parse_frontmatter


def _doc(extra_frontmatter: str = "", **overrides: object) -> str:
    base = {
        "name": "Test Peak",
        "province": "Gandaki",
        "district": "Kaski",
        "tags": ["trek"],
        "difficulty": 2,
        "best_time_to_visit": ["Mar-May"],
        "avoided_season": ["Jun-Aug"],
        "route": "road",
        "distance_from_kathmandu": "~30 km",
        "time_required": "3 days",
        "days_min": 3,
        "days_max": 3,
        "max_altitude_m": 3000,
        "permits_required": [],
        "estimated_budget_npr": "NPR 2,000-4,000",
        "budget_min_npr": 2000,
        "budget_max_npr": 4000,
        "budget_per_day": False,
        "last_updated": "2026-09-11",
    }
    fields = {**base, **overrides}
    lines = ["---"]
    for key, value in fields.items():
        lines.append(
            f"{key}: {value!r}" if not isinstance(value, str) else f'{key}: "{value}"'
        )
    if extra_frontmatter:
        lines.append(extra_frontmatter)
    lines.append("---")
    lines.append("")
    lines.append("# Test Peak")
    lines.append("")
    lines.append("Intro.")
    return "\n".join(lines)


def test_valid_frontmatter_parses() -> None:
    frontmatter, body = parse_frontmatter(_doc())
    assert frontmatter.name == "Test Peak"
    assert frontmatter.difficulty == 2
    assert body == "# Test Peak\n\nIntro."


def test_unknown_province_rejected() -> None:
    document = _doc(province="Pokhara")
    with pytest.raises(ValueError, match="province"):
        parse_frontmatter(document)


def test_unknown_tag_rejected() -> None:
    document = _doc(tags=["beach"])
    with pytest.raises(ValueError, match="tag"):
        parse_frontmatter(document)


def test_unknown_field_rejected() -> None:
    document = _doc(extra_frontmatter="surprise: 1")
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_frontmatter(document)


def test_inverted_day_range_rejected() -> None:
    document = _doc(days_min=5, days_max=2)
    with pytest.raises(ValueError, match="days_min"):
        parse_frontmatter(document)


def test_difficulty_out_of_range_rejected() -> None:
    document = _doc(difficulty=9)
    with pytest.raises(ValueError, match="difficulty"):
        parse_frontmatter(document)


def test_vocabularies_nonempty() -> None:
    assert PROVINCES
    assert ALLOWED_TAGS
