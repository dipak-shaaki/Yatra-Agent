"""Unit tests for the agent tools against the real corpus (no LLM, no Chroma)."""

from app.agent.tool_registry import call_tool, get_tool_schemas
from app.agent.tools.calculate_budget_estimate_tool import calculate_budget_estimate
from app.agent.tools.compare_destinations_tool import compare_destinations
from app.agent.tools.filter_by_criteria_tool import filter_by_criteria


def test_filter_by_province_is_case_insensitive() -> None:
    matches = filter_by_criteria(province="gandaki")
    names = {m["name"] for m in matches}
    assert "Manaslu Circuit" in names
    assert "Bandipur" in names
    assert all(m["province"] == "Gandaki" for m in matches)


def test_filter_by_easy_trek_finds_nearby_difficulty() -> None:
    # difficulty 1 (easy) matches 1 and 2 — Mardi Himal (2) must be included,
    # Manaslu Circuit (4) must not.
    matches = filter_by_criteria(difficulty_level=1, type="trek")
    names = {m["name"] for m in matches}
    assert "Mardi Himal" in names
    assert "Manaslu Circuit" not in names


def test_filter_by_plural_tag_tolerated() -> None:
    matches = filter_by_criteria(type="treks")
    assert matches
    assert all("trek" in m["tags"] for m in matches)


def test_filter_with_no_criteria_returns_empty() -> None:
    assert filter_by_criteria() == []


def test_budget_per_day_scales_arithmetically() -> None:
    # Bandipur is 3,000-6,000 NPR per day.
    result = calculate_budget_estimate("Bandipur", num_days=3)
    assert result["estimated_total_npr"] == [9000, 18000]


def test_budget_within_typical_range_uses_corpus_estimate() -> None:
    # Manaslu is 35k-60k for 12-18 days; 14 days is inside that range.
    result = calculate_budget_estimate("Manaslu Circuit", num_days=14)
    assert result["estimated_total_npr"] == [35000, 60000]


def test_budget_outside_range_scales_proportionally() -> None:
    # Manaslu: 35k-60k for 12-18 days. Implied daily rates are
    # 35k/18 (frugal) .. 60k/12 (comfort), so 24 days extrapolates to
    # ~46.7k-120k — above the corpus range but below a naive 24x multiply.
    result = calculate_budget_estimate("Manaslu Circuit", num_days=24)
    est_min, est_max = result["estimated_total_npr"]
    assert est_min > 35000
    assert est_max > 60000
    assert est_min < est_max
    assert est_min < 24 * 35000


def test_budget_without_days_returns_corpus_range() -> None:
    result = calculate_budget_estimate("rara")  # alias resolution
    assert result["destination"] == "Rara Lake"
    assert result["typical_budget_npr"] == [25000, 45000]


def test_budget_unknown_destination_and_bad_days_error() -> None:
    assert "error" in calculate_budget_estimate("Everest Base Camp")
    assert "error" in calculate_budget_estimate("Bandipur", num_days=0)


def test_compare_includes_shared_sections_only() -> None:
    comparison = compare_destinations(["Manaslu Circuit", "Rara Lake"])
    assert "Manaslu Circuit" in comparison
    assert "Rara Lake" in comparison
    assert "Permits" in comparison["Manaslu Circuit"]
    assert "Difficulty" in comparison["Rara Lake"]


def test_compare_reports_unresolved_names() -> None:
    comparison = compare_destinations(["abc", "Everest Base Camp"])
    assert "Annapurna Base Camp (ABC)" in comparison
    assert comparison["_unresolved"] == ["Everest Base Camp"]


def test_tool_schemas_carry_corpus_enums() -> None:
    schemas = {t["function"]["name"]: t["function"] for t in get_tool_schemas()}
    props = schemas["filter_by_criteria"]["parameters"]["properties"]
    assert "Gandaki" in props["province"]["enum"]
    assert props["difficulty_level"]["enum"] == sorted(
        props["difficulty_level"]["enum"]
    )
    assert "trek" in props["type"]["enum"]


def test_call_tool_dispatches_by_name() -> None:
    result = call_tool("filter_by_criteria", province="Lumbini")
    assert {m["name"] for m in result} == {"Tansen (Palpa)"}
