"""Unit tests for the section-based chunker (pure logic, no external services)."""

from pathlib import Path

from app.data_ingestion.chunker import chunk_document

SAMPLE_DOC = """---
name: Test Peak
aliases: ["test peak trek"]
province: Gandaki
district: Kaski
tags: ["trek"]
difficulty: 2
best_time_to_visit: ["Mar-May", "Sep-Nov"]
avoided_season: ["Jun-Aug (monsoon)"]
route: "Short road from the valley floor"
distance_from_kathmandu: "~30 km"
time_required: "3 days"
days_min: 3
days_max: 3
max_altitude_m: null
permits_required: []
estimated_budget_npr: "NPR 2,000-4,000 per person"
budget_min_npr: 2000
budget_max_npr: 4000
budget_per_day: false
last_updated: "2026-09-11"
---

# Test Peak

Test Peak is a small mountain east of Pokhara.

## Route and Access

Reached by short road from the valley floor.

## Difficulty

Easy day hike.
"""


def test_chunk_document_captures_pre_header_intro_as_overview(tmp_path: Path) -> None:
    file_path = tmp_path / "test_peak.md"
    file_path.write_text(SAMPLE_DOC, encoding="utf-8")

    chunks = chunk_document(file_path)
    overview = [c for c in chunks if c["metadata"]["section_title"] == "Overview"]

    assert len(overview) == 1
    assert "Test Peak is a small mountain east of Pokhara." in overview[0]["text"]


def test_none_metadata_is_dropped_not_stringified(tmp_path: Path) -> None:
    file_path = tmp_path / "test_peak.md"
    file_path.write_text(SAMPLE_DOC, encoding="utf-8")

    chunks = chunk_document(file_path)

    for chunk in chunks:
        assert "max_altitude_m" not in chunk["metadata"]
        assert chunk["metadata"]["province"] == "Gandaki"


def test_each_section_becomes_a_chunk(tmp_path: Path) -> None:
    file_path = tmp_path / "test_peak.md"
    file_path.write_text(SAMPLE_DOC, encoding="utf-8")

    chunks = chunk_document(file_path)
    sections = [c["metadata"]["section_title"] for c in chunks]

    assert sections == ["Overview", "Route and Access", "Difficulty"]


def test_horizontal_rule_in_body_is_not_a_frontmatter_marker(tmp_path: Path) -> None:
    # Valid frontmatter from SAMPLE_DOC; the body's mid-document `---` rule
    # must not be mistaken for a frontmatter marker.
    frontmatter_block = SAMPLE_DOC.split("---")[1]  # between the two --- lines
    doc = (
        "---" + frontmatter_block + "---\n"
        "# Test Peak\n\nIntro paragraph.\n---\n"
        "A section below a horizontal rule.\n\n## Later Section\n\nStill parsed.\n"
    )
    file_path = tmp_path / "test_peak.md"
    file_path.write_text(doc, encoding="utf-8")

    chunks = chunk_document(file_path)

    sections = [c["metadata"]["section_title"] for c in chunks]
    assert "Later Section" in sections
    assert sections[0] == "Overview"


def test_document_without_frontmatter_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "broken.md"
    file_path.write_text("# No frontmatter here", encoding="utf-8")

    try:
        chunk_document(file_path)
    except ValueError:
        return
    raise AssertionError("chunk_document should have raised ValueError")
