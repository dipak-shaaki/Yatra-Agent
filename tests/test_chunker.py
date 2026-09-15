"""Unit tests for the section-based chunker (pure logic, no external services)."""

from pathlib import Path

from app.data_ingestion.chunker import chunk_document

SAMPLE_DOC = """---
name: Test Peak
province: Gandaki
max_altitude_m: null
time_required: 3 days
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
    doc = """---
name: Test Peak
---
# Test Peak

Intro paragraph.
---
A section below a horizontal rule.

## Later Section

Still parsed.
"""
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
