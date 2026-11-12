"""Corpus metadata: name resolution from frontmatter aliases, controlled
vocabularies, and the schema-derived comparison sections."""

from app.data_ingestion import corpus_meta
from app.data_ingestion.corpus_meta import (
    comparison_sections,
    destination_names,
    get_corpus_chunks,
    get_corpus_snapshot,
    get_destinations,
    resolve_destination,
)


def test_corpus_loads_all_destinations() -> None:
    destinations = get_destinations()
    assert len(destinations) == 10
    assert len(destination_names()) == len(destinations)


def test_every_destination_has_numeric_budget_and_days() -> None:
    for dest in get_destinations():
        assert 0 < dest.budget_min_npr <= dest.budget_max_npr
        assert 0 < dest.days_min <= dest.days_max
        assert 1 <= dest.difficulty <= 5
        assert dest.tags


def test_aliases_resolve_from_frontmatter() -> None:
    assert resolve_destination("abc") == "Annapurna Base Camp (ABC)"
    assert resolve_destination("Kori Danda") == "Kori (Kori Danda)"
    assert resolve_destination("palpa") == "Tansen (Palpa)"
    assert resolve_destination("manaslu") == "Manaslu Circuit"
    assert resolve_destination("Rara Lake Trek") == "Rara Lake"


def test_every_canonical_name_resolves_to_itself() -> None:
    for name in destination_names():
        assert resolve_destination(name) == name


def test_unknown_name_returns_none() -> None:
    assert resolve_destination("Everest Base Camp") is None
    assert resolve_destination("pokhara") is None


def test_comparison_sections_derived_from_corpus() -> None:
    sections = comparison_sections()
    assert "Difficulty" in sections
    assert "Estimated Budget" in sections
    # Sections present in fewer than half the documents are excluded.
    assert "Things to Do" not in sections


def test_corpus_chunks_have_section_and_source_metadata() -> None:
    chunks = get_corpus_chunks()
    assert chunks
    for chunk in chunks:
        assert chunk["metadata"]["source_file"].endswith(".md")
        assert chunk["metadata"]["section_title"]


def test_snapshot_rebuilds_when_shared_version_changes(monkeypatch) -> None:
    state = {"version": "1"}
    loaded: list[str] = []

    def _fake_load(corpus_dir: str, version: str) -> corpus_meta.CorpusSnapshot:
        loaded.append(version)
        return corpus_meta.CorpusSnapshot(version=version, destinations=(), chunks=())

    monkeypatch.setattr(corpus_meta, "current_corpus_version", lambda: state["version"])
    monkeypatch.setattr(corpus_meta, "_load_snapshot", _fake_load)
    corpus_meta.reset_corpus_meta()
    try:
        first = get_corpus_snapshot()
        assert get_corpus_snapshot() is first
        assert loaded == ["1"]  # cached within a version

        state["version"] = "2"  # another worker upserted
        assert get_corpus_snapshot() is not first
        assert loaded == ["1", "2"]
    finally:
        corpus_meta.reset_corpus_meta()
