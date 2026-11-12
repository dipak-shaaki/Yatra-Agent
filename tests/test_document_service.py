"""document_service: validation, embed-before-write ordering, stale-chunk swap."""

import pytest

import app.services.document_service as doc_service
from app.core.config import get_settings
from tests.test_frontmatter import _doc


def _valid_doc() -> str:
    return (
        _doc()
        + "\n\n## Overview\n\nA short intro section.\n\n## Permits\n\nBring a TIMS card."
    )


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_invalid_document_is_rejected_before_any_write(corpus) -> None:
    with pytest.raises(ValueError):
        doc_service.upsert_document("bad", "not markdown at all")
    assert list(corpus.glob("*.md")) == []


def test_embed_failure_writes_nothing(corpus, monkeypatch) -> None:
    calls: list[str] = []

    def _fail_embed(texts):
        calls.append("embed")
        raise RuntimeError("model down")

    monkeypatch.setattr(doc_service, "embed_documents", _fail_embed)
    monkeypatch.setattr(
        doc_service, "upsert_chunks", lambda **kw: calls.append("upsert")
    )
    content = _valid_doc()

    with pytest.raises(RuntimeError, match="model down"):
        doc_service.upsert_document("test_peak", content)

    assert not (corpus / "test_peak.md").exists()
    assert calls == ["embed"]  # no index write


def test_swap_upserts_before_deleting_only_stale_ids(corpus, monkeypatch) -> None:
    events: list[tuple] = []
    monkeypatch.setattr(
        doc_service, "embed_documents", lambda texts: [[0.0] * 8 for _ in texts]
    )
    monkeypatch.setattr(
        doc_service,
        "get_chunk_ids_by_source_file",
        lambda source: ["test_peak__old_section", "test_peak__overview"],
    )
    monkeypatch.setattr(
        doc_service,
        "upsert_chunks",
        lambda **kw: events.append(("upsert", sorted(kw["ids"]))),
    )
    monkeypatch.setattr(
        doc_service,
        "delete_chunks",
        lambda ids: events.append(("delete", ids)) or len(ids),
    )
    monkeypatch.setattr(
        doc_service, "reset_corpus_meta", lambda: events.append(("reset",))
    )
    monkeypatch.setattr(
        doc_service, "bump_corpus_version", lambda: events.append(("bump",))
    )

    result = doc_service.upsert_document("test_peak", _valid_doc())

    written = (corpus / "test_peak.md").read_text(encoding="utf-8")
    assert written == _valid_doc()
    assert result["new_chunks_written"] == len(result["chunk_ids"])
    # New chunks are visible before the old ones disappear (no gap), and
    # only the id that no longer exists is deleted.
    assert [e[0] for e in events[:2]] == ["upsert", "delete"]
    assert events[1] == ("delete", ["test_peak__old_section"])
    assert events[2:] == [("reset",), ("bump",)]
