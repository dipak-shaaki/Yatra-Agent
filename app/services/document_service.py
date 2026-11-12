"""
Corpus ingestion/update, ordered so a failure never leaves the corpus
half-written:

1. validate the frontmatter and build chunks in memory (no writes)
2. embed every chunk (the slow, failure-prone step)
3. write the markdown file atomically (temp file + os.replace)
4. upsert the new chunks, then delete only the ids that disappeared
5. invalidate in-process caches + bump the shared corpus version

Embedding before any write means a schema or model failure leaves both the
file and the index untouched. Upserting before deleting stale ids means
there is never a window where the destination has no chunks in Chroma.

BM25 has no incremental-update API, so any content change forces a full
rebuild — cheap (tokenization only, no embedding calls).
"""

import os
import tempfile
from pathlib import Path

from app.core.config import get_settings
from app.data_ingestion.chunker import chunk_text
from app.data_ingestion.corpus_meta import reset_corpus_meta
from app.data_ingestion.corpus_version import bump_corpus_version
from app.db.chroma.client import (
    delete_chunks,
    get_chunk_ids_by_source_file,
    upsert_chunks,
)
from app.retrieval.embeddings import embed_documents
from app.utils.logger import log_event


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def upsert_document(filename: str, content: str) -> dict:
    # chunk_text validates the full frontmatter schema — bad documents are
    # rejected before anything is written to disk, embedded, or indexed.
    source_file = f"{filename}.md"
    chunks = chunk_text(content, stem=filename, source_file=source_file)
    if not chunks:
        log_event(
            "document_upsert_rejected", filename=filename, reason="no chunks produced"
        )
        raise ValueError("Document produced zero chunks — check formatting")

    file_path = Path(get_settings().corpus_dir) / source_file
    is_update = file_path.exists()  # reported in the response/logs

    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(texts)

    _atomic_write(file_path, content)
    log_event(
        "document_written", filename=filename, bytes=len(content), is_update=is_update
    )

    new_ids = [c["id"] for c in chunks]
    old_ids = get_chunk_ids_by_source_file(source_file)
    upsert_chunks(
        ids=new_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )
    stale_ids = [chunk_id for chunk_id in old_ids if chunk_id not in set(new_ids)]
    deleted_count = delete_chunks(stale_ids)
    if deleted_count:
        log_event(
            "document_stale_chunks_deleted",
            filename=filename,
            deleted_count=deleted_count,
        )
    log_event("document_upserted", filename=filename, chunk_count=len(chunks))

    # Invalidate every in-process corpus view, then bump the shared corpus
    # version so other workers' caches and BM25 indexes rebuild on next use.
    reset_corpus_meta()
    bump_corpus_version()
    log_event("corpus_caches_invalidated", reason=f"document_update:{filename}")

    return {
        "filename": filename,
        "is_update": is_update,
        "old_chunks_deleted": deleted_count,
        "new_chunks_written": len(chunks),
        "chunk_ids": new_ids,
    }
