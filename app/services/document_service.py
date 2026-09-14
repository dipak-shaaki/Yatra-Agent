"""
Handles document ingestion/updates: writes corpus markdown to disk,
re-chunks just that file, re-embeds its chunks, upserts into Chroma
(only touches this destination's chunk IDs), and resets BM25 so it
rebuilds from the full corpus on next use.

BM25 has no incremental-update API (unlike Chroma's upsert), so a full
rebuild is unavoidable on any content change — cheap (tokenization only,
no embedding calls) but worth being explicit about, not hidden.
"""
from pathlib import Path

from app.data_ingestion.chunker import chunk_document
from app.db.chroma.client import upsert_chunks
from app.retrieval.bm25_index import reset_bm25_index
from app.retrieval.embeddings import embed_documents
from app.utils.logger import log_event
from app.data_ingestion.chunker import chunk_document, _parse_frontmatter
from app.db.chroma.client import upsert_chunks, delete_chunks_by_source_file

CORPUS_DIR = Path("data/corpus")

def upsert_document(filename: str, content: str) -> dict:
    try:
        frontmatter, _ = _parse_frontmatter(content)
    except ValueError as e:
        raise ValueError(f"Invalid document format, not written: {e}")

    if not frontmatter.get("name"):
        raise ValueError("Document frontmatter must include a 'name' field, not written")

    file_path = CORPUS_DIR / f"{filename}.md"
    is_update = file_path.exists()  # distinguish create vs. update for logging/response clarity

    file_path.write_text(content, encoding="utf-8")
    log_event("document_written", filename=filename, bytes=len(content), is_update=is_update)

    chunks = chunk_document(file_path)
    if not chunks:
        log_event("document_upsert_warning", filename=filename, reason="no chunks produced")
        return {"filename": filename, "chunks_written": 0, "warning": "Document produced zero chunks — check formatting"}

    # Delete ALL existing chunks for this file first — not just overwrite
    # matching IDs — so a removed or renamed section doesn't leave a stale,
    # orphaned chunk behind in Chroma.
    source_file = file_path.name
    deleted_count = delete_chunks_by_source_file(source_file)
    if deleted_count:
        log_event("document_old_chunks_deleted", filename=filename, deleted_count=deleted_count)

    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(texts)

    upsert_chunks(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )
    log_event("document_upserted", filename=filename, chunk_count=len(chunks))

    reset_bm25_index()
    log_event("bm25_index_invalidated", reason=f"document_update:{filename}")

    return {
        "filename": filename,
        "is_update": is_update,
        "old_chunks_deleted": deleted_count,
        "new_chunks_written": len(chunks),
        "chunk_ids": [c["id"] for c in chunks],
    }