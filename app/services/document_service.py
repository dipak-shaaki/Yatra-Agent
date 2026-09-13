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

CORPUS_DIR = Path("data/corpus")


def upsert_document(filename: str, content: str) -> dict:
    """
    Writes/overwrites one destination's markdown file, re-chunks it,
    re-embeds and upserts into Chroma, and invalidates the BM25 cache.

    Returns a summary dict with counts, so the caller can confirm what
    actually happened rather than just trusting a 200 response.
    """
    file_path = CORPUS_DIR / f"{filename}.md"
    file_path.write_text(content, encoding="utf-8")
    log_event("document_written", filename=filename, bytes=len(content))

    chunks = chunk_document(file_path)
    if not chunks:
        log_event("document_upsert_warning", filename=filename, reason="no chunks produced")
        return {"filename": filename, "chunks_written": 0, "warning": "Document produced zero chunks — check formatting"}

    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(texts)

    upsert_chunks(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )
    log_event("document_upserted", filename=filename, chunk_count=len(chunks))

    # BM25 has no incremental update — invalidate so the next search
    # rebuilds from the full corpus (including this updated file).
    reset_bm25_index()
    log_event("bm25_index_invalidated", reason=f"document_update:{filename}")

    return {
        "filename": filename,
        "chunks_written": len(chunks),
        "chunk_ids": [c["id"] for c in chunks],
    }