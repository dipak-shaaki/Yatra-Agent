# ADR 0001: Chroma as the vector store

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

The corpus is 10 destinations / 82 chunks. Retrieval is embedded in a single
FastAPI service, and the same markdown corpus also feeds BM25 and the
metadata tools. The store must persist locally (a checkout should work after
`python -m scripts.build_index`) and support metadata filters for tools.

## Decision

Use Chroma in persistent local mode (`data/chroma_db`), with the embedding
model, pinned revision (commit SHA) and dimension recorded in the collection
metadata and verified at startup.

## Consequences

- No extra service to run, no Docker-only path; tests can point
  `CHROMA_PERSIST_DIR` at a temp directory and use a fake embedder.
- Sufficient at this scale; a larger corpus or multi-writer ingestion would
  push toward a server-mode store with real concurrent-write guarantees.
- Chroma has no transaction with the filesystem, so the upsert path is
  ordered (validate → embed → atomic file write → upsert → delete stale IDs)
  rather than relying on a rollback.
