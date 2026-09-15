# Yatra — Architecture & Engineering Reference

This document describes how the Yatra codebase actually works: the request
path, the modules involved at each step, the configuration surface, the data
model, and the operational concerns. It is a companion to the feature-focused
`README.md`; prefer this file when you need to know *where* something lives
and *why* it is shaped the way it is.

---

## 1. System Overview

Yatra is an agentic retrieval-augmented-generation (RAG) assistant for Nepal
domestic tourism. A user question is processed by a **Turn Handler**, which
decides between zero-, one-, or many-LLM-call paths, and then by an
**agent loop** that lets the language model choose which retrieval-backed
**tools** to invoke.

```
Browser / CLI / eval harness
            │  HTTP (JSON or SSE)
            ▼
      FastAPI app ───────────────┐
            │                    │
            ▼                    ▼
     Turn Handler          Admin document upsert
   (stage1 + stage2 +     (corpus ingestion pipeline)
    loop breaker)
            │
            ▼
     Agent loop (Groq function calling)
            │
  ┌─────────┼──────────┬─────────────┐
  ▼         ▼          ▼             ▼
check_   search_    filter_by_   compare_       budget /
scope    destinations criteria    destinations   context tools
            │
            ▼
   Hybrid retriever (dense + sparse + RRF)
            │
      ┌─────┴─────┐
      ▼           ▼
    Chroma       BM25          Redis (memory + filler counter)
```

Every turn ends by appending the user message and the final answer to the
session's Redis history so follow-up questions have context.

---

## 2. Request Path (End to End)

### 2.1 HTTP layer

- `app/main.py` — builds the FastAPI app, wires routers under `/api/v1`,
  registers `/health`, and preloads the embedding model and Chroma collection
  in the lifespan hook so the first request avoids cold-load latency.
- `app/routers/chat_router.py` — `POST /api/v1/chat` accepts `query`, `sender`,
  and `stream` as query parameters. With `stream=false` it returns
  `{"answer", "retrieved_context"}`; with `stream=true` it returns a
  Server-Sent Events stream of `{chunk}` payloads terminated by `[DONE]`.
  Two debug endpoints expose `check_scope` and `search_destinations` directly
  for manual tool testing.
- `app/routers/documents_routers.py` — `POST /api/v1/documents/upsert`
  ingests or updates one destination document (see §5.3). Protected by the
  `X-Admin-Token` header against `ADMIN_API_KEY`; returns 503 if the key is
  not configured.

### 2.2 Turn Handler

The turn orchestration lives in `app/agent/graph.py` (`TurnGraph`), a small
framework-free class whose nodes are async functions wired via constructor
injection (default = the real module-level functions, so production needs no
configuration):

1. **Stage 1 — rule fast-path** (`app/agent/nodes/stage1_rules_node.py`):
   regex match on pure filler (greetings, acknowledgments, farewells).
   Zero LLM calls. The patterns and templated replies are deliberately
   hardcoded here; they are curated copy that should never be duplicated
   elsewhere.
2. **Stage 2 — LLM classifier** (`app/agent/nodes/stage2_classifier_node.py`,
   **async**): for anything the regex did not match, classify the turn as
   `new_question` / `filler` / `unclear`. It passes the assistant's last
   message as context so short engaged replies ("yes", "I wanna compare")
   are not misread as filler. Fails safe to `new_question` on any error.
3. **Loop breaker** (`app/agent/nodes/loop_breaker_node.py`): a per-session
   Redis counter that trips after `FILLER_LOOP_THRESHOLD` consecutive filler
   turns and nudges the user back to a real question, then resets.
4. **Agent node** — runs the full async Groq tool-calling loop (§ 2.3).

`app/services/chat_service.py` is now thin glue over the graph: it delegates
to `TurnGraph.run()` / `run_stream()` and emits the `turn_handled` log line.
All Redis and sync-tool calls inside the graph and the agent run through
`asyncio.to_thread` so they never block the event loop.

The streaming path (`run_stream()`) mutates a caller-owned `Turn` (async
generators cannot return values) and the router reads it afterwards, emitting
a final SSE metadata event (`{"retrieved_context": [...]}`) before `[DONE]`.

### 2.3 Agent loop

`app/agent/agent.py` implements the tool-calling loop (now fully async,
using `AsyncGroq`):

1. Build `messages` = system prompt + Redis history (limited by
   `CONVERSATION_HISTORY_LIMIT`, fetched via `asyncio.to_thread`) + the
   user query.
2. Call Groq with the tool schemas from `app/agent/tool_registry.py`.
3. If the model returns no tool calls — done; return `{answer,
   retrieved_context}`.
4. Otherwise append the assistant message **with** `tool_calls`, execute each
   call (via `asyncio.to_thread`, since the tool functions are synchronous),
   append each result as a `tool` message, and loop (bounded by
   `MAX_TOOL_ITERATIONS`, then a graceful apology message).

Two entry points:

- `run_agent(query, sender)` — non-streaming. Returns `{"answer",
  "retrieved_context"}`. It captures every chunk actually retrieved this turn
  so the faithfulness judge can score against exactly what the agent saw.
- `run_agent_stream(query, sender)` — streaming. Yields text deltas; tool-call
  rounds happen silently in between. Writes the retrieved chunks into the
  caller-supplied `retrieved_context` list (since async generators cannot
  return values).

Content deltas are buffered per `tool_calls` index to reassemble multi-part
function names/arguments.

### 2.4 Tools

`app/agent/tools/` holds the six tool implementations plus the name resolver:

| Tool | Purpose |
| --- | --- |
| `check_scope_tool` | Guardrail. Keyword fast-path for obvious out-of-scope phrases, otherwise an LLM classifier. **Fails safe** (refuses) on error — an out-of-scope slip is worse than an over-cautious refusal. |
| `search_destinations_tool` | Wraps `hybrid_search()`; flattens results to `{destination, section, text}` for the LLM. Most-called tool. |
| `compare_destinations_tool` | Resolves names, then queries Chrome for the comparison sections (`Overview`, `Difficulty`, `Duration`, `Estimated Budget`, `Best Time to Visit`, `Permits`) per destination. Unknown names are reported under `_unresolved`. |
| `filter_by_criteria_tool` | Metadata-filtered discovery (province / difficulty / type), deduped by destination name, default `max_results=10`. |
| `calculate_budget_estimate_tool` | Returns raw budget + typical duration + requested days. Deliberately does **no** arithmetic — corpus budget values are inconsistently formatted, so the LLM reasons over the raw estimate instead. |
| `get_conversation_context_tool` | Recent message history for referential follow-ups ("that place"). |
| `destination_resolver` | Alias map → exact corpus name (e.g. `abc` → `Annapurna Base Camp (ABC)`). |

`tool_registry.py` exports the schemas handed to Groq (`get_tool_schemas()`)
and the dispatch map (`call_tool(name, **kwargs)`).

### 2.5 Hybrid retrieval

`app/retrieval/hybrid_retriever.py` fuses two independent rankers:

- **Dense**: `bge-base-en-v1.5` embeddings (`retrieval/embeddings.py`), query
  side gets the bge retrieval prefix, stored in Chroma
  (`db/chroma/client.py`, cosine space).
- **Sparse**: in-memory BM25 (`retrieval/bm25_index.py`) over the same chunks,
  with comma-in-number tokenization normalisation (`5,106` → `5106`).
- **Fusion**: Reciprocal Rank Fusion (`retrieval/fusion.py`, `k=60`).

The result-set is rehydrated through BM25's chunk lookup and returned as
`{id, text, metadata, score}`.

---

## 3. Configuration Surface

Single source of truth: `app/core/config.py` (pydantic-settings) reads
`app/core/.env`; `app/configs/agent_config.py` holds agent-level tunables.
Everything else imports from these two.

| Setting | Default | Used by |
| --- | --- | --- |
| `app_name` | `Yatra` | FastAPI title, `/health` |
| `environment` | `development` | — |
| `groq_api_key` | **required** | Groq client |
| `admin_api_key` | `None` | Protects `documents/upsert`; if unset the endpoint returns 503 |
| `redis_url` | `redis://localhost:6379/0` | Redis client |
| `chroma_persist_dir` | `<project>/data/chroma_db` | Chroma client |
| `corpus_dir` | `<project>/data/corpus` | chunking, BM25, document service |
| `embedding_model` | `BAAI/bge-base-en-v1.5` | embeddings |

Agent tunables in `app/configs/agent_config.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `AGENT_MODEL` | `openai/gpt-oss-20b` | All agent/classifier/judge LLM calls |
| `MAX_TOOL_ITERATIONS` | `5` | Tool-loop budget |
| `CONVERSATION_HISTORY_LIMIT` | `12` | Messages injected per turn |
| `MAX_STORED_HISTORY_MESSAGES` | `20` | Redis list cap (ltrim) |
| `SESSION_TTL_SECONDS` | `7200` | Session/TTL, also used by loop breaker |
| `FILLER_LOOP_THRESHOLD` | `3` | Consecutive filler turns before break |
| `FINAL_ANSWER_MAX_TOKENS` | `400` | Final answer cap |

File-system defaults are anchored to the project root, not the process CWD,
so the app behaves the same regardless of launch directory.

---

## 4. Data Model

### 4.1 Corpus documents (`data/corpus/*.md`)

Each destination is one Markdown file: a YAML frontmatter block (name,
province, district, type, difficulty, time_required, max_altitude_m, permits,
budget, etc.) followed by `##`-section bodies. 10 destinations, curated for
domestic tourism.

### 4.2 Chunks

`app/data_ingestion/chunker.py` parses frontmatter and splits sections:

- Text before the first `##` header becomes an `Overview` chunk (a fix for an
  early bug that silently dropped it).
- Each `## Section` becomes one chunk.
- Chunk IDs are `<filename>__<section>` (or `__overview`).
- Metadata = flattened frontmatter + `section_title` + `source_file`.
  **`None` frontmatter values are dropped**, not stringified to `"None"`,
  so absent attributes never poison Chroma metadata.

### 4.3 Conversation memory (`app/components/redis/session_store.py`)

Each sender maps to a Redis list `session:<sender>:messages` of JSON
`{"role", "content"}` objects, capped at `MAX_STORED_HISTORY_MESSAGES`, TTL
reset on every write. A separate `session:<sender>:filler_count` key feeds
the loop breaker.

### 4.4 Vector store (`app/db/chroma/client.py`)

Persistent Chroma under `data/chroma_db`, collection `yatra_destinations`,
cosine space. Chunks are upserted by ID; `delete_chunks_by_source_file`
removes stale chunks for a file before re-ingesting.

---

## 5. Write Paths

### 5.1 Index build (`scripts/build_index.py`)

Chunk the corpus → embed → reset the Chroma collection → upsert everything.
Run after the corpus changes wholesale, or to initialise a fresh checkout.

### 5.2 On-disk persistence

Search paths that mutate disk: `data/corpus/` (document upserts),
`data/chroma_db/` (Chroma persistence). Both are gitignored except the
markdown corpus itself.

### 5.3 Document upsert (`app/services/document_service.py`)

`POST /api/v1/documents/upsert` validates frontmatter, writes
`data/corpus/<filename>.md`, re-chunks the single file, deletes that file's
old chunks from Chroma, re-embeds, upserts, and resets the BM25 cache so the
index rebuilds from disk on next use. Returns `{filename, is_update,
old_chunks_deleted, new_chunks_written, chunk_ids}`.

---

## 6. Evaluation

All eval tools live in `scripts/` (the older `eval/run_eval.py` is removed —
superseded):

| Script | What it measures | Needs |
| --- | --- | --- |
| `scripts/eval_retrieval.py` | Precision@K / Recall@K / MRR against labeled `source_docs` cases | Chroma + embedding model (no LLM) |
| `scripts/run_eval.py` | End-to-end per-case behavior over HTTP (12 cases) | Running server + Groq |
| `scripts/eval_faithfulness.py` | LLM-as-judge over `eval/results.jsonl` using the **captured** `retrieved_context` | Running service already having produced results |

Evaluation cases: `eval/cases/basic_eval.jsonl` (12 cases: factual,
comparison, discovery, budget, out-of-scope, adversarial, conflicting-source,
filler). Results land in `eval/results.jsonl` and
`eval/faithfulness_results.jsonl`.

`eval/faithfulness_judge.py` is a strict fact-checker: given the retrieved
context and the answer, it returns `{faithful, unsupported_claims, notes}`.
Returns `faithful: null` on judge failure rather than a confident wrong
verdict.

---

## 7. Tests

- `tests/test_chunker.py` — pure-logic unit tests for the chunker, including
  the `None`-metadata handling and the pre-header Overview capture.
- `tests/test_app.py` — FastAPI smoke tests (`/health`, `/openapi.json`
  routes). `TestClient` is used without entering the lifespan context so the
  embedding model / Chroma are not warmed up during the test.

Run: `uv run pytest tests/ -v`. CI runs ruff + pytest on Python 3.13
(`.github/workflows/ci.yml`).

---

## 8. Operational Notes

### 8.1 Running

```bash
uv sync
cp app/core/.env.example app/core/.env   # set GROQ_API_KEY
docker compose -f docker/docker-compose.yml up -d   # Redis
uv run python scripts/build_index.py               # initial index
uv run uvicorn app.main:app --reload
```

### 8.2 Dependencies on external services at runtime

- **Groq API** — every real agent turn, plus stage-2 classification.
- **Redis** — conversation memory and filler counter. The lifespan hook
  pings Redis at startup (warn-only — the app starts even if Redis is down)
  and closes the connection on shutdown. Session memory silently degrades to
  empty history when Redis is unreachable.
- **Chroma** — persisted locally; preloaded at startup.
- **Sentence-transformers / `bge-base-en-v1.5`** — preloaded at startup; with
  `HF_HUB_OFFLINE=1` it expects the model in the local HF cache.

### 8.3 Security notes

- `groq_api_key` is loaded only from the environment / `app/core/.env`, which
  is gitignored. A committed `okay.zip` snapshot that packaged an old `.env`
  has been removed from the working tree; **rewrite git history** to scrub it
  if it was ever pushed:
  `git filter-repo --path okay.zip --invert-paths`.
- No user input is executed; the debug endpoints dispatch only to the six
  registered tools.
- `documents/upsert` writes only into the configured corpus directory and
  validates the filename to prevent path traversal. It is protected by
  `X-Admin-Token` header against `ADMIN_API_KEY` (returns 403 if the header
  is missing or wrong; 503 if the key is not configured on the server).
- Redis, Chroma, and sync tool calls are dispatched via `asyncio.to_thread`
  inside the agent loop and graph nodes so they never block the FastAPI event
  loop.

### 8.4 Known limitations (unchanged by refactor)

1. **Streaming path does not capture `retrieved_context` or log token usage**
   (`run_agent_stream`); the non-streaming path does. Faithfulness eval on
   streamed turns is therefore not supported yet.
2. **Budget estimates are approximate** by design — corpus budget text is
   inconsistently formatted and the tool does not do arithmetic.
3. **Latency** on real agent turns is ~3–4 s, higher under Groq rate limiting.
4. **English-first corpus**; Nepali support is future work.
5. **10-destination corpus** — a deliberate, documented scope decision.

---

## 9. Directory Map (Actual)

```
app/
├── main.py                      # FastAPI app factory + lifespan
├── configs/agent_config.py      # agent/LLM tunables (single source)
├── core/
│   ├── config.py                # environment settings (pydantic-settings)
│   └── logging_config.py        # stdout logging setup
├── routers/                     # chat + documents + debug endpoints
├── services/                    # chat_service (turn handler), document_service
├── agent/
│   ├── graph.py                 # TurnGraph — stage1 → stage2 → agent orchestration
│   ├── agent.py                 # async Groq tool loop (streaming + non-streaming)
│   ├── tool_registry.py         # schemas + dispatch
│   ├── nodes/                   # stage1, stage2, loop breaker
│   └── tools/                   # 6 tools + destination_resolver
├── components/
│   ├── groq/client.py           # cached Groq client
│   └── redis/                   # client + session_store
├── db/chroma/client.py          # persistent vector store
├── retrieval/                   # hybrid_retriever, bm25_index, embeddings, fusion
├── data_ingestion/chunker.py    # frontmatter + section chunking
├── schemas/document_schema.py   # upsert request schema
└── utils/logger.py              # structured event logger
data/{corpus,raw,chroma_db}/     # source docs, JSON copies, persisted index
eval/                            # cases, faithfulness judge, results
scripts/                         # build_index, eval scripts
tests/                           # chunker unit tests + app smoke tests
ui/                              # React + Vite chat frontend
docker/docker-compose.yml        # Redis
docs/ARCHITECTURE.md             # this file
```