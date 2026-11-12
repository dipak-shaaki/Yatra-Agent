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
        │  HTTP (JSON or SSE, X-Session-Token)
        ▼
   FastAPI app ──────────────────────┐
        │                             │
        ▼                             ▼
 Turn Handler                    Admin document upsert
 (stage1 regex fast-path,        (corpus ingestion
  stage2 LLM classifier,          pipeline)
  loop breaker)
        │
        │ new_question only — unclear/out_of_scope/filler
        │ get fixed replies and never reach the agent
        ▼
 Agent loop (Groq function calling, streaming)
        │
        ├─ search_destinations ──► hybrid retriever ──► Chroma + BM25
        ├─ filter_by_criteria ───► corpus metadata (frontmatter model)
        ├─ compare_destinations ─► corpus chunks + metadata
        └─ calculate_budget ─────► corpus metadata

 Redis: session tokens, history, filler counter, rate limit
```

Out-of-scope and unclear turns are answered by the classifier branch and
never reach the agent loop. Every completed turn appends the user message
and the final answer to the session's Redis history so follow-up questions
have context.

---

## 2. Request Path (End to End)

### 2.1 HTTP layer

- `app/main.py` — builds the FastAPI app, wires routers under `/api/v1`,
  registers `/health`, and preloads the embedding model and Chroma collection
  in the lifespan hook so the first request avoids cold-load latency.
- `app/routers/chat_router.py`:
  - `POST /api/v1/session` mints a server-generated session token (the
    caller's identity for all later calls).
  - `POST /api/v1/chat` takes a JSON body `{query, stream}` and the
    `X-Session-Token` header. The token is validated, then a fixed-window
    rate limit (`chat_rate_limit_per_minute`, default 30) is enforced per
    token — 401 for missing/unknown tokens, 429 with `Retry-After` when
    over the limit, 503 (fail-closed) when Redis is unreachable.
    `stream=true` returns SSE `{chunk}` payloads terminated by `[DONE]`
    (plus a final `{retrieved_context, tools_called}` event when
    `EXPOSE_EVAL_CONTEXT=true`); `stream=false` returns `{answer}`.
- `app/routers/dependencies.py` — `require_admin` dependency: mutating
  endpoints need `X-Admin-Token` matching `ADMIN_API_KEY` (403 on mismatch,
  503 if the key is not configured server-side).
- `app/routers/documents_routers.py` — `POST /api/v1/documents/upsert`
  ingests or updates one destination document (§5.3), admin-guarded.
- Debug endpoints (`/api/v1/debug/*`) expose `search_destinations`
  directly for manual tool testing; also admin-guarded.

### 2.2 Turn Handler

The turn orchestration lives in `app/agent/graph.py` (`TurnGraph`), a small
framework-free class whose nodes are async functions wired via constructor
injection (default = the real module-level functions, so production needs no
configuration):

1. **Stage 1 — rule fast-path** (`app/agent/nodes/stage1_rules_node.py`):
   regex match on pure filler (greetings, acknowledgments, farewells), so no
   classifier or agent loop runs. Patterns live in one `FILLER_PATTERNS` dict
   (category → patterns) compiled once at import. Greetings/farewells get a
   short reply written by `filler_reply_node.py` (router model, no tools,
   80-token cap, 8s timeout); any failure falls back to the curated templates
   in the same module. Acknowledgments keep the template so "thanks" stays
   free, and loop-break replies take priority over both.
2. **Stage 2 — LLM classifier** (`app/agent/nodes/stage2_classifier_node.py`,
   **async**): for anything the regex did not match, classify the turn as
   `new_question` / `filler` / `unclear` / `out_of_scope`. It passes the
   assistant's last message as context so short engaged replies ("yes",
   "I wanna compare") are not misread as filler. `unclear` and `out_of_scope`
   get templated replies and never reach the agent. Fails safe to
   `new_question` on any error — an over-cautious refusal is worse than a
   question that gets answered.
3. **Loop breaker** (`app/agent/nodes/loop_breaker_node.py`): a per-session
   Redis counter that trips after `filler_loop_threshold` consecutive filler
   turns and nudges the user back to a real question, then resets.
4. **Agent node** — runs the full async Groq tool-calling loop (§2.3).

`app/services/chat_service.py` is thin glue over the graph: it delegates to
`TurnGraph.run()` / `run_stream()` and emits the `turn_handled` log line.
All Redis and sync-tool calls inside the graph and the agent run through
`asyncio.to_thread` so they never block the event loop.

Both delivery modes share one streaming loop: `run_stream()` mutates a
caller-owned `Turn` (async generators cannot return values); the JSON path
simply discards the deltas and reads the assembled reply, the SSE path
forwards them.

### 2.3 Agent loop

`app/agent/agent.py` implements a single streaming tool-calling loop
(`run_agent_stream`, `AsyncGroq`), used by both delivery modes:

1. Build `messages` = system prompt + Redis history (limited by
   `conversation_history_limit`, fetched via `asyncio.to_thread`) + the
   user query.
2. Call Groq with the tool schemas from `app/agent/tool_registry.py`,
   `stream=True`, `reasoning_effort="low"` for reasoning models
   (`reasoning_params` in the Groq client) so reasoning tokens do not eat
   the answer budget.
3. Content deltas are yielded as they arrive. Tool-call deltas are buffered
   per `tool_calls` index to reassemble multi-part function names/arguments.
   Token usage rides `x_groq.usage` on the final chunk and is logged per
   round (`llm_call_stream`).
4. If the model returns no tool calls: done. Empty final content (reasoning
   tokens ate the budget) triggers the `agent_empty_content` guard and a
   graceful error reply instead of an empty stream.
5. Otherwise append the assistant message **with** `tool_calls`, execute
   **independent calls in parallel** (`asyncio.gather`), append each result
   as a `tool` message, and loop — bounded by `max_tool_iterations`, then
   a graceful apology message.

State (answer, retrieved context, tools called) is written into a
caller-owned `AgentResult` dataclass, since async generators cannot return
values. Captured context is what the eval judge scores against — exactly
what the agent saw, not a re-retrieval.

### 2.4 Tools

`app/agent/tools/` holds four implementations. The old `check_scope` and
`get_conversation_context` tools were deleted — scope is enforced upstream
by the classifier, history is already in the prompt:

| Tool | Purpose |
| --- | --- |
| `search_destinations_tool` | Wraps `hybrid_search()`; flattens results to `{id, text, metadata, score}` for the LLM. Most-called tool. |
| `filter_by_criteria_tool` | Metadata-filtered discovery (province / difficulty / type) over the frontmatter model. Difficulty matches within ±1 level (the corpus's 1–5 scale is coarse); type tolerates plurals ("treks" → "trek"). Default `max_results=10`. |
| `compare_destinations_tool` | Resolves names via the alias map, then queries corpus chunks for the comparison sections (`Overview`, `Difficulty`, `Duration`, `Estimated Budget`, `Best Time to Visit`, `Permits`) per destination. Unknown names are reported under `_unresolved`. |
| `calculate_budget_estimate_tool` | Does the arithmetic in code: per-day budgets scale linearly with the requested day count; requests outside the typical day range are scaled proportionally and marked as extrapolations. The corpus stores numeric ranges + day bounds for exactly this. |

`tool_registry.py` exports the schemas handed to Groq (`get_tool_schemas()`)
and the dispatch map (`call_tool(name, **kwargs)`).

### 2.5 Hybrid retrieval

`app/retrieval/hybrid_retriever.py` fuses two independent rankers:

- **Dense**: `bge-base-en-v1.5` embeddings (`retrieval/embeddings.py`), query
  side gets the bge retrieval prefix, stored in Chroma
  (`db/chroma/client.py`, cosine space). Sentence-transformers is imported
  lazily so config-only imports stay fast.
- **Sparse**: in-memory BM25 (`retrieval/bm25_index.py`) over the same chunks,
  with comma-in-number tokenization normalisation (`5,106` → `5106`).
- **Fusion**: Reciprocal Rank Fusion (`retrieval/fusion.py`, `rrf_k=60`),
  each ranker contributing `candidate_pool` (default 10) candidates.

The BM25 index is cached per process against `(corpus_dir, corpus_version)`;
the version lives in Redis (`yatra:corpus_version`) and is bumped by every
document upsert, so other workers rebuild automatically instead of serving a
silently stale index. If Redis is unreachable the cached index is served
(no thrash). The fused result-set is rehydrated through BM25's chunk lookup
and returned as `{id, text, metadata, score}`.

---

## 3. Configuration Surface

Single source of truth: `app/core/config.py` (pydantic-settings) reads
`app/core/.env` plus real environment variables. The old
`app/configs/agent_config.py` is gone — every tunable, agent-level included,
lives in `Settings`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `groq_api_key` | **required** | Groq client |
| `agent_model` | `openai/gpt-oss-20b` | The agent/classifier model |
| `judge_model` | `llama-3.3-70b-versatile` | Eval judge — deliberately a *different* model than the generator, or the judge inherits the generator's blind spots |
| `reasoning_effort` | `low` | Sent to reasoning models only |
| `final_answer_max_tokens` | `2000` | Final answer cap |
| `max_tool_iterations` | `5` | Tool-loop budget |
| `conversation_history_limit` | `12` | Messages injected per turn |
| `max_stored_history_messages` | `20` | Redis list cap (ltrim) |
| `session_ttl_seconds` | `7200` | Session token TTL |
| `filler_loop_threshold` | `3` | Consecutive filler turns before break |
| `candidate_pool` | `10` | Per-ranker results fetched before RRF |
| `rrf_k` | `60` | Standard RRF constant |
| `chroma_persist_dir` | `<project>/data/chroma_db` | Chroma client |
| `corpus_dir` | `<project>/data/corpus` | chunking, BM25, document service |
| `embedding_model` | `BAAI/bge-base-en-v1.5` | embeddings |
| `hf_offline` | `false` | Opt-in `HF_HUB_OFFLINE` for airgapped runs |
| `admin_api_key` | `None` | Protects admin endpoints; unset → 503 |
| `chat_rate_limit_per_minute` | `30` | Fixed-window limit per session token |
| `expose_eval_context` | `false` | Adds `retrieved_context`/`tools_called` to chat responses (eval/debug) |
| `redis_url` | `redis://localhost:6379/0` | Redis client |

File-system defaults are anchored to the project root, not the process CWD,
so the app behaves the same regardless of launch directory.

---

## 4. Data Model

### 4.1 Corpus documents (`data/corpus/*.md`)

Each destination is one Markdown file: a YAML frontmatter block (name,
aliases, province, district, tags, difficulty, days_min/max, budget bounds +
`budget_per_day`, permits, etc.) followed by `##`-section bodies. 10
destinations, curated for domestic tourism.

**Provenance:** the corpus is authored and maintained in this repository —
there is no automated ingestion from external sources. Every document
carries a `last_updated` date in its frontmatter, and budget figures are
rough domestic-traveler estimates (stated as such in the corpus text
itself), which is why the budget tool marks scaled or out-of-range
estimates as approximations. Cite the file and date when answering
(§2.3 system prompt), so permit/price staleness is visible to the user.

### 4.2 Typed metadata (`app/data_ingestion/corpus_meta.py`)

Frontmatter is parsed into a **pydantic model** (`Destination`) — malformed
or missing fields fail loudly at load time instead of poisoning queries.
The module also provides the alias resolver (`resolve_destination`,
e.g. `abc` → `Annapurna Base Camp (ABC)`) and `get_corpus_chunks()`, the
in-memory chunk list BM25 rehydrates from.

### 4.3 Chunks

`app/data_ingestion/chunker.py` parses frontmatter and splits sections:

- Text before the first `##` header becomes an `Overview` chunk.
- Each `## Section` becomes one chunk.
- Chunk IDs are `<filename>__<section>` (or `__overview`), e.g.
  `manaslu_circuit__permits`.
- Metadata = flattened frontmatter + `section_title` + `source_file`.
  **`None` frontmatter values are dropped**, not stringified to `"None"`.

### 4.4 Conversation memory & sessions (`app/components/redis/session_store.py`)

Each sender (the session token) maps to a Redis list
`session:<sender>:messages` of JSON `{"role", "content"}` objects, capped at
`max_stored_history_messages`, TTL reset on every write. Separate keys hold
the filler count (`:filler_count`) and the fixed-window rate-limit counter.
Session tokens are server-minted random strings with TTL
`session_ttl_seconds`; they are the sender identity, so clients cannot read
or overwrite another session's history.

### 4.5 Vector store (`app/db/chroma/client.py`)

Persistent Chroma under `data/chroma_db`, collection `yatra_destinations`,
cosine space. The collection records `embedding_model`, the pinned
`embedding_revision` (commit SHA) and `embedding_dimension` in its metadata;
startup verifies all three and fails fast on a mismatch instead of silently
returning garbage. The revision is pinned because an upstream retag keeps the
same model name and dimension while changing the weights. Chunks are
upserted by ID; `get_chunk_ids_by_source_file` + `delete_chunks(ids)` remove
only the IDs that disappeared, after the new chunks are in place.

---

## 5. Write Paths

### 5.1 Index build (`scripts/build_index.py`)

Chunk the corpus → embed → reset the Chroma collection → upsert everything.
Run after the corpus changes wholesale, or to initialise a fresh checkout.
Run as `python -m scripts.build_index` from the project root (no sys.path
bootstrap).

### 5.2 On-disk persistence

Search paths that mutate disk: `data/corpus/` (document upserts),
`data/chroma_db/` (Chroma persistence). Both are gitignored except the
markdown corpus itself.

### 5.3 Document upsert (`app/services/document_service.py`)

`POST /api/v1/documents/upsert` validates the frontmatter against the
pydantic model and builds chunks in memory, embeds every chunk, writes
`data/corpus/<filename>.md` atomically (temp file + `os.replace`), upserts the
new chunks, deletes only the IDs that disappeared, then bumps the Redis corpus
version so every worker's corpus snapshot / BM25 index invalidates. An invalid
document or a failed embedding writes nothing; the upsert-before-delete order
means there is never a window where the destination has no chunks.
Returns `{filename, is_update, old_chunks_deleted, new_chunks_written,
chunk_ids}` (422 for a schema-invalid document).

---

## 6. Evaluation

All eval tooling lives in `scripts/` and `eval/`:

| Script | What it measures | Needs |
| --- | --- | --- |
| `python -m scripts.eval_retrieval` | Chunk-level Precision@5 / Recall@5 / MRR@5 against labeled `source_chunks`, with 95% bootstrap CIs and `--gate-*` thresholds that exit nonzero; `--retriever dense\|bm25` doubles as a hybrid ablation | Built Chroma index + embedding model (no LLM, no server) |
| `python -m scripts.run_eval` | End-to-end per-case behavior over HTTP; mints a fresh session per case; captures `retrieved_context` + `tools_called` per turn (server must run with `EXPOSE_EVAL_CONTEXT=true`) | Running server + Groq |
| `python -m scripts.eval_answers` | LLM answer judge over `eval/results.jsonl`, per-category verdict requirements, exact-set tool-selection accuracy, p50/p95 latency, mean LLM calls; writes `eval/judged.jsonl` | Groq (judge model) |

Evaluation cases: `eval/cases/*.jsonl` — **78 behavioral cases across 9
categories** (factual 20, comparison 6, discovery 8, budget 8, out_of_scope 8,
adversarial 4, ambiguous_conflicting 8, filler 8, unclear 8). Every case
labels `source_docs`, `source_chunks`, `expected_tools`, `category`,
`difficulty`, plus per-case `requires_refusal` / `hedge` flags; all facts
are verified against the corpus. `eval/retrieval_cases/*.jsonl` adds **145
retrieval-routed cases** generated from the corpus frontmatter (per-destination
best time, route, duration, difficulty, budget, permits, altitude, plus
exact-term/number queries) so the retrieval gate is not fitting to ~25 cases.

`eval_retrieval.py` loads both directories by default (`--eval-dir` is
repeatable). `eval/BASELINE.md` records a committed baseline with the git SHA
and settings used.

Two design notes:

- The retrieval eval scores only cases whose expected path is
  `search_destinations` — discovery/budget questions go through the
  metadata tools, so scoring them against hybrid search would measure
  nothing. Gates are calibrated below current hybrid numbers
  (P=0.19 / R=0.96 / MRR=0.75 over 176 scored cases); precision tops out
  near |expected|/K because most cases label a single chunk. BM25 alone
  misses the same gate (it is reported as an ablation, not gated in CI).
- `eval/answer_judge.py` (model: `judge_model`, different from the
  generator) returns `faithful` / `correct` / `refusal_correct` / `hedged`
  (true/false/null). A case passes when every verdict its category
  requires is true; cases that captured no context are not required to be
  faithful to a context they never saw. Verdicts are null on judge failure
  rather than a confident wrong verdict.

CI (`.github/workflows/ci.yml`) runs ruff + compileall + pytest against a
Redis service container, plus a separate `retrieval-eval` job that builds
the index and runs the retrieval gate.

---

## 7. Tests

- `tests/test_agent.py` — agent loop with a stubbed AsyncGroq: split-delta
  tool-call reassembly, AgentResult capture, empty-content guard,
  max-iterations bound, provider-error replies.
- `tests/test_app.py` — HTTP surface: sessions + per-IP mint throttle, auth
  failures, rate limit, chat modes, typed SSE, upsert 422.
- `tests/test_chunker.py` — chunker logic incl. `None`-metadata handling
  and pre-header Overview capture.
- `tests/test_corpus_meta.py` — frontmatter model + alias resolution and
  version-aware snapshot rebuilds.
- `tests/test_document_service.py` — validate/embed-before-write ordering,
  atomic swap, stale-chunk-only deletion.
- `tests/test_eval_scripts.py` — eval harness logic (verdict requirements,
  exact-set tool selection, percentiles, bootstrap CIs, retrieval metrics,
  prompt splicing).
- `tests/test_frontmatter.py` — frontmatter parsing edge cases.
- `tests/test_graph.py` — turn-handler routing per category; failed replies
  are not stored as assistant history.
- `tests/test_hybrid_retriever.py` — real temporary Chroma + deterministic
  fake embedder, empty-collection path.
- `tests/test_retrieval_units.py` — RRF fusion, BM25 empty corpus, and the
  stale-index-after-version-bump regression.
- `tests/test_stage2_classifier.py` — router model + JSON mode, category
  parsing, fail-open, no raw user text in logs.
- `tests/test_tools.py` — the four tools against the real corpus.

Run: `uv run pytest tests/ -v`. A real Redis is used when reachable;
otherwise the autouse fixture falls back to `fakeredis`, so no daemon is
required locally.

---

## 8. Operational Notes

### 8.1 Running

Host (Redis via Docker):

```bash
uv sync
cp app/core/.env.example app/core/.env   # set GROQ_API_KEY
docker compose -f docker/docker-compose.yml up -d redis
uv run python -m scripts.build_index
uv run uvicorn app.main:app --reload
```

Full stack in Docker (`Dockerfile` + `docker/docker-compose.yml`): the API
image rebuilds the Chroma index from the baked-in corpus on boot and mounts
named volumes for `chroma_db` and the HF model cache, so restarts are cheap.

### 8.2 Dependencies on external services at runtime

- **Groq API** — every real agent turn, plus stage-2 classification.
- **Redis** — **required for chat**: session minting/validation and the
  rate limit fail closed (503) when Redis is down; the startup ping is
  warn-only so the app still boots for /health.
- **Chroma** — persisted locally; preloaded at startup.
- **Sentence-transformers / `bge-base-en-v1.5`** — preloaded at startup;
  with `hf_offline=true` it expects the model in the local HF cache
  (`HF_HOME`).

### 8.3 Security notes

- `groq_api_key` is loaded only from the environment / `app/core/.env`,
  which is gitignored.
- Session tokens are server-minted and act as the sender identity;
  clients cannot guess or forge another session's history. Redis outage
  fails closed (503), never anonymous access.
- Chat input is length-capped (`query` ≤ 2000 chars); no user input is
  executed anywhere; debug endpoints dispatch only to registered tools.
- `documents/upsert` writes only into the configured corpus directory and
  validates the filename to prevent path traversal. It is protected by
  the `X-Admin-Token` header against `ADMIN_API_KEY` (403 on mismatch,
  503 if the key is not configured server-side).
- Redis, Chroma, and sync tool calls are dispatched via `asyncio.to_thread`
  inside the agent loop and graph nodes so they never block the FastAPI
  event loop.

### 8.4 Known limitations

1. **Budget estimates are approximate** by design — the corpus ranges are
   rough domestic-traveler estimates, and out-of-range day counts are
   linear extrapolations the tool marks as such.
2. **Latency** is dominated by Groq round trips (classifier + agent turns);
   measure it rather than assume — `scripts/run_eval.py` records per-case
   times and `scripts/eval_answers.py` reports p50/p95.
3. **English-first corpus**; Nepali support is future work.
4. **10-destination corpus** — a deliberate, documented scope decision.

---

## 9. Directory Map (Actual)

```
app/
├── main.py                      # FastAPI app factory + lifespan
├── core/
│   ├── config.py                # all settings (pydantic-settings)
│   └── logging_config.py        # stdout logging setup
├── routers/                     # chat (+session/debug), documents, dependencies
├── services/                    # chat_service (turn handler), document_service
├── agent/
│   ├── graph.py                 # TurnGraph — stage1 → stage2 → agent orchestration
│   ├── agent.py                 # streaming Groq tool loop (AgentResult)
│   ├── tool_registry.py         # schemas + dispatch
│   ├── nodes/                   # stage1, stage2, loop breaker
│   └── tools/                   # 4 tools (search, filter, compare, budget)
├── components/
│   ├── groq/client.py           # cached Groq client + reasoning params
│   └── redis/                   # client + session_store
├── db/chroma/client.py          # persistent vector store
├── retrieval/                   # hybrid_retriever, bm25_index, embeddings, fusion
├── data_ingestion/             # chunker, corpus_meta (pydantic frontmatter model)
├── schemas/                     # request/response schemas
└── utils/logger.py              # structured event logger
data/{corpus,raw,chroma_db}/     # source docs, JSON copies, persisted index
eval/                            # behavioral + retrieval cases, judge, baseline
scripts/                         # build_index, eval_retrieval, run_eval, eval_answers
tests/                           # unit + integration tests (pytest, fakeredis fallback)
ui/                              # React + Vite chat frontend
Dockerfile                       # API image
docker/docker-compose.yml        # API + Redis
docs/ARCHITECTURE.md             # this file
docs/decisions/                  # short ADRs (Chroma, no framework, model choice)
```
