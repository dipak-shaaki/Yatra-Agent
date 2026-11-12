# Yatra — Agentic RAG Assistant for Nepal Domestic Tourism

Yatra is a conversational AI assistant covering 10 curated Nepal destinations — treks, heritage towns, and pilgrimage sites. It's an **agentic RAG** system: the LLM picks and calls the tools it needs instead of following a fixed query → retrieve → generate pipeline.

## Features

- **Agentic tool calling** — the LLM (Groq) dynamically selects between `search_destinations`, `filter_by_criteria`, `compare_destinations`, and `calculate_budget_estimate`. Scope checking is enforced upstream by the turn classifier.
- **Hybrid retrieval** — dense (Chroma + `bge-base-en-v1.5`) and sparse (BM25) search fused with Reciprocal Rank Fusion for the best of both.
- **Section-aware chunking** — corpus Markdown is split on `##` headings instead of arbitrary character counts, preserving semantic boundaries.
- **Cost-aware turn handling** — a regex fast path answers greetings/acknowledgments with 0 LLM calls; an LLM classifier catches the rest. Repeated filler triggers a loop break.
- **Conversation memory** — Redis stores per-session history and the filler counter.
- **SSE streaming** — `POST /api/v1/chat` with `{"query": ..., "stream": true}` streams the answer token-by-token; sessions are minted via `POST /api/v1/session` and rate-limited per token.
- **Evaluation harness** — 78 labeled behavioral cases across 9 categories (factual, comparison, discovery, budget, out-of-scope, adversarial, ambiguous, filler, unclear) plus 145 retrieval-routed cases, a chunk-level retrieval gate (Precision@K / Recall@K / MRR with bootstrap CIs), and an LLM answer judge scoring faithfulness, correctness, refusals, and hedging. Both the retrieval gate and the test suite run in CI.

## Tech Stack

| Component | Technology |
| --- | --- |
| Backend | FastAPI (Python ≥ 3.13, uv) |
| LLM | Groq — `openai/gpt-oss-20b` |
| Vector store / embeddings | Chroma, `bge-base-en-v1.5` |
| Sparse retrieval / fusion | BM25, RRF |
| Memory | Redis |
| Frontend | React + Vite (`ui/`) |

## Project Structure

```
├── app/
│   ├── routers/           # FastAPI routes (chat, documents, debug)
│   ├── services/          # chat_service (turn handler), document_service
│   ├── agent/
│   │   ├── agent.py       # tool-calling loop against Groq (streaming)
│   │   ├── graph.py       # turn-handler graph wiring the nodes
│   │   ├── nodes/         # stage1_rules, stage2_classifier, loop_breaker
│   │   ├── tool_registry.py
│   │   └── tools/         # 4 agent tools (search, filter, compare, budget)
│   ├── components/        # groq client, redis client/session store
│   ├── core/              # settings (single config), logging
│   ├── data_ingestion/    # section-based chunker, corpus metadata
│   ├── db/chroma/         # chroma client
│   ├── retrieval/         # hybrid_retriever, bm25_index, embeddings, fusion
│   ├── schemas/           # request/response schemas
│   └── utils/             # structured logger
├── data/
│   ├── corpus/            # 10 destination Markdown documents
│   └── raw/               # raw destination JSON documents
├── eval/                  # behavioral + retrieval cases, answer judge, baseline
├── scripts/               # build_index, eval_retrieval, run_eval, eval_answers
├── tests/                 # unit + integration tests (pytest)
├── docker/                # docker-compose (API + Redis)
├── ui/                    # React chat frontend
└── docs/                  # architecture reference
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module map.

## Architecture

```
Client (React UI / curl)
  │  POST /api/v1/session                     POST /api/v1/chat
  ▼                                              │
FastAPI (app/main.py) ── JSON logs (request/session id), /metrics, /ready
  │                                              │
  │                                   ┌──────────▼───────────┐
  │                                   │ TurnGraph            │
  │                                   │ stage 1: regex ──────┼──► filler reply, 0 LLM calls
  │                                   │ stage 2: classifier ─┼──► new / filler / unclear / out_of_scope
  │                                   │ loop-break ──────────┼──► repeated-filler guard
  │                                   │ agent loop (≤5) ─────┘
  │                                   └──────────┬───────────┘
  │                                              │ tool calls
  │            ┌─────────────────────────────────┼──────────────────────────┐
  │            ▼                                 ▼                          ▼
  │   search_destinations              filter_by_criteria         compare_destinations
  │            │                                 │                  calculate_budget_estimate
  │      hybrid_search                           │                          │
  │   ┌────────┴────────┐                        └──────► corpus_meta (versioned snapshot)
  │   ▼                 ▼
  │ Chroma (dense)   BM25 (sparse) ──► RRF fusion ──► top-5 chunks
  │
  └── Redis: session tokens · history · rate limits · corpus version
      Groq: agent_model (gpt-oss-20b) · router_model (8B classifier) · judge_model
```

## Evaluation Results

Retrieval gate over all labeled cases whose expected path is
`search_destinations` (176 cases, K=5). Brackets are 95% bootstrap CIs
(1000 resamples, seed 42). Commands:
`python -m scripts.eval_retrieval --retriever <name>`.

| Retriever | Precision@5 | Recall@5 | MRR@5 | Full misses |
| --- | --- | --- | --- | --- |
| **hybrid (gated)** | **0.193** [0.186, 0.199] | **0.955** [0.920, 0.983] | **0.751** [0.697, 0.798] | 8 |
| dense | 0.200 [0.195, 0.205] | 0.986 [0.969, 1.000] | 0.800 [0.755, 0.839] | 2 |
| BM25 | 0.164 [0.151, 0.176] | 0.812 [0.750, 0.869] | 0.617 [0.549, 0.679] | 33 |

The ablation is reported honestly: on the expanded synthetic set dense ranks
best, BM25 fails the gate, and hybrid sits between them. CI gates only hybrid
(`P≥0.15 / R≥0.85 / MRR≥0.70`); the others are diagnostics, not targets.
`eval/BASELINE.md` pins the git SHA, settings and exact commands behind these
numbers. The behavioral/LLM-judge half needs a valid `GROQ_API_KEY` and a
running server (`python -m scripts.run_eval` then `python -m scripts.eval_answers`).

## Quick Start

With uv on the host (Redis still via Docker):

```bash
uv sync
cp app/core/.env.example app/core/.env   # set GROQ_API_KEY
docker compose -f docker/docker-compose.yml up -d redis
uv run python -m scripts.build_index
uv run uvicorn app.main:app --reload
```

Or the whole stack in Docker (no local Python needed):

```bash
GROQ_API_KEY=gsk_... docker compose -f docker/docker-compose.yml up --build
```

Docs at `http://localhost:8000/docs`. UI: `cd ui && npm install && npm run dev`.

Design decisions are recorded in [`docs/decisions/`](docs/decisions/):
[Chroma](docs/decisions/0001-chroma.md), [no agent framework](docs/decisions/0002-no-framework.md),
[model choice](docs/decisions/0003-model-choice.md).

## License

MIT — see [`LICENSE`](LICENSE) for full text.

```
Copyright (c) 2026 dipak-shaaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```