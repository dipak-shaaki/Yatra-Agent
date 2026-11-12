# Yatra — Agentic RAG Assistant for Nepal Domestic Tourism

Yatra is a conversational AI assistant covering 10 curated Nepal destinations — treks, heritage towns, and pilgrimage sites. It's an **agentic RAG** system: the LLM picks and calls the tools it needs instead of following a fixed query → retrieve → generate pipeline.

## Features

- **Agentic tool calling** — the LLM (Groq) dynamically selects between `search_destinations`, `filter_by_criteria`, `compare_destinations`, and `calculate_budget_estimate`. Scope checking is enforced upstream by the turn classifier.
- **Hybrid retrieval** — dense (Chroma + `bge-base-en-v1.5`) and sparse (BM25) search fused with Reciprocal Rank Fusion for the best of both.
- **Section-aware chunking** — corpus Markdown is split on `##` headings instead of arbitrary character counts, preserving semantic boundaries.
- **Cost-aware turn handling** — a regex fast path answers greetings/acknowledgments with 0 LLM calls; an LLM classifier catches the rest. Repeated filler triggers a loop break.
- **Conversation memory** — Redis stores per-session history and the filler counter.
- **SSE streaming** — `POST /api/v1/chat` with `{"query": ..., "stream": true}` streams the answer token-by-token; sessions are minted via `POST /api/v1/session` and rate-limited per token.
- **Evaluation harness** — 60 labeled eval cases across 9 categories (factual, comparison, discovery, budget, out-of-scope, adversarial, ambiguous, filler, unclear), a chunk-level retrieval gate (Precision@K / Recall@K / MRR), and an LLM answer judge scoring faithfulness, correctness, refusals, and hedging. Both the retrieval gate and the test suite run in CI.

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
├── eval/                  # 60 eval cases (9 categories) + answer judge
├── scripts/               # build_index, eval_retrieval, run_eval, eval_answers
├── tests/                 # unit + integration tests (pytest)
├── docker/                # docker-compose (API + Redis)
├── ui/                    # React chat frontend
└── docs/                  # architecture reference
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module map.

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