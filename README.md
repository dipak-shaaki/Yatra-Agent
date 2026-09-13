# Yatra — Agentic RAG Assistant for Nepal Domestic Tourism

Yatra is a conversational, agentic RAG assistant covering 10 curated Nepal destinations — treks, heritage towns, and pilgrimage sites.

It is built as a production-style portfolio project demonstrating:

- Hybrid information retrieval
- Genuine LLM tool-calling agency
- Cost-aware conversation handling
- Redis-based conversation memory
- RAG evaluation and retrieval metrics
- LLM-based faithfulness evaluation
- Production-oriented backend architecture
- Explicit documentation of known limitations

The project intentionally documents limitations and evaluation weaknesses rather than hiding them.

## Architecture

```
                         ┌─────────────────┐
                         │      User       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    FastAPI      │
                         └────────┬────────┘
                                  │
                                  ▼
                   ┌──────────────────────────┐
                   │       Turn Handler       │
                   │                          │
                   │ Stage 1: Regex fast-path │
                   │ Stage 2: LLM classifier  │
                   │ Loop-break / filler      │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │       Agent Loop         │
                   │                          │
                   │ Groq Function Calling    │
                   │ Dynamic tool selection   │
                   └────────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
     check_scope       search_destinations    filter_by_criteria
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
   compare_destinations  calculate_budget_estimate  get_conversation_context
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │   Hybrid Retriever       │
                   │                          │
                   │ Chroma  → Dense Search   │
                   │ BM25    → Sparse Search  │
                   │ RRF     → Fusion         │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │       Destination        │
                   │        Corpus            │
                   │                          │
                   │ 10 curated destinations  │
                   └──────────────────────────┘

                                │
                                ▼
                   ┌──────────────────────────┐
                   │          Redis           │
                   │                          │
                   │ Conversation memory      │
                   │ Filler counter           │
                   └──────────────────────────┘
```

### Request Flow

A typical user request follows this flow:

```
User Query
    ↓
FastAPI
    ↓
Turn Handler
    ↓
Agent Loop
    ↓
LLM decides which tools are needed
    ↓
Tool execution
    ↓
Hybrid Retrieval when required
    ↓
Tool results returned to LLM
    ↓
Final response
    ↓
Conversation state stored in Redis
```

The important distinction is that Yatra is **agentic**, rather than a fixed `query → retrieve → generate` pipeline. The LLM dynamically decides which available tools to call and can call multiple tools when necessary.

For example:

```
User:
"Compare an easy trek near Pokhara with something under NPR 30,000."

Agent:
    ↓
check_scope
    ↓
search_destinations
    ↓
filter_by_criteria
    ↓
compare_destinations
    ↓
final response
```

## Features

### 1. Agentic Tool Calling

Yatra uses Groq's function-calling capabilities directly. The LLM can dynamically select tools such as:

- `check_scope`
- `search_destinations`
- `filter_by_criteria`
- `compare_destinations`
- `calculate_budget_estimate`
- `get_conversation_context`

This allows the system to handle different types of requests without requiring a fixed workflow for every query.

### 2. Hybrid Retrieval

Yatra combines two retrieval strategies.

**Dense Retrieval** — semantic similarity is handled using:

- Chroma
- Self-hosted `bge-base-en-v1.5` embeddings

This works well for queries such as `"easy trek near Pokhara"`, where the user may not use exactly the same words as the source document.

**Sparse Retrieval** — BM25 handles exact keyword matching. This is especially useful for queries containing:

- `TIMS card`
- `5106m`
- `permit`
- `altitude`
- `budget`

where exact terminology matters.

**Reciprocal Rank Fusion** — the dense and sparse rankings are combined using Reciprocal Rank Fusion (RRF):

```
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
   Dense Retrieval      BM25 Retrieval
      Chroma              Sparse
          │                 │
          └────────┬────────┘
                   ▼
                  RRF
                   │
                   ▼
            Final Ranking
```

**Why Hybrid Retrieval?**

Dense-only retrieval is not sufficient for this corpus. During testing, a BM25-isolation test initially missed Manaslu for an altitude query because of a tokenization issue: `5,106` was treated differently from `5106`. The tokenization was fixed and the retrieval behavior was re-tested successfully. This demonstrates why both semantic and lexical retrieval are useful for the project.

### 3. Section-Based Chunking

The destination documents are not split using arbitrary fixed character sizes. Instead, documents are split according to their own Markdown sections:

```
## Overview
...
## Highlights
...
## Difficulty
...
## Budget
...
```

Each section becomes a logical chunk. This preserves semantic boundaries and allows destination metadata such as *Province*, *Difficulty*, *Budget*, *Destination*, and *Category* to be attached to each chunk.

**Data-Loss Bug Discovered**

An early version of the chunker silently discarded text appearing before the first `##` heading. For example:

```
Manaslu Circuit is one of Nepal's...
Altitude: 5,106m

## Overview
...
```

The introductory content was lost, which caused the Manaslu altitude information to disappear from the indexed content. The chunking logic was fixed so that text before the first section header is captured as `Overview`. This is an example of a real data-pipeline issue discovered and fixed during development.

### 4. Cost-Aware Turn Handler

Not every message needs an LLM call. Yatra therefore uses a two-stage Turn Handler:

```
                User Message
                      │
                      ▼
             ┌─────────────────┐
             │    Stage 1      │
             │ Regex Fast Path │
             └────────┬────────┘
                      │
             ┌────────┴────────┐
             │                 │
          Filler            Ambiguous
             │                 │
             ▼                 ▼
       Respond directly    Stage 2 LLM
       0 LLM calls         classification
                               │
                               ▼
                         Agent / response
```

Examples of filler messages:

- `huss`
- `okay`
- `eh`
- `lala`
- `huncha`
- `thik cha`

These can be handled without sending a request to the LLM.

**Why?** Testing showed filler messages could complete at approximately **0.06 seconds / 0 tokens**, while a real agentic turn typically required approximately **3–4 seconds / 700–1700 tokens**. This reduces unnecessary latency and API usage.

**Loop Breaking**

The Turn Handler also tracks repeated filler messages. This prevents conversations from becoming stuck in patterns such as:

```
User: okay
Bot: ...
User: okay
Bot: ...
User: okay
Bot: ...
```

After a repeated filler streak, the system breaks the loop and prompts the user toward an actual tourism-related request.

### 5. Redis Conversation Memory

Redis is used for lightweight conversation state. Current uses include:

- Conversation history/context
- Filler-message counter

This allows the system to maintain conversational context without requiring a larger orchestration framework.

### 6. Budget Estimation

Yatra includes a `calculate_budget_estimate` tool. The tool deliberately does not perform complex arithmetic itself. The corpus contains budget information with inconsistent formats:

- `NPR X–Y per day`
- `NPR X–Y for the full trip`

Automatically multiplying values could therefore produce confidently incorrect results. Instead, the tool provides the raw estimate and requested duration to the LLM, allowing the model to reason over the available information. Budget estimates should therefore be treated as approximate rather than authoritative calculations.

### 7. Configuration Consolidation

Model names, thresholds, and system limits are centralized in `app/configs/agent_config.py`. For example, instead of repeatedly defining `"openai/gpt-oss-20b"` and other magic numbers throughout the project, configuration is kept in one place. This improves maintainability, consistency, configuration management, and future model changes.

## Technology Stack

| Component | Technology |
| --- | --- |
| Backend | FastAPI |
| Environment / Package Manager | uv |
| LLM | Groq — `openai/gpt-oss-20b` |
| Agent Tool Calling | Groq Function Calling |
| Dense Vector Store | Chroma |
| Sparse Retrieval | BM25 |
| Embeddings | `bge-base-en-v1.5` |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Conversation Memory | Redis |
| Language | Python |

## Project Structure

The project follows a layered architecture with routers, services, and repositories/components separated by responsibility.

```
.
├── app/
│   ├── routers/              # FastAPI routes (chat, debug)
│   ├── services/             # chat_service (turn handler), token_logger
│   ├── agent/
│   │   ├── nodes/            # stage1_rules, stage2_classifier, loop_breaker
│   │   ├── tools/            # 6 agent tools + destination_resolver
│   │   └── schemas/          # state schema
│   ├── components/           # groq client, redis client/session store
│   ├── configs/              # agent_config.py, retrieval_config.py
│   ├── core/                 # settings, exceptions, logging
│   ├── data_ingestion/       # loader, section-based chunker
│   ├── db/chroma/            # chroma client
│   ├── middleware/           # error_handler, request_id
│   ├── models/               # destination, conversation models
│   ├── repositories/         # destination, conversation repos
│   ├── retrieval/            # hybrid_retriever, bm25_index, embeddings, fusion
│   ├── schemas/              # chat, retrieval, token_usage schemas
│   └── utils/                # logger
├── data/
│   ├── corpus/               # 10 destination Markdown documents
│   ├── raw/                  # 10 destination JSON documents
│   └── chroma_db/            # persisted Chroma store
├── eval/
│   ├── cases/                # eval case sets (JSONL)
│   ├── run_eval.py           # functional eval runner
│   ├── faithfulness_judge.py # LLM faithfulness evaluator
│   └── results*.jsonl        # recorded results
├── scripts/                  # build_index, eval, seed, test scripts
├── tests/                    # unit + integration tests
├── docker/                   # Dockerfile.api, docker-compose (Redis)
├── ui/                       # React + Vite chat frontend
├── README.md
├── pyproject.toml
└── uv.lock
```

The architecture follows a Router → Service → Repository-style separation adapted for a single-tenant RAG application. It intentionally avoids unnecessary multi-tenant or enterprise orchestration complexity because the current corpus contains only 10 destinations.

## Data

Yatra currently covers 10 curated Nepal destinations consisting of:

- Trekking destinations
- Heritage towns
- Pilgrimage destinations

The corpus is currently designed for domestic tourism. The documents contain information such as:

- Overview
- Location
- Province
- Highlights
- Difficulty
- Duration
- Altitude
- Budget
- Permits
- Best time to visit
- Other destination-specific information

The current corpus is intentionally limited in size so the project can focus on engineering quality, retrieval behavior, agentic architecture, and evaluation rather than simply scaling the number of documents.

## Evaluation

Yatra includes a self-built evaluation harness designed to measure both retrieval quality and end-to-end system behavior. The evaluation is inspired by RAGAS-style evaluation concepts but is implemented directly rather than adopting the RAGAS library. This was chosen because the project does not use a LangChain-shaped evaluation pipeline and already has direct access to the retrieval and agent components.

### Retrieval Evaluation

Retrieval was evaluated against 7 labeled test cases using:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

**Results**

| K | Precision | Recall | MRR |
| --- | --- | --- | --- |
| 3 | 0.905 | 0.952 | 1.000 |
| 5 | 0.829 | 0.952 | 1.000 |
| 10 | 0.629 | 0.952 | 1.000 |

The correct destination was retrieved at rank 1 in every test case. The decrease in precision at larger K is expected because retrieving more documents also introduces more irrelevant chunks.

### Functional Evaluation

The end-to-end functional evaluation contains 12 cases covering:

- Factual questions
- Comparisons
- Destination discovery
- Budget questions
- Out-of-scope requests ×3
- Adversarial input
- Conflicting-source scenario
- Filler conversation

**Result:** 12 / 12 completed successfully, 0 errors.

This validates that the major system paths execute successfully across the current evaluation set.

### Faithfulness Evaluation

Yatra also includes an LLM-based faithfulness judge. The current result was **3 / 8** in-scope cases were judged fully faithful. One likely genuine hallucination was identified: the Rara Lake response included details about rhododendrons and crowding that were not present in the source corpus.

However, the current faithfulness result should not be interpreted as a direct measurement of the agent's true faithfulness.

**Evaluation Harness Limitation**

The faithfulness evaluator currently re-retrieves context independently instead of capturing the exact chunks actually retrieved and supplied to the agent. Therefore:

```
Agent retrieved context                Faithfulness judge retrieved context
        ≠
```

This can result in inconsistent judgments. Some cases flagged as unfaithful were traced to this evaluation-design limitation rather than an actual unsupported statement in the agent's response. This limitation is intentionally documented rather than presenting the evaluation score as more reliable than it currently is.

## Known Limitations

1. **Faithfulness Context Capture** — The faithfulness judge does not currently receive the exact context used by the agent; instead it performs its own retrieval. This can produce inconsistent evaluation results. *Planned fix:* capture the actual retrieved chunks during each agent turn and pass those exact chunks to the faithfulness judge.

2. **Approximate Budget Calculations** — Budget information in the corpus is not consistently structured. Some values represent daily costs while others represent complete-trip estimates. The final budget answer is therefore an LLM-based estimate rather than independently verified arithmetic.

3. **Variable Latency** — Typical agentic requests take approximately 3–4 seconds, but latency can increase significantly, reaching approximately 30 seconds under Groq API rate limiting on the current tier.

4. **Research-Level Source Verification** — The current corpus contains single-pass research estimates for some budget figures, permit information, and destination details. These have not yet been comprehensively cross-verified against official sources. This is documented intentionally as a current data-quality limitation.

5. **Domestic Tourism Only** — The current system is designed for Nepal domestic travelers. It does not currently implement foreign-tourist pricing, visa requirements, or international traveler-specific logic. This is a deliberate scope decision rather than an unfinished feature.

6. **Limited Corpus** — The current system contains only 10 destinations. The architecture is designed as a production-style demonstration rather than a complete Nepal tourism information platform.

## Future Work

1. **Fix Faithfulness Evaluation** — Capture the exact retrieval context used by the agent:

```
User Query
    ↓
Agent
    ↓
Retriever
    ↓
Retrieved Chunks
    ↓
Agent Response
    ↓
Faithfulness Judge
```

instead of:

```
User Query         User Query
    ↓                  ↓
Agent             Independent Retrieval
    ↓                  ↓
Response          Faithfulness Judge
```

This will make faithfulness evaluation substantially more meaningful.

2. **SSE Streaming** — Add Server-Sent Events (SSE) streaming to the `/chat` endpoint:

```
User → FastAPI → Agent → Streaming response → Client receives tokens incrementally
```

This will improve perceived response latency.

3. **Incremental Document Ingestion** — Add a document ingestion/update endpoint that can update one destination document, re-chunk it, re-generate embeddings, update Chroma and BM25 indexes, and avoid rebuilding the entire corpus. For example, `PUT /destinations/{destination_id}` would update only the affected destination.

4. **Expand Evaluation Dataset** — The current evaluation set contains 12 functional cases and 7 labeled retrieval cases. Future evaluation should expand coverage across more destinations, more query formulations, multi-hop questions, more adversarial queries, retrieval edge cases, budget questions, comparison questions, conversation-context questions, and tool-selection behavior.

5. **Nepali Language Support** — The current corpus is English-first. Future versions can support English, Romanized Nepali, and Devanagari Nepali. This will require evaluating multilingual embeddings, query normalization, cross-language retrieval, Nepali response generation, and Romanized Nepali handling.

## Why This Project?

Yatra is designed to demonstrate practical AI engineering rather than simply calling an LLM API. The project combines:

```
                 ┌────────────────────┐
                 │      FastAPI        │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │    Turn Handler    │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │    Agent Loop      │
                 └─────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
       Tools          Hybrid RAG        Redis
          │                │                │
          │        ┌───────┴───────┐        │
          │        │               │        │
          │      Chroma           BM25      │
          │        │               │        │
          └────────┴───────┬───────┴────────┘
                           │
                           ▼
                    Final Response
```

The focus is on demonstrating engineering decisions such as:

- When to use an LLM
- When not to use an LLM
- How an agent selects tools
- How hybrid retrieval improves search
- How chunking affects retrieval
- How conversation state is maintained
- How to evaluate retrieval
- How to evaluate generated answers
- How to identify evaluation-system weaknesses
- How to document real limitations

## Getting Started

### Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/)
- Redis (or `docker compose up -d` in `docker/`)
- A Groq API key

### Setup

1. Install dependencies:

```bash
uv sync
```

2. Configure environment:

```bash
cp app/core/.env.example app/core/.env
# set GROQ_API_KEY in app/core/.env
```

3. Start Redis (optional if you already have one running):

```bash
docker compose -f docker/docker-compose.yml up -d
```

4. Build the retrieval index:

```bash
uv run python scripts/build_index.py
```

5. Run the API:

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`, with the `/health` health check and `/api/v1/chat` chat endpoint.

### Running the UI (optional)

```bash
cd ui
npm install
npm run dev
```

### Running Evaluations

```bash
uv run python scripts/eval_retrieval.py      # retrieval metrics
uv run python scripts/eval_faithfulness.py   # LLM faithfulness judge
```

## Current Status

| Area | Status |
| --- | --- |
| FastAPI backend | ✅ |
| Agent loop | ✅ |
| Groq function calling | ✅ |
| Dynamic tool selection | ✅ |
| Dense retrieval | ✅ |
| BM25 retrieval | ✅ |
| RRF fusion | ✅ |
| Section-based chunking | ✅ |
| Redis conversation memory | ✅ |
| Cost-aware filler handling | ✅ |
| Loop breaking | ✅ |
| Retrieval evaluation | ✅ |
| Functional evaluation | ✅ |
| LLM faithfulness evaluation | ⚠️ Needs improved context capture |
| SSE streaming | 🔲 Planned |
| Incremental ingestion | 🔲 Planned |
| Expanded evaluation set | 🔲 Planned |
| Nepali language support | 🔲 Planned |

## Conclusion

Yatra is a production-style agentic RAG assistant for Nepal domestic tourism built around a deliberately small corpus.

Rather than hiding limitations behind a single benchmark number, the project emphasizes measurable behavior and engineering trade-offs:

- Hybrid retrieval instead of dense-only search
- Section-aware chunking instead of arbitrary splitting
- Dynamic tool calling instead of a fixed RAG pipeline
- Cost-aware handling of simple conversational turns
- Redis-based conversation state
- Explicit retrieval and functional evaluation
- LLM-based faithfulness evaluation
- Documented evaluation weaknesses
- Documented data-quality limitations

The current implementation is intentionally scoped to 10 destinations while providing a foundation that can be extended with streaming, incremental ingestion, larger evaluation datasets, and multilingual support.