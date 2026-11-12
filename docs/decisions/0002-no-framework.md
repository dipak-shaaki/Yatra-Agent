# ADR 0002: No agent/RAG framework

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

Frameworks such as LangChain/LangGraph or LlamaIndex would provide the tool
loop, routing and tracing out of the box. The whole turn pipeline here is
four steps (regex fast path → 4-way classifier → loop break → tool loop), and
the failure modes that matter are provider errors, timeouts, stale caches and
observability.

## Decision

Implement the turn graph and the Groq tool loop directly (`app/agent/`), with
constructor-injected nodes so tests can stub IO.

## Consequences

- The orchestration is a small class readable in one pass; no framework
  upgrade churn and no hidden prompt/format coupling.
- Nothing to learn for a reviewer, and the SSE contract is ours (typed
  `token` / `meta` / `error` / `done` events).
- We own what a framework would give us: retries/timeouts (delegated to the
  Groq SDK plus an `asyncio.timeout` per call/turn), tracing (structured JSON
  logs) and concurrency caps (semaphores). If the graph grows branching
  multi-agent state, revisit a framework.
