# CV & Interview Preparation — Yatra

Positioning line: **"Agentic RAG with tool calling and staged routing"** —
one LLM agent loop plus a rule/classifier turn router. Do **not** call it
"multi-agent": there is one agent; the "stages" are a classifier and a
tool-calling loop, not collaborating agents.

---

## Problem → fix bullets (with real before/after numbers)

**1. Silent BM25 index staleness across workers**
Every document upsert bumped a Redis corpus-version key, but the version
reader called `.decode()` on an already-decoded string, so the read always
raised and fell back to `"0"` — every worker's in-memory BM25 index stayed
cached against the same version, and upserts were invisible to them until
restart.
*Fix:* one-line corrected read + cache keyed on `(corpus_dir, version)`
with automatic rebuild on change.
*Before:* upserts silently stale in all non-handling workers; *after:*
rebuild happens on the next query in every worker.

**2. Measuring the wrong thing in retrieval eval**
The first retrieval eval scored every case with labeled chunks (47/60) —
including discovery/budget questions whose real path is a metadata tool
(filter/budget), not hybrid search. The gates failed on numbers that were
an artifact of case selection.
*Fix:* score only cases whose expected path is `search_destinations`;
gates calibrated with headroom (Precision@5 ≥ 0.15, Recall@5 ≥ 0.85,
MRR@5 ≥ 0.70) and a comment warning not to "fix" the precision ceiling
(most cases label one chunk, so P caps near 1/5).
*Before:* hybrid scored P=0.204 / R=0.792 / MRR=0.723 (misleading);
*after:* hybrid P=0.193 / R=0.955 / MRR=0.751 over 176 scored cases
(95% bootstrap CIs reported for each metric). The ablation keeps the
trade-off visible rather than assumed: dense R=0.986 / MRR=0.800,
BM25 R=0.812 / MRR=0.617. Hybrid remains the served path and the only gated
one; the gate runs in CI on every push.

**3. Empty streamed answers (reasoning tokens vs max_tokens)**
The reasoning model (`openai/gpt-oss-20b`) spends reasoning tokens from the
same `max_tokens` budget; with caps of 400/300 the final round could
produce zero content tokens — an empty SSE stream with a 200 response.
*Fix:* raised the answer cap to 2000 with `reasoning_effort=low`, moved
classification to a small `router_model` (own 400-token cap), and added an
empty-content guard that returns an explicit error reply + log event; covered
by a unit test with a stubbed Groq client.
*Before:* intermittent empty answers, no signal; *after:* guarded path
with `agent_empty_content` logging.

**4. Forging other users' conversation history**
`/chat` trusted a client-chosen `sender`; anyone could read or overwrite
another session's history by changing one string.
*Fix:* server-minted session tokens (`POST /session`) as the sender
identity, fixed-window rate limit (30 req/min/token, `Retry-After` on 429)
plus a per-IP limit on token minting so the per-token limit cannot be
bypassed by farming tokens, fail-closed on Redis outage (503, never
anonymous access), admin-token protection on mutating/debug endpoints.

**5. Two wasted LLM calls per question**
The original design spent `check_scope` (guardrail tool) and
`get_conversation_context` (history re-fetch) calls before retrieval —
history was already in the prompt.
*Fix:* scope folded into the stage-2 LLM classifier (one 4-way call:
new_question/filler/unclear/out_of_scope), context tool deleted, regex
fast path answers pure filler with 0 calls, independent tool calls run in
parallel (`asyncio.gather`). Per-turn LLM-call count is now a recorded
metric (mean/max in the eval report).

---

## Prepared interview answers

**Why hybrid retrieval?** Dense embeddings miss exact identifiers ("5,106
m pass", "RAP permit") and BM25 misses paraphrase. On the expanded labeled
set (176 retrieval-routed cases, 95% CIs): dense Recall@5 0.986 / MRR 0.800,
BM25 recall 0.812 / MRR 0.617, fused (RRF, k=60) recall 0.955 / MRR 0.751.
The fusion is score-free rank fusion, so no score normalization between
cosine distance and BM25, and the `--retriever` ablation is committed so the
trade-off stays visible instead of assumed. Numbers from
`python -m scripts.eval_retrieval --retriever <name>`.

**What failed in your eval and what did you change?** Two concrete
failures: (a) case selection — I was scoring metadata-tool questions
against the hybrid retriever, which produced misleading metrics; fixed by
scoring only the retrieval-path cases and keeping an ablation mode.
(b) judging — judging with the generating model inherits its blind spots,
so the judge is a different model; also, cases that capture no context
(metadata tools, short-circuits) are not required to be faithful to a
context they never saw. Remaining retrieval misses are visible in the gate
output (8/176 full misses on hybrid, mostly synthetic budget/altitude
queries) rather than hidden.

**Why an agent instead of a fixed RAG pipeline?** The question type
determines the right retrieval: comparisons need per-section chunk
queries, discovery needs metadata filtering (embeddings blur "easy
trek near Pokhara"), budget questions need arithmetic on stored numeric
ranges — done in code, never improvised by the LLM. A fixed pipeline would
run everything through vector search and hope; the agent picks the tool
per turn (and can call several in parallel), while the router keeps cost
down (0–1 LLM calls for filler/unclear/out-of-scope).

**How do you know the answers are grounded?** The system prompt requires
citations (source file + `last_updated`), hedged corpus language must be
preserved in answers (judge scores `hedged`), and the eval judge checks
faithfulness against exactly the chunks the agent retrieved (captured at
turn time, not re-retrieved). 78 labeled behavioral cases across 9
categories — including refusals (live prices, bookings, medical advice) and
conflicting-source questions — run through `run_eval` + `eval_answers`
with per-category pass criteria, alongside 145 retrieval-only cases.

---

## Known gaps to mention honestly if asked

- No reranker: at 82 chunks, a cross-encoder would add latency for
  marginal gain. On the expanded set hybrid sits between dense and BM25
  (dense ranks best, BM25 alone misses the gate); hybrid remains the served
  path because the ablation is measured, not assumed.
- Baseline "before" numbers for the original 12-case eval were not saved;
  the committed `eval/BASELINE.md` (SHA + settings + CIs) is the documented
  baseline going forward.
- Latency is Groq-bound; measured p50/p95 via `run_eval` + `eval_answers`
  rather than claimed (needs a valid GROQ_API_KEY to run).
- `/metrics` is a per-process JSON snapshot (turns, LLM calls, tokens,
  estimated cost), not a Prometheus endpoint; structured logs still carry
  the per-call detail.
- Deployment: Dockerfile + compose are ready and the image builds, but no
  VM/Caddy HTTPS deployment or live demo link yet.
