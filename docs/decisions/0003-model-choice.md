# ADR 0003: `openai/gpt-oss-20b` as the agent model

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

The agent needs reliable tool calling, short answers under a token cap, and
predictable cost/latency on Groq. It also needs to stream SSE tokens. The
classifier runs on a separate small model, and the eval judge must be a
different model from the generator.

## Decision

Use `openai/gpt-oss-20b` with `reasoning_effort=low` for the agent, and
`llama-3.1-8b-instant` for the stage-2 router/classifier.

## Consequences

- Reasoning tokens count against `max_tokens` on gpt-oss, so the answer cap is
  2000 and `reasoning_effort=low`; without that the final round could emit
  zero content tokens (the empty-stream bug, now guarded and tested).
- A normal question costs one cheap router call plus the tool loop, instead of
  paying reasoning-model latency just to classify the turn.
- `judge_model` stays `llama-3.3-70b-versatile` so the judge does not share the
  generator's blind spots. Model IDs are settings, not scattered constants.
