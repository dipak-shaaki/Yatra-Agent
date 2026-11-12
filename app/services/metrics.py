"""
Process-local turn metrics for the /metrics endpoint: how many turns ran,
how many LLM calls they cost, and token/cost totals.

Deliberately tiny and dependency-free (no Prometheus client). It is
per-process, so scale-out scrapes should sum across workers; the eval
reports remain the durable record.
"""

import threading
from dataclasses import dataclass, field

from app.core.config import get_settings

_lock = threading.Lock()


@dataclass
class _Metrics:
    turns: int = 0
    errors: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    per_turn_llm_calls: list[int] = field(default_factory=list)


_metrics = _Metrics()


def record_turn(llm_calls: int, failed: bool = False) -> None:
    with _lock:
        _metrics.turns += 1
        _metrics.llm_calls += llm_calls
        _metrics.per_turn_llm_calls.append(llm_calls)
        if failed:
            _metrics.errors += 1


def record_llm_usage(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    """Accumulate token counts; cost uses the configured per-model rates."""
    settings = get_settings()
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    cost = (
        prompt_tokens / 1_000_000 * settings.input_cost_per_million
        + completion_tokens / 1_000_000 * settings.output_cost_per_million
    )
    with _lock:
        _metrics.prompt_tokens += prompt_tokens
        _metrics.completion_tokens += completion_tokens
        _metrics.cost_usd += cost


def metrics_snapshot() -> dict:
    with _lock:
        turns = _metrics.turns
        calls = _metrics.per_turn_llm_calls
        return {
            "turns": turns,
            "errors": _metrics.errors,
            "llm_calls": _metrics.llm_calls,
            "mean_llm_calls_per_turn": (sum(calls) / len(calls)) if calls else 0.0,
            "prompt_tokens": _metrics.prompt_tokens,
            "completion_tokens": _metrics.completion_tokens,
            "total_tokens": _metrics.prompt_tokens + _metrics.completion_tokens,
            "estimated_cost_usd": round(_metrics.cost_usd, 6),
            "cost_per_turn_usd": round(_metrics.cost_usd / turns, 6) if turns else 0.0,
        }
