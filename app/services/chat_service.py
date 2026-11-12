"""
Thin adapter over the TurnGraph: maps a graph run to the reply shape the
router expects, applies the turn-level deadline, records turn metrics and
emits one `turn_handled` log line per turn.

All orchestration (stage1 -> stage2 -> loop-break -> agent) lives in
`app/agent/graph.py`; this module must stay transparent glue.
"""

import asyncio
from collections.abc import AsyncGenerator

from app.agent.graph import Turn, TurnGraph, unclear_reply
from app.core.config import get_settings
from app.services.metrics import record_turn
from app.utils.logger import log_event

_graph = TurnGraph()

TURN_TIMEOUT_REPLY = (
    "Sorry, that took too long to answer. Please try again — "
    "a shorter or more specific question usually helps."
)


def _log_turn(turn: Turn) -> None:
    log_event(
        "turn_handled",
        stage=turn.stage,
        category=turn.category,
        llm_calls=turn.llm_calls,
        failed=turn.failed,
    )
    record_turn(turn.llm_calls, failed=turn.failed)


async def handle_turn(query: str, sender: str) -> dict:
    """Returns {"answer": str, "retrieved_context": list[str], "tools_called":
    list[str], "llm_calls": int}. Short-circuit paths (stage1/stage2
    filler/unclear/out_of_scope) return empty context and tools, since no
    retrieval happened for those."""
    turn = Turn(query=query, sender=sender)
    try:
        async with asyncio.timeout(get_settings().turn_timeout_seconds):
            await _graph.run(turn)
    except TimeoutError:
        log_event("turn_timeout", stage=turn.stage)
        turn.reply = TURN_TIMEOUT_REPLY
        turn.failed = True
    _log_turn(turn)
    return {
        "answer": turn.reply,
        "retrieved_context": turn.retrieved_context,
        "tools_called": turn.tools_called,
        "llm_calls": turn.llm_calls,
        "failed": turn.failed,
    }


async def handle_turn_stream(
    query: str, sender: str, turn_sink: list[Turn] | None = None
) -> AsyncGenerator[str]:
    """Yields answer chunks. The completed Turn is appended to `turn_sink`
    (async generators cannot return values) so the router can emit its
    retrieved_context as a final metadata event."""
    turn = Turn(query=query, sender=sender)
    try:
        async with asyncio.timeout(get_settings().turn_timeout_seconds):
            async for chunk in _graph.run_stream(turn):
                yield chunk
    except TimeoutError:
        log_event("turn_timeout", stage=turn.stage)
        turn.reply = TURN_TIMEOUT_REPLY
        turn.failed = True
        yield TURN_TIMEOUT_REPLY
    _log_turn(turn)
    if turn_sink is not None:
        turn_sink.append(turn)


# Kept importable here for anyone that referenced the constant at this path.
__all__ = ["handle_turn", "handle_turn_stream", "unclear_reply"]
