"""
Thin adapter over the TurnGraph: maps a graph run to the reply shape the
router expects and emits one `turn_handled` log line per turn.

All orchestration (stage1 -> stage2 -> loop-break -> agent) lives in
`app/agent/graph.py`; this module must stay transparent glue.
"""

from collections.abc import AsyncGenerator

from app.agent.graph import Turn, TurnGraph, unclear_reply
from app.utils.logger import log_event

_graph = TurnGraph()


def _log_turn(turn: Turn) -> None:
    log_event(
        "turn_handled",
        stage=turn.stage,
        category=turn.category,
        llm_calls=turn.llm_calls,
    )


async def handle_turn(query: str, sender: str) -> dict:
    """Returns {"answer": str, "retrieved_context": list[str], "tools_called":
    list[str], "llm_calls": int}. Short-circuit paths (stage1/stage2
    filler/unclear/out_of_scope) return empty context and tools, since no
    retrieval happened for those."""
    turn = Turn(query=query, sender=sender)
    await _graph.run(turn)
    _log_turn(turn)
    return {
        "answer": turn.reply,
        "retrieved_context": turn.retrieved_context,
        "tools_called": turn.tools_called,
        "llm_calls": turn.llm_calls,
    }


async def handle_turn_stream(
    query: str, sender: str, turn_sink: list[Turn] | None = None
) -> AsyncGenerator[str]:
    """Yields answer chunks. The completed Turn is appended to `turn_sink`
    (async generators cannot return values) so the router can emit its
    retrieved_context as a final metadata event."""
    turn = Turn(query=query, sender=sender)
    async for chunk in _graph.run_stream(turn):
        yield chunk
    _log_turn(turn)
    if turn_sink is not None:
        turn_sink.append(turn)


# Kept importable here for anyone that referenced the constant at this path.
__all__ = ["handle_turn", "handle_turn_stream", "unclear_reply"]
