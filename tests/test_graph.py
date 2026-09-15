"""Unit tests for the TurnGraph wiring (no network, no Redis)."""

import asyncio
from collections.abc import AsyncGenerator

from app.agent.graph import Turn, TurnGraph


async def _noop_stage1(turn: Turn) -> None:
    """Stage 1 passes through — no short-circuit."""


async def _shortcircuit_stage1(turn: Turn) -> None:
    turn.stage = "stage1"
    turn.category = "greeting"
    turn.reply = "Hi!"
    turn.short_circuited = True


async def _filler_stage2(turn: Turn, last: str) -> None:
    turn.stage = "stage2"
    turn.category = "filler"
    turn.reply = "Acknowledged."
    turn.short_circuited = True


async def _new_question_stage2(turn: Turn, last: str) -> None:
    turn.stage = "stage2"
    turn.category = "new_question"


async def _agent_stub(turn: Turn) -> None:
    turn.stage = "agent"
    turn.reply = "Full answer"
    turn.retrieved_context = ["chunk one", "chunk two"]


async def _agent_stream_stub(turn: Turn) -> AsyncGenerator[str]:
    turn.stage = "agent"
    for part in ("Full ", "answer"):
        yield part
    turn.reply = "Full answer"
    turn.retrieved_context = ["chunk one"]


async def _last_msg(sender: str) -> str:
    return "earlier assistant reply"


# --- tests ----------------------------------------------------------------


def test_agent_path_sets_answer_and_context() -> None:
    graph = TurnGraph(
        stage1=_noop_stage1,
        stage2=_new_question_stage2,
        last_assistant_message=_last_msg,
        agent=_agent_stub,
    )
    turn = asyncio.run(graph.run(Turn(query="any", sender="user-1")))

    assert turn.stage == "agent"
    assert turn.category == "new_question"
    assert turn.reply == "Full answer"
    assert turn.retrieved_context == ["chunk one", "chunk two"]
    assert not turn.short_circuited


def test_short_circuit_skips_agent() -> None:
    called = {"agent": False}

    async def _agent(turn: Turn) -> None:
        called["agent"] = True

    graph = TurnGraph(
        stage1=_shortcircuit_stage1,
        last_assistant_message=_last_msg,
        agent=_agent,
    )
    turn = asyncio.run(graph.run(Turn(query="hi", sender="user-1")))

    assert turn.short_circuited
    assert turn.reply == "Hi!"
    assert turn.category == "greeting"
    assert called["agent"] is False


def test_stream_yields_chunks_and_populates_turn() -> None:
    graph = TurnGraph(
        stage1=_noop_stage1,
        stage2=_new_question_stage2,
        last_assistant_message=_last_msg,
        agent_stream=_agent_stream_stub,
    )

    async def _collect() -> tuple[list[str], Turn]:
        turn = Turn(query="q", sender="user-1")
        chunks: list[str] = []
        async for chunk in graph.run_stream(turn):
            chunks.append(chunk)
        return chunks, turn

    chunks, turn = asyncio.run(_collect())
    assert chunks == ["Full ", "answer"]
    assert turn.reply == "Full answer"
    assert turn.retrieved_context == ["chunk one"]
    assert not turn.short_circuited


def test_stream_short_circuit_yields_single_reply() -> None:
    graph = TurnGraph(
        stage1=_shortcircuit_stage1,
        last_assistant_message=_last_msg,
    )

    async def _collect() -> tuple[list[str], Turn]:
        turn = Turn(query="hi", sender="user-1")
        chunks: list[str] = []
        async for chunk in graph.run_stream(turn):
            chunks.append(chunk)
        return chunks, turn

    chunks, turn = asyncio.run(_collect())
    assert chunks == ["Hi!"]
    assert turn.short_circuited
    assert turn.stage == "stage1"


def test_filler_stage2_short_circuits() -> None:
    graph = TurnGraph(
        stage1=_noop_stage1,
        stage2=_filler_stage2,
        last_assistant_message=_last_msg,
    )
    turn = asyncio.run(graph.run(Turn(query="thanks", sender="user-1")))

    assert turn.short_circuited
    assert turn.stage == "stage2"
    assert turn.category == "filler"
    assert turn.reply == "Acknowledged."
