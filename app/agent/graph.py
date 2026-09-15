"""
Turn-handler graph: Stage 1 -> Stage 2 -> loop-break -> agent loop.

This is how a single user turn is orchestrated. A `Turn` is threaded through
a linear pipeline of async node functions; each node either short-circuits
(regex filler, LLM-classified filler/unclear) or hands the turn to the next
node. The agent node runs the full Groq tool-calling loop.

`run()` drives the pipeline for a JSON reply, `run_stream()` drives the same
pipeline for SSE: identical wiring, so the two call paths cannot drift apart.
The graph is deliberately framework-free — the orchestration is a small class
you can read in one pass. Nodes are constructor-injected with module-level
defaults, so tests can stub IO without touching Redis or the network.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field

from app.agent.agent import run_agent, run_agent_stream
from app.agent.nodes.loop_breaker_node import (
    check_loop_break,
    increment_filler_count,
    reset_filler_count,
)
from app.agent.nodes.stage1_rules_node import classify_stage1, get_templated_reply
from app.agent.nodes.stage2_classifier_node import classify_stage2
from app.components.redis.session_store import append_message, get_history
from app.configs.agent_config import DESTINATION_NAMES

_unclear_destinations = ", ".join(DESTINATION_NAMES)
_unclear_count = len(DESTINATION_NAMES)
UNCLEAR_REPLY = (
    "I'm not sure I caught that! I can help with treks, permits, budgets, culture, or "
    f"best time to visit for {_unclear_count} Nepal destinations — {_unclear_destinations}. "
    "What would you like to know?"
)


@dataclass
class Turn:
    query: str
    sender: str
    category: str | None = None
    reply: str | None = None
    retrieved_context: list[str] = field(default_factory=list)
    stage: str | None = None
    short_circuited: bool = False


def _get_last_assistant_message(sender: str) -> str:
    """Looks back a couple of messages (not just the last one) in case the
    most recent stored message happens to be the user's own prior turn."""
    history = get_history(sender, limit=2)
    for msg in reversed(history):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


async def _append_turn_messages(sender: str, user_text: str, reply: str) -> None:
    await asyncio.to_thread(append_message, sender, "user", user_text)
    await asyncio.to_thread(append_message, sender, "assistant", reply)


# --- default nodes (module-level so the class can reference them) ---------


async def _stage1_node(turn: Turn) -> None:
    """Regex fast path for pure filler; short-circuits or passes through."""
    category = classify_stage1(turn.query)
    if not category:
        return
    turn.stage = "stage1"
    turn.category = category
    await asyncio.to_thread(increment_filler_count, turn.sender)
    loop_break_msg = await asyncio.to_thread(check_loop_break, turn.sender)
    turn.reply = loop_break_msg or get_templated_reply(category)
    turn.short_circuited = True
    await _append_turn_messages(turn.sender, turn.query, turn.reply)


async def _stage2_node(turn: Turn, last_assistant_message: str) -> None:
    """LLM classifier for what Stage 1's regex did not catch."""
    turn.stage = "stage2"
    turn.category = await classify_stage2(
        turn.query, last_assistant_message=last_assistant_message
    )

    if turn.category == "filler":
        await asyncio.to_thread(increment_filler_count, turn.sender)
        loop_break_msg = await asyncio.to_thread(check_loop_break, turn.sender)
        turn.reply = loop_break_msg or get_templated_reply("acknowledgment")
        turn.short_circuited = True
        await _append_turn_messages(turn.sender, turn.query, turn.reply)
    elif turn.category == "unclear":
        turn.reply = UNCLEAR_REPLY
        turn.short_circuited = True
        await _append_turn_messages(turn.sender, turn.query, turn.reply)
    else:
        # A real question ends any filler streak.
        await asyncio.to_thread(reset_filler_count, turn.sender)


async def _agent_node(turn: Turn) -> None:
    """Full agent loop — the only node that spends real LLM tokens."""
    turn.stage = "agent"
    result = await run_agent(turn.query, sender=turn.sender)
    turn.reply = result["answer"]
    turn.retrieved_context = result["retrieved_context"]
    await _append_turn_messages(turn.sender, turn.query, turn.reply)


async def _agent_stream_node(turn: Turn) -> AsyncGenerator[str]:
    """Agent loop, streaming: yields text, stores answer + context on the turn."""
    turn.stage = "agent"
    retrieved_context: list[str] = []
    stream = run_agent_stream(
        turn.query, sender=turn.sender, retrieved_context=retrieved_context
    )
    full_answer = ""
    async for chunk in stream:
        full_answer += chunk
        yield chunk
    turn.reply = full_answer
    turn.retrieved_context = retrieved_context
    await _append_turn_messages(turn.sender, turn.query, full_answer)


async def _last_assistant_message_node(sender: str) -> str:
    return await asyncio.to_thread(_get_last_assistant_message, sender)


# --- the graph ------------------------------------------------------------


Node = Callable[[Turn], Awaitable[None]]
Stage2Node = Callable[[Turn, str], Awaitable[None]]
StreamNode = Callable[[Turn], AsyncGenerator[str]]
LastAssistNode = Callable[[str], Awaitable[str]]


class TurnGraph:
    """Linear turn pipeline: stage1 -> stage2 -> agent (or short-circuit).

    Both entry points work on a caller-provided `Turn` (async generators
    cannot return values, so the caller owns the Turn and reads its fields
    after the pipeline completes).
    """

    def __init__(
        self,
        *,
        stage1: Node = _stage1_node,
        stage2: Stage2Node = _stage2_node,
        agent: Node = _agent_node,
        agent_stream: StreamNode = _agent_stream_node,
        last_assistant_message: LastAssistNode = _last_assistant_message_node,
    ) -> None:
        self._stage1 = stage1
        self._stage2 = stage2
        self._agent = agent
        self._agent_stream = agent_stream
        self._last_assistant_message = last_assistant_message

    async def run(self, turn: Turn) -> Turn:
        await self._stage1(turn)
        if turn.short_circuited:
            return turn
        last_assistant_message = await self._last_assistant_message(turn.sender)
        await self._stage2(turn, last_assistant_message)
        if turn.short_circuited:
            return turn
        await self._agent(turn)
        return turn

    async def run_stream(self, turn: Turn) -> AsyncGenerator[str]:
        await self._stage1(turn)
        if not turn.short_circuited:
            last_assistant_message = await self._last_assistant_message(turn.sender)
            await self._stage2(turn, last_assistant_message)
        if turn.short_circuited:
            yield turn.reply
            return
        async for chunk in self._agent_stream(turn):
            yield chunk