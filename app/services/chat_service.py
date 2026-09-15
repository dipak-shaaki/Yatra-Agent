"""
The real entry point for a user turn: Stage 1 -> Stage 2 -> loop-break ->
(only if none of those short-circuit) the full agent loop.

handle_turn() is the non-streaming version (returns a complete string).
handle_turn_stream() is the streaming version (yields chunks) — Turn
Handler's instant replies are yielded as a single chunk; only real agent
turns stream incrementally.
"""
from collections.abc import AsyncGenerator

from app.agent.agent import run_agent, run_agent_stream
from app.agent.nodes.loop_breaker_node import (
    check_loop_break,
    increment_filler_count,
    reset_filler_count,
)
from app.agent.nodes.stage1_rules_node import classify_stage1, get_templated_reply
from app.agent.nodes.stage2_classifier_node import classify_stage2
from app.components.redis.session_store import append_message, get_history
from app.utils.logger import log_event


def _get_last_assistant_message(sender: str) -> str:
    """Looks back a couple of messages (not just the last one) in case the
    most recent stored message happens to be the user's own prior turn."""
    history = get_history(sender, limit=2)
    for msg in reversed(history):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""

async def handle_turn(query: str, sender: str) -> dict:
    """Returns {"answer": str, "retrieved_context": list[str]}. Turn Handler
    short-circuit paths (stage1/stage2 filler/unclear) return empty context,
    since no retrieval happened for those."""
    stage1_category = classify_stage1(query)

    if stage1_category:
        log_event("turn_handled", stage="stage1", category=stage1_category, llm_calls=0)
        increment_filler_count(sender)
        loop_break_msg = check_loop_break(sender)
        reply = loop_break_msg or get_templated_reply(stage1_category)
        append_message(sender, "user", query)
        append_message(sender, "assistant", reply)
        return {"answer": reply, "retrieved_context": []}

    last_assistant_message = _get_last_assistant_message(sender)
    stage2_category = classify_stage2(query, last_assistant_message=last_assistant_message)

    if stage2_category == "filler":
        log_event("turn_handled", stage="stage2", category="filler", llm_calls=1)
        increment_filler_count(sender)
        loop_break_msg = check_loop_break(sender)
        reply = loop_break_msg or get_templated_reply("acknowledgment")
        append_message(sender, "user", query)
        append_message(sender, "assistant", reply)
        return {"answer": reply, "retrieved_context": []}

    if stage2_category == "unclear":
        log_event("turn_handled", stage="stage2", category="unclear", llm_calls=1)
        reply = "I didn't quite catch that — could you rephrase your question about Nepal travel?"
        append_message(sender, "user", query)
        append_message(sender, "assistant", reply)
        return {"answer": reply, "retrieved_context": []}

    reset_filler_count(sender)
    log_event("turn_handled", stage="agent", category=stage2_category, llm_calls="full_agent")
    result = await run_agent(query, sender=sender)
    append_message(sender, "user", query)
    append_message(sender, "assistant", result["answer"])
    return result

async def handle_turn_stream(query: str, sender: str) -> AsyncGenerator[str]:
    stage1_category = classify_stage1(query)

    if stage1_category:
        log_event("turn_handled", stage="stage1", category=stage1_category, llm_calls=0)
        increment_filler_count(sender)
        loop_break_msg = check_loop_break(sender)
        reply = loop_break_msg or get_templated_reply(stage1_category)
        append_message(sender, "user", query)
        append_message(sender, "assistant", reply)
        yield reply
        return

    last_assistant_message = _get_last_assistant_message(sender)
    stage2_category = classify_stage2(query, last_assistant_message=last_assistant_message)

    if stage2_category == "filler":
        log_event("turn_handled", stage="stage2", category="filler", llm_calls=1)
        increment_filler_count(sender)
        loop_break_msg = check_loop_break(sender)
        reply = loop_break_msg or get_templated_reply("acknowledgment")
        append_message(sender, "user", query)
        append_message(sender, "assistant", reply)
        yield reply
        return

    if stage2_category == "unclear":
        log_event("turn_handled", stage="stage2", category="unclear", llm_calls=1)
        reply = (
            "I'm not sure I caught that! I can help with treks, permits, budgets, culture, or "
            "best time to visit for 10 Nepal destinations — Manaslu Circuit, Annapurna Base Camp, "
            "Mardi Himal, Kori, Badimalika, Bandipur, Panauti, Gorkha, Rara Lake, and Tansen/Palpa. "
            "What would you like to know?"
        )
        append_message(sender, "user", query)
        append_message(sender, "assistant", reply)
        yield reply
        return

    reset_filler_count(sender)
    log_event("turn_handled", stage="agent", category=stage2_category, llm_calls="full_agent")

    full_answer = ""
    async for chunk in run_agent_stream(query, sender=sender):
        full_answer += chunk
        yield chunk

    append_message(sender, "user", query)
    append_message(sender, "assistant", full_answer)