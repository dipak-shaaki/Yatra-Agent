"""
The real entry point for a user turn: Stage 1 -> Stage 2 -> loop-break ->
(only if none of those short-circuit) the full agent loop.
"""
from app.agent.agent import run_agent
from app.agent.nodes.stage1_rules_node import classify_stage1, get_templated_reply
from app.agent.nodes.stage2_classifier_node import classify_stage2
from app.agent.nodes.loop_breaker_node import increment_filler_count, reset_filler_count, check_loop_break
from app.components.redis.session_store import append_message
from app.utils.logger import log_event


async def handle_turn(query: str, sender: str) -> str:
    stage1_category = classify_stage1(query)

    if stage1_category:
        increment_filler_count(sender)
        loop_break_msg = check_loop_break(sender)
        if loop_break_msg:
            log_event("loop_break_triggered", sender=sender)
        reply = loop_break_msg or get_templated_reply(stage1_category)
        log_event("turn_handled", stage="stage1", category=stage1_category, llm_calls=0, loop_break=bool(loop_break_msg))
        append_message(sender, "user", query)
        append_message(sender, "assistant", reply)
        return reply

    stage2_category = classify_stage2(query)

    if stage2_category == "filler":
        log_event("turn_handled", stage="stage2", category="filler", llm_calls=1)
        increment_filler_count(sender)
        loop_break_msg = check_loop_break(sender)
        reply = loop_break_msg or "Got it! Let me know if you have a question about Nepal travel."
        append_message(sender, "user", query)
        append_message(sender, "assistant", reply)
        return reply

    if stage2_category == "unclear":
        log_event("turn_handled", stage="stage2", category="unclear", llm_calls=1)
        reply = "I didn't quite catch that — could you rephrase your question about Nepal travel?"
        append_message(sender, "user", query)
        append_message(sender, "assistant", reply)
        return reply

    # Only real, understood questions reach the full agent
    reset_filler_count(sender)
    log_event("turn_handled", stage="agent", category=stage2_category, llm_calls="full_agent")
    answer = await run_agent(query, sender=sender)
    append_message(sender, "user", query)
    append_message(sender, "assistant", answer)
    return answer

