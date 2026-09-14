"""
The agent loop: send messages + tool schemas to Groq, execute whichever
tool(s) it calls, feed results back as tool messages, and repeat until
the model responds with plain content instead of another tool call.

This is what makes Yatra agentic rather than a fixed retrieve-then-generate
pipeline — the model decides, per turn, how many tools to call and in what
order (e.g. check_scope -> search_destinations -> final answer).

Two entry points: run_agent() (non-streaming) and run_agent_stream()
(yields text chunks as they arrive). The /chat endpoint picks between them
via its `stream=true|false` query parameter.
"""
import json
import time
from typing import AsyncGenerator

from app.agent.tool_registry import call_tool, get_tool_schemas
from app.components.groq.client import get_groq_client
from app.components.redis.session_store import get_history
from app.configs.agent_config import AGENT_MODEL, MAX_TOOL_ITERATIONS, CONVERSATION_HISTORY_LIMIT , FINAL_ANSWER_MAX_TOKENS
from app.utils.logger import log_event

SYSTEM_PROMPT = """You are Yatra, a helpful assistant for Nepali domestic tourists, covering \
exactly 10 destinations: Manaslu Circuit, Annapurna Base Camp, Mardi Himal, Kori, Badimalika, \
Bandipur, Panauti, Gorkha, Rara Lake, and Tansen/Palpa.

Always call check_scope first on any real question to confirm it's within your domain. If in \
scope, use search_destinations (or other tools) to retrieve real information before answering — \
never answer from memory alone. If out of scope, politely explain you can only help with these \
10 destinations.

Response style:
- Be concise. Answer only what was asked.
- DEFAULT to plain sentences or a short bullet list. Example — for "best time to visit Rara \
Lake?", write: "Spring (March-May) and autumn (September-November) are best — clear skies and \
good visibility. Avoid monsoon (June-August, rain) and winter (heavy snow)." NOT a table.
- ONLY use a table when comparing 2+ destinations side by side, or listing 4+ structured items \
(e.g. a multi-day budget breakdown). A single destination's single attribute (best time, \
difficulty, one permit) never needs a table.
- End with ONE brief, natural follow-up question when it fits — skip it if the answer is already \
a direct yes/no or the conversation seems to be wrapping up."""

async def run_agent(user_query: str, sender: str) -> dict:
    """
    Returns {"answer": str, "retrieved_context": list[str]} — the context
    list captures every chunk actually retrieved during this turn, so eval
    scripts can judge faithfulness against what the agent really saw,
    not a fresh, potentially different retrieval.
    """
    turn_start = time.perf_counter()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_history(sender, limit=CONVERSATION_HISTORY_LIMIT))
    messages.append({"role": "user", "content": user_query})

    client = get_groq_client()
    retrieved_context: list[str] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        llm_start = time.perf_counter()
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=get_tool_schemas(),
            tool_choice="auto",
            max_tokens=FINAL_ANSWER_MAX_TOKENS,
        )
        llm_elapsed_s = f"{time.perf_counter() - llm_start:.2f}"

        usage = response.usage
        log_event(
            "llm_call", iteration=iteration + 1, time_secs=llm_elapsed_s,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            total_s = f"{time.perf_counter() - turn_start:.2f}"
            log_event("agent_final_answer", iterations=iteration + 1, total_time_secs=total_s)
            return {"answer": msg.content, "retrieved_context": retrieved_context}

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_start = time.perf_counter()
            try:
                args = json.loads(tc.function.arguments)
                if tool_name == "get_conversation_context":
                    args["sender"] = sender
                result = call_tool(tool_name, **args)
                elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
                log_event("tool_called", tool=tool_name, args=args, time_secs=elapsed_s)

                # Capture retrieved text for faithfulness judging, regardless
                # of which retrieval-backed tool produced it.
                if tool_name == "search_destinations" and isinstance(result, list):
                    retrieved_context.extend(chunk.get("text", "") for chunk in result)
                elif tool_name == "compare_destinations" and isinstance(result, dict):
                    for sections in result.values():
                        if isinstance(sections, dict):
                            retrieved_context.extend(sections.values())

            except Exception as e:
                elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
                result = {"error": str(e)}
                log_event("tool_call_failed", tool=tool_name, error=str(e), time_secs=elapsed_s)

            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": json.dumps(result),
            })

    total_s = f"{time.perf_counter() - turn_start:.2f}"
    log_event("agent_max_iterations_hit", max_iterations=MAX_TOOL_ITERATIONS, total_time_secs=total_s)
    return {"answer": "Sorry, I'm having trouble processing that request right now. Could you rephrase it?", "retrieved_context": retrieved_context}

async def run_agent_stream(user_query: str, sender: str) -> AsyncGenerator[str, None]:
    """
    Yields text chunks as they're generated. Tool-call rounds happen
    silently between yields — only final-answer generation actually
    streams text to the caller.
    """
    turn_start = time.perf_counter()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_history(sender, limit=CONVERSATION_HISTORY_LIMIT))
    messages.append({"role": "user", "content": user_query})

    client = get_groq_client()

    for iteration in range(MAX_TOOL_ITERATIONS):
        llm_start = time.perf_counter()
        stream = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=get_tool_schemas(),
            tool_choice="auto",
            stream=True,
        )

        content_buffer = ""
        tool_calls_buffer: dict[int, dict] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                content_buffer += delta.content
                yield delta.content

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_buffer[idx]["id"] = tc_delta.id
                    if tc_delta.function.name:
                        tool_calls_buffer[idx]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments

        llm_elapsed_s = f"{time.perf_counter() - llm_start:.2f}"
        log_event("llm_call_stream", iteration=iteration + 1, time_secs=llm_elapsed_s)

        if not tool_calls_buffer:
            total_s = f"{time.perf_counter() - turn_start:.2f}"
            log_event("agent_final_answer", iterations=iteration + 1, total_time_secs=total_s)
            return

        messages.append({
            "role": "assistant",
            "content": content_buffer or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls_buffer.values()
            ],
        })

        for tc in tool_calls_buffer.values():
            tool_start = time.perf_counter()
            try:
                args = json.loads(tc["arguments"])
                if tc["name"] == "get_conversation_context":
                    args["sender"] = sender
                result = call_tool(tc["name"], **args)
                elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
                log_event("tool_called", tool=tc["name"], args=args, time_secs=elapsed_s)
            except Exception as e:
                elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
                result = {"error": str(e)}
                log_event("tool_call_failed", tool=tc["name"], error=str(e), time_secs=elapsed_s)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result),
            })

    total_s = f"{time.perf_counter() - turn_start:.2f}"
    log_event("agent_max_iterations_hit", max_iterations=MAX_TOOL_ITERATIONS, total_time_secs=total_s)
    yield "Sorry, I'm having trouble processing that request right now. Could you rephrase it?"