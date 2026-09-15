"""
Tool-calling loop against Groq: send the conversation plus tool schemas,
execute any tool calls the model returns, feed the results back as tool
messages, and repeat until the model answers with plain content.

Two entry points: run_agent() (non-streaming) and run_agent_stream()
(streaming). /chat selects between them via `stream=true|false`. Both are
async: the Groq client is the AsyncGroq variant, and the sync retrieval /
tool / Redis calls are pushed to a worker thread so they never block the
event loop.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator

from app.agent.tool_registry import call_tool, get_tool_schemas
from app.components.groq.client import get_async_groq_client
from app.components.redis.session_store import get_history
from app.configs.agent_config import (
    AGENT_MODEL,
    CONVERSATION_HISTORY_LIMIT,
    DESTINATION_NAMES,
    FINAL_ANSWER_MAX_TOKENS,
    MAX_TOOL_ITERATIONS,
)
from app.utils.logger import log_event

_destinations = ", ".join(DESTINATION_NAMES)
_destination_count = len(DESTINATION_NAMES)

SYSTEM_PROMPT = f"""You are Yatra, a helpful assistant for Nepali domestic tourists, covering \
exactly {_destination_count} destinations: {_destinations}.

Always call check_scope first on any real question to confirm it's within your domain. If in \
scope, use search_destinations (or other tools) to retrieve real information before answering — \
never answer from memory alone. If out of scope, politely explain you can only help with these \
{_destination_count} destinations.

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


def _capture_retrieved_context(
    tool_name: str, result, retrieved_context: list[str]
) -> None:
    """Collect the text the retrieval-backed tools actually returned, so the
    faithfulness judge can score against exactly what the agent saw."""
    if tool_name == "search_destinations" and isinstance(result, list):
        retrieved_context.extend(chunk.get("text", "") for chunk in result)
    elif tool_name == "compare_destinations" and isinstance(result, dict):
        for sections in result.values():
            if isinstance(sections, dict):
                retrieved_context.extend(sections.values())


async def run_agent(user_query: str, sender: str) -> dict:
    """
    Returns {"answer": str, "retrieved_context": list[str]} — the context
    list captures every chunk actually retrieved during this turn, so eval
    scripts can judge faithfulness against what the agent really saw,
    not a fresh, potentially different retrieval.
    """
    turn_start = time.perf_counter()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        await asyncio.to_thread(
            get_history, sender, limit=CONVERSATION_HISTORY_LIMIT
        )
    )
    messages.append({"role": "user", "content": user_query})

    client = get_async_groq_client()
    retrieved_context: list[str] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        llm_start = time.perf_counter()
        response = await client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=get_tool_schemas(),
            tool_choice="auto",
            max_tokens=FINAL_ANSWER_MAX_TOKENS,
        )
        llm_elapsed_s = f"{time.perf_counter() - llm_start:.2f}"

        usage = response.usage
        log_event(
            "llm_call",
            iteration=iteration + 1,
            time_secs=llm_elapsed_s,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            total_s = f"{time.perf_counter() - turn_start:.2f}"
            log_event(
                "agent_final_answer", iterations=iteration + 1, total_time_secs=total_s
            )
            return {"answer": msg.content, "retrieved_context": retrieved_context}

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_start = time.perf_counter()
            try:
                args = json.loads(tc.function.arguments)
                if tool_name == "get_conversation_context":
                    args["sender"] = sender
                result = await asyncio.to_thread(call_tool, tool_name, **args)
                elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
                log_event("tool_called", tool=tool_name, args=args, time_secs=elapsed_s)

                _capture_retrieved_context(tool_name, result, retrieved_context)

            except Exception as e:
                elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
                result = {"error": str(e)}
                log_event(
                    "tool_call_failed",
                    tool=tool_name,
                    error=str(e),
                    time_secs=elapsed_s,
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    total_s = f"{time.perf_counter() - turn_start:.2f}"
    log_event(
        "agent_max_iterations_hit",
        max_iterations=MAX_TOOL_ITERATIONS,
        total_time_secs=total_s,
    )
    return {
        "answer": "Sorry, I'm having trouble processing that request right now. Could you rephrase it?",
        "retrieved_context": retrieved_context,
    }


async def run_agent_stream(
    user_query: str,
    sender: str,
    retrieved_context: list[str] | None = None,
) -> AsyncGenerator[str]:
    """Yield text chunks as they are generated.

    Tool-call rounds run silently between yields; only final-answer
    generation streams text to the caller. The retrieved chunks are written
    into the caller-supplied `retrieved_context` list (async generators
    cannot return values, so the caller owns that list and reads it after
    the generator completes).
    """
    if retrieved_context is None:
        retrieved_context = []
    turn_start = time.perf_counter()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        await asyncio.to_thread(
            get_history, sender, limit=CONVERSATION_HISTORY_LIMIT
        )
    )
    messages.append({"role": "user", "content": user_query})

    client = get_async_groq_client()

    for iteration in range(MAX_TOOL_ITERATIONS):
        llm_start = time.perf_counter()
        stream = await client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=get_tool_schemas(),
            tool_choice="auto",
            stream=True,
        )

        content_buffer = ""
        tool_calls_buffer: dict[int, dict] = {}

        async for chunk in stream:
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
                        tool_calls_buffer[idx]["arguments"] += (
                            tc_delta.function.arguments
                        )

        llm_elapsed_s = f"{time.perf_counter() - llm_start:.2f}"
        log_event("llm_call_stream", iteration=iteration + 1, time_secs=llm_elapsed_s)

        if not tool_calls_buffer:
            total_s = f"{time.perf_counter() - turn_start:.2f}"
            log_event(
                "agent_final_answer", iterations=iteration + 1, total_time_secs=total_s
            )
            return

        messages.append(
            {
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
            }
        )

        for tc in tool_calls_buffer.values():
            tool_start = time.perf_counter()
            try:
                args = json.loads(tc["arguments"])
                if tc["name"] == "get_conversation_context":
                    args["sender"] = sender
                result = await asyncio.to_thread(call_tool, tc["name"], **args)
                elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
                log_event(
                    "tool_called", tool=tc["name"], args=args, time_secs=elapsed_s
                )

                _capture_retrieved_context(tc["name"], result, retrieved_context)
            except Exception as e:
                elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
                result = {"error": str(e)}
                log_event(
                    "tool_call_failed",
                    tool=tc["name"],
                    error=str(e),
                    time_secs=elapsed_s,
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                }
            )

    total_s = f"{time.perf_counter() - turn_start:.2f}"
    log_event(
        "agent_max_iterations_hit",
        max_iterations=MAX_TOOL_ITERATIONS,
        total_time_secs=total_s,
    )
    yield "Sorry, I'm having trouble processing that request right now. Could you rephrase it?"