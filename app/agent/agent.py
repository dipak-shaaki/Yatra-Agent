"""
Tool-calling loop against Groq (streaming): send the conversation plus tool
schemas, execute any tool calls the model returns — in parallel — feed the
results back as tool messages, and repeat until the model answers with plain
content.

One loop serves both /chat delivery modes: the graph drains
run_agent_stream() and either forwards the text deltas (SSE) or discards
them (JSON). State (answer, retrieved context, tools called) is written into
a caller-owned AgentResult, since async generators cannot return values.

The Groq client is the AsyncGroq variant; the sync retrieval / tool / Redis
calls are pushed to worker threads so they never block the event loop.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from app.agent.tool_registry import call_tool, get_tool_schemas
from app.components.groq.client import get_async_groq_client, reasoning_params
from app.components.redis.session_store import get_history
from app.core.config import get_settings
from app.data_ingestion.corpus_meta import destination_names
from app.utils.logger import log_event

AGENT_ERROR_REPLY = (
    "Sorry, I'm having trouble processing that request right now. "
    "Could you rephrase it?"
)


@dataclass
class AgentResult:
    """What a finished agent turn produced; owned by the caller."""

    answer: str = ""
    retrieved_context: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    llm_calls: int = 0


def _system_prompt() -> str:
    names = destination_names()
    _destinations = ", ".join(names)
    _destination_count = len(names)
    return f"""You are Yatra, a helpful assistant for Nepali domestic tourists, covering \
exactly {_destination_count} destinations: {_destinations}.

Use the tools to retrieve real information before answering — never answer from memory \
alone. The user's question has already been confirmed to be within your domain, so go \
straight to retrieving and answering.

Response style:
- Be concise. Answer only what was asked.
- DEFAULT to plain sentences or a short bullet list. Example — for "best time to visit Rara \
Lake?", write: "Spring (March-May) and autumn (September-November) are best — clear skies and \
good visibility. Avoid monsoon (June-August, rain) and winter (heavy snow)." NOT a table.
- ONLY use a table when comparing 2+ destinations side by side, or listing 4+ structured items \
(e.g. a multi-day budget breakdown). A single destination's single attribute (best time, \
difficulty, one permit) never needs a table.
- PRESERVE uncertainty: if retrieved sources hedge ("reportedly", "may have changed", \
"confirm before travel"), keep that hedge in your answer instead of stating it as certain fact. \
Never silently drop a caveat like "check current requirements before booking".
- CITE sources for key facts using the chunk's source_file and last_updated, e.g. \
"(source: manaslu_circuit.md, updated 2026-09-11)". One citation style, used consistently.
- End with ONE brief, natural follow-up question when it fits — skip it if the answer is already \
a direct yes/no or the conversation seems to be wrapping up."""


def _capture_tool_result(tool_name: str, result, run: AgentResult) -> None:
    """Collect retrieved text + tool usage so evals can score against what
    the agent actually saw and called."""
    run.tools_called.append(tool_name)
    if tool_name == "search_destinations" and isinstance(result, list):
        run.retrieved_context.extend(chunk.get("text", "") for chunk in result)
    elif tool_name == "compare_destinations" and isinstance(result, dict):
        for sections in result.values():
            if isinstance(sections, dict):
                run.retrieved_context.extend(sections.values())


async def _execute_tool_call(tool_call: dict, sender: str, run: AgentResult) -> dict:
    """Runs one tool call on a worker thread; returns the tool message."""
    tool_start = time.perf_counter()
    name = tool_call["name"]
    try:
        args = json.loads(tool_call["arguments"])
        result = await asyncio.to_thread(call_tool, name, **args)
        elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
        log_event("tool_called", tool=name, args=args, time_secs=elapsed_s)
        _capture_tool_result(name, result, run)
        payload = result
    except Exception as e:
        elapsed_s = f"{time.perf_counter() - tool_start:.2f}"
        payload = {"error": str(e)}
        log_event("tool_call_failed", tool=name, error=str(e), time_secs=elapsed_s)
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": json.dumps(payload),
    }


async def run_agent_stream(
    user_query: str,
    sender: str,
    result: AgentResult | None = None,
) -> AsyncGenerator[str]:
    """Yield final-answer text chunks as they are generated.

    Tool-call rounds run silently between yields. On completion the joined
    answer, retrieved context and tool calls are written to `result` (created
    if not supplied) for the caller to read.
    """
    run = result if result is not None else AgentResult()
    settings = get_settings()
    turn_start = time.perf_counter()
    # Heterogeneous message dicts (system/user/assistant-with-tool_calls/tool);
    # Groq's TypedDict param union cannot express this shape (silenced at the
    # create() call below).
    messages: list[dict[str, Any]] = [{"role": "system", "content": _system_prompt()}]
    messages.extend(
        await asyncio.to_thread(
            get_history, sender, limit=settings.conversation_history_limit
        )
    )
    messages.append({"role": "user", "content": user_query})

    client = get_async_groq_client()

    for iteration in range(settings.max_tool_iterations):
        llm_start = time.perf_counter()
        # Groq's TypedDict param union cannot express our heterogeneous
        # message dicts, and the unpacked reasoning_params dict defeats
        # overload matching — runtime accepts both fine.
        stream = await client.chat.completions.create(  # type: ignore[call-overload]
            model=settings.agent_model,
            messages=messages,
            tools=get_tool_schemas(),
            tool_choice="auto",
            stream=True,
            max_tokens=settings.final_answer_max_tokens,
            **reasoning_params(settings.agent_model, settings.reasoning_effort),
        )
        run.llm_calls += 1

        content_buffer = ""
        tool_calls_buffer: dict[int, dict] = {}
        usage = None

        async for chunk in stream:
            # Usage rides on x_groq in the final chunk (some chunks carry
            # no choices at all — skip those before touching delta).
            if getattr(chunk, "x_groq", None) and chunk.x_groq.usage:
                usage = chunk.x_groq.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                content_buffer += delta.content
                yield delta.content

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        tool_calls_buffer[idx]["id"] = tc_delta.id
                    if tc_delta.function.name:
                        tool_calls_buffer[idx]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_buffer[idx]["arguments"] += (
                            tc_delta.function.arguments
                        )

        llm_elapsed_s = f"{time.perf_counter() - llm_start:.2f}"
        log_event(
            "llm_call_stream",
            iteration=iteration + 1,
            time_secs=llm_elapsed_s,
            **(
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage
                else {}
            ),
        )

        if not tool_calls_buffer:
            total_s = f"{time.perf_counter() - turn_start:.2f}"
            if not content_buffer.strip():
                # The model stopped without content — typically reasoning
                # tokens ate the budget. Fail loudly instead of streaming
                # an empty answer.
                log_event(
                    "agent_empty_content",
                    iterations=iteration + 1,
                    total_time_secs=total_s,
                )
                run.answer = AGENT_ERROR_REPLY
                yield AGENT_ERROR_REPLY
                return
            log_event(
                "agent_final_answer", iterations=iteration + 1, total_time_secs=total_s
            )
            run.answer = content_buffer
            return

        messages.append(
            {
                "role": "assistant",
                "content": content_buffer or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in tool_calls_buffer.values()
                ],
            }
        )

        # Independent tool calls in one round run concurrently.
        tool_messages = await asyncio.gather(
            *(_execute_tool_call(tc, sender, run) for tc in tool_calls_buffer.values())
        )
        messages.extend(tool_messages)

    total_s = f"{time.perf_counter() - turn_start:.2f}"
    log_event(
        "agent_max_iterations_hit",
        max_iterations=settings.max_tool_iterations,
        total_time_secs=total_s,
    )
    run.answer = AGENT_ERROR_REPLY
    yield AGENT_ERROR_REPLY
