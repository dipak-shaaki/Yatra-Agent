"""
The agent loop: send messages + tool schemas to Groq, execute whichever
tool(s) it calls, feed results back as tool messages, and repeat until
the model responds with plain content instead of another tool call.

This is what makes Yatra agentic rather than a fixed retrieve-then-generate
pipeline — the model decides, per turn, how many tools to call and in what
order (e.g. check_scope -> search_destinations -> final answer).
"""

import json
import time

from app.agent.tool_registry import call_tool, get_tool_schemas
from app.components.groq.client import get_groq_client
from app.utils.logger import log_event
from app.components.redis.session_store import get_history
from app.configs.agent_config import (
    AGENT_MODEL,
    MAX_TOOL_ITERATIONS,
    CONVERSATION_HISTORY_LIMIT,
)


MODEL = AGENT_MODEL


SYSTEM_PROMPT = """You are Yatra, a helpful assistant for Nepali domestic tourists, covering \
exactly 10 destinations: Manaslu Circuit, Annapurna Base Camp, Mardi Himal, Kori, Badimalika, \
Bandipur, Panauti, Gorkha, Rara Lake, and Tansen/Palpa.

Always call check_scope first on any real question to confirm it's within your domain. If in \
scope, use search_destinations (or other tools) to retrieve real information before answering — \
never answer from memory alone. If out of scope, politely explain you can only help with these \
10 destinations.

Format your final answer in Markdown — use headers or bullets for comparisons, bold key figures \
like budgets and altitudes."""


async def run_agent(user_query: str, sender: str) -> str:
    turn_start = time.perf_counter()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        get_history(sender, limit=CONVERSATION_HISTORY_LIMIT)
    )
    messages.append({"role": "user", "content": user_query})

    client = get_groq_client()

    for iteration in range(MAX_TOOL_ITERATIONS):
        llm_start = time.perf_counter()

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=get_tool_schemas(),
            tool_choice="auto",
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
                "agent_final_answer",
                iterations=iteration + 1,
                total_time_secs=total_s,
            )

            return msg.content

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

                result = call_tool(tool_name, **args)

                elapsed_s = f"{time.perf_counter() - tool_start:.2f}"

                log_event(
                    "tool_called",
                    tool=tool_name,
                    args=args,
                    time_secs=elapsed_s,
                )

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

    return "Sorry, I'm having trouble processing that request right now. Could you rephrase it?"