"""
Reply text for filler turns. Stage 1 still detects greetings/farewells with
regex (free, deterministic); this node only writes the words, using the small
router model with no tools and a short cap.

Any failure — missing key, timeout, rate limit, empty content — falls back to
the templated copy, so a greeting never depends on the provider being up.
Acknowledgments ("thanks", "ok") stay templated: the rule already knows what
kind of filler it is, and paying the classifier plus a reply call for "thanks"
is not worth it.
"""

import asyncio

from groq.types.chat import ChatCompletionMessageParam

from app.agent.nodes.stage1_rules_node import get_templated_reply
from app.components.groq.client import get_async_groq_client, reasoning_params
from app.core.config import get_settings
from app.data_ingestion.corpus_meta import destination_names
from app.services.metrics import record_llm_usage
from app.utils.logger import log_event

# Only these are worth an LLM call; stage 1 decides the category.
DYNAMIC_CATEGORIES = frozenset({"greeting", "farewell"})

_CATEGORY_INSTRUCTIONS = {
    "greeting": (
        "The user is greeting you. Greet them back and invite them to ask about "
        "treks, permits, budgets, culture, or the best time to visit."
    ),
    "farewell": (
        "The user is saying goodbye. Give a warm one-line sign-off and invite "
        "them back for more Nepal trip planning."
    ),
}


def _build_messages(category: str, message: str) -> list[ChatCompletionMessageParam]:
    names = destination_names()
    instruction = _CATEGORY_INSTRUCTIONS.get(
        category, _CATEGORY_INSTRUCTIONS["greeting"]
    )
    system = (
        "You are Yatra, a Nepal domestic travel assistant covering exactly "
        f"{len(names)} destinations: {', '.join(names)}. "
        f"{instruction} "
        "Reply in at most two short sentences. Do not call tools. Do not invent "
        "facts, prices, or availability."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]


async def generate_filler_reply(category: str, message: str) -> str:
    settings = get_settings()
    try:
        client = get_async_groq_client()
        async with asyncio.timeout(settings.filler_reply_timeout_seconds):
            response = await client.chat.completions.create(
                model=settings.router_model,
                messages=_build_messages(category, message),
                temperature=0.7,
                max_tokens=settings.filler_reply_max_tokens,
                **reasoning_params(settings.router_model, settings.reasoning_effort),
            )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("empty filler reply")
        usage = getattr(response, "usage", None)
        if usage:
            record_llm_usage(
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
            )
        log_event(
            "filler_reply_generated", category=category, model=settings.router_model
        )
        return content
    except Exception as e:
        # A greeting is not worth a 500 or a 30s wait; the curated copy is fine.
        log_event(
            "filler_reply_failed",
            category=category,
            error=type(e).__name__,
            detail=str(e)[:300],
        )
        return get_templated_reply(category)
