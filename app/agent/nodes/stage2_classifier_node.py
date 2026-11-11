"""
Stage 2 — LLM classification for turns Stage 1's regex did not catch.

Scope checking lives here too (the old check_scope tool is gone): one call
classifies the turn as new_question / filler / unclear / out_of_scope, which
saves the two extra LLM calls the tool-based guardrail spent per question.

The assistant's last message is included as context so short follow-ups
("yes", "I wanna compare") are not misread as filler or unclear. User text is
delimited in <<< blocks and spliced in by concatenation, never str.replace,
so a message containing placeholder-looking text cannot escape its block.
"""

import json

from app.components.groq.client import get_async_groq_client, reasoning_params
from app.core.config import get_settings
from app.data_ingestion.corpus_meta import destination_names
from app.utils.logger import log_event

CATEGORIES = ("new_question", "filler", "unclear", "out_of_scope")


def _build_prompt(
    message: str, last_assistant_message: str, names: tuple[str, ...]
) -> str:
    count = len(names)
    return (
        "Classify the user's LATEST message as exactly one of: "
        '"new_question", "filler", "unclear", "out_of_scope".\n'
        "You are classifying for a Nepal domestic tourism assistant covering "
        f"exactly {count} destinations: {', '.join(names)}.\n\n"
        "- new_question: a real question or request needing information about these "
        "destinations (routes, permits, budgets, difficulty, best time, culture), OR "
        "comparison/discovery questions across them, OR a genuine engaged response to "
        'something the assistant just asked — including short answers ("yes", '
        '"okay", "I wanna compare") and expressions of uncertainty ("I\'m not '
        'sure", "no idea"). A nearby city or landmark mentioned only as a '
        'reference point (e.g. "easy treks near Pokhara") is still in scope.\n'
        "- filler: pure acknowledgment/greeting/farewell with no real content, and NOT "
        'a response to something the assistant just asked (e.g. "ok", "thanks", '
        '"bye" with nothing else)\n'
        "- unclear: genuinely nonsensical, garbled, or impossible to interpret even "
        "accounting for context (e.g. random characters, gibberish)\n"
        "- out_of_scope: a real question, but outside the assistant's domain — live "
        "weather/currency data, actual bookings/reservations, medical/emergency "
        f"advice, or questions specifically ABOUT a place that is not one of the "
        f'{count} destinations (e.g. "tell me about Pokhara city itself", "best '
        'hotel in Kathmandu", "usd to npr rate")\n\n'
        "Respond with ONLY a JSON object: "
        '{"category": "new_question" or "filler" or "unclear" or "out_of_scope"}\n\n'
        "Assistant's last message (may be empty on the first turn):\n"
        f"<<<ASSISTANT\n{last_assistant_message or '(none)'}\n>>>\n\n"
        "User's latest message:\n"
        f"<<<USER\n{message}\n>>>"
    )


async def classify_stage2(message: str, last_assistant_message: str = "") -> str:
    client = get_async_groq_client()
    settings = get_settings()
    prompt = _build_prompt(message, last_assistant_message, destination_names())
    try:
        response = await client.chat.completions.create(
            model=settings.agent_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=settings.classifier_max_tokens,
            **reasoning_params(settings.agent_model, settings.reasoning_effort),
        )
        result = json.loads(response.choices[0].message.content)
        category = result.get("category", "new_question")
        if category not in CATEGORIES:
            category = "new_question"
        log_event("stage2_classified", message=message[:50], category=category)
        return category
    except Exception as e:
        # Fail open on classifier errors: the agent's own system prompt still
        # constrains it to the documented destinations, so a hiccup here
        # costs a slightly longer answer rather than blocking the user.
        log_event("stage2_classify_failed", error=str(e))
        return "new_question"
