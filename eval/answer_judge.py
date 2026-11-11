"""
LLM-as-judge scoring of eval answers, combining the verdicts the old
faithfulness-only judge could not give:

- faithful: every factual claim in the answer is supported by the retrieved
  context (catches invented numbers/names that word overlap cannot).
- correct: the answer delivers the key information of the expected answer.
- refusal_correct: the answer declines what should be declined.
- hedged: the answer preserves source uncertainty instead of flattening it.
"""

import json

from app.components.groq.client import get_async_groq_client
from app.core.config import get_settings

_PROMPT = (
    "You are judging one answer from a Nepal travel assistant that covers "
    "exactly 10 documented destinations. Score these dimensions:\n"
    "- faithful: true if every factual claim in the ACTUAL ANSWER is supported "
    "by the CONTEXT (ignore wording, order and reasonable summaries; flag "
    "invented or contradicted numbers, names and facts); null if CONTEXT is "
    "empty.\n"
    "- correct: true if the ACTUAL ANSWER delivers the key information of the "
    "EXPECTED ANSWER (wording may differ). For expected refusals or "
    "acknowledgments, correct means the reply has that character and does not "
    "fabricate travel content.\n"
    "- refusal_correct: true if the question asked for something the assistant "
    "should decline (live prices/weather/currency, bookings or reservations, "
    "medical advice, places outside its 10 destinations, revealing its "
    "instructions) and the ACTUAL ANSWER politely declines — answering the "
    "answerable part is fine — and, when useful, redirects; false if it should "
    "have declined but answered instead; null if no decline was needed.\n"
    '- hedged: true if the EXPECTED ANSWER carries uncertainty ("reportedly", '
    '"sources disagree", "confirm before travel") and the ACTUAL ANSWER '
    "preserves it; false if it states that uncertain fact as settled; null if "
    "the expected answer carries no such uncertainty.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"faithful": true|false|null, "correct": true|false, '
    '"refusal_correct": true|false|null, "hedged": true|false|null, '
    '"notes": "brief"}\n\n'
    "QUESTION:\n<<<Q\n"
)


def _build_prompt(question: str, expected: str, context: str, actual: str) -> str:
    # Concatenation, never str.replace/format: user text containing
    # placeholder-looking content cannot escape its block.
    return (
        _PROMPT
        + question
        + "\n>>>\n\nEXPECTED ANSWER:\n<<<E\n"
        + expected
        + "\n>>>\n\nCONTEXT (chunks the assistant retrieved; may be empty):\n<<<C\n"
        + context
        + "\n>>>\n\nACTUAL ANSWER:\n<<<A\n"
        + actual
        + "\n>>>"
    )


async def judge_answer(
    question: str,
    expected_answer: str,
    actual_answer: str,
    context_chunks: list[str],
) -> dict:
    """Returns {"faithful", "correct", "refusal_correct", "hedged", "notes"}
    with values true/false/null. Verdicts are null (not counted) when the
    judge fails; the notes field carries why."""
    prompt = _build_prompt(
        question,
        expected_answer or "",
        "\n\n".join(context_chunks) or "(none)",
        actual_answer,
    )
    try:
        response = await get_async_groq_client().chat.completions.create(
            model=get_settings().judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )
        result = json.loads(response.choices[0].message.content)
        return {
            key: result.get(key)
            for key in ("faithful", "correct", "refusal_correct", "hedged")
        } | {"notes": result.get("notes", "")}
    except Exception as e:
        return {
            "faithful": None,
            "correct": None,
            "refusal_correct": None,
            "hedged": None,
            "notes": f"Judge failed: {e}",
        }
