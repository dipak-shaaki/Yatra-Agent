"""
Guardrail tool — determines whether a query is within Yatra's domain
(the 10 documented Nepal destinations) before any retrieval happens.

A small keyword fast-path only catches unambiguous, high-confidence cases
(saves a Groq call on the clearest ones). Everything else — including
booking requests, medical questions, and anything not obviously in either
bucket — goes through the LLM classifier, which generalizes far better
than a hardcoded phrase list. On any classifier failure, this fails SAFE
(refuses) rather than open, since this is a guardrail, not a UX nicety.
"""

import json

from app.components.groq.client import get_groq_client
from app.configs.agent_config import AGENT_MODEL

# Only unambiguous, low-risk-of-false-positive fast-path matches.
# Everything else is left to the LLM classifier below.
OBVIOUS_OUT_OF_SCOPE = [
    "usd to npr",
    "npr to usd",
    "exchange rate today",
    "currency rate today",
]

SCOPE_CHECK_PROMPT = """You are a scope classifier for a Nepal domestic tourism assistant covering \
exactly 10 destinations: Manaslu Circuit, Annapurna Base Camp, Mardi Himal, Kori, Badimalika, \
Bandipur, Panauti, Gorkha, Rara Lake, and Tansen/Palpa.

In scope: questions about these destinations' routes, permits, budgets, difficulty, best time to \
visit, culture, or general Nepal trekking/travel planning relevant to them. This includes discovery \
questions that mention a NEARBY city or landmark only as a reference point for filtering or \
location context (e.g. "easy treks near Pokhara", "destinations close to Kathmandu") — these ARE \
in scope, since the answer should draw from the 10 covered destinations, even though the reference \
city itself isn't one of them.

Out of scope: live weather/currency data, actual bookings/reservations, medical/emergency advice, \
or questions specifically ABOUT a place that is not one of the 10 destinations (e.g. "tell me about \
Pokhara city itself", "what's the best hotel in Kathmandu").

Respond with ONLY a JSON object: {"in_scope": true or false, "reason": "brief reason"}

Query: {query}"""


def check_scope(query: str) -> dict:
    """Returns {"in_scope": bool, "reason": str}"""
    query_lower = query.lower()

    for keyword in OBVIOUS_OUT_OF_SCOPE:
        if keyword in query_lower:
            return {
                "in_scope": False,
                "reason": f"Matched out-of-scope pattern: '{keyword}'",
            }

    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": SCOPE_CHECK_PROMPT.replace("{query}", query),
                }
            ],
            temperature=0,
            max_tokens=300,
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "in_scope": bool(result.get("in_scope", True)),
            "reason": result.get("reason", ""),
        }
    except Exception:
        # Fail safe: an out-of-scope query slipping through is worse than
        # an occasional over-cautious refusal, since this is a guardrail.
        return {
            "in_scope": False,
            "reason": "Scope check unavailable — defaulting to refuse for safety",
        }
