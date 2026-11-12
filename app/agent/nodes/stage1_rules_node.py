"""
Stage 1 — regex fast path for pure filler (greetings, acknowledgments,
farewells). Matches skip classification and the agent loop entirely.

Greetings/farewells get a reply written by filler_reply_node (small model,
template fallback); acknowledgments use the curated templates below. The
patterns and fallback copy live here on purpose: this is the one place that
knows what "pure filler" means.
"""

import re
import secrets

from app.data_ingestion.corpus_meta import get_destinations

# One dict instead of parallel pattern/category lists: the category can
# never drift out of sync with the pattern it belongs to.
FILLER_PATTERNS: dict[str, list[str]] = {
    "greeting": [
        r"^hi+$",
        r"^hello+$",
        r"^hey+$",
        r"^namaste$",
        r"^good (morning|afternoon|evening)$",
        r"^yo+$",
        r"^h(ey|i|ello)\s+(bro|there|friend|man)$",
    ],
    "acknowledgment": [
        r"^(kk?|sure|alright|got it|cool|nice|great|thanks?( you)?|thank you)$"
    ],
    "farewell": [r"^(bye|goodbye|see you|take care)$"],
}

COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), category)
    for category, patterns in FILLER_PATTERNS.items()
    for pattern in patterns
]

TEMPLATED_REPLIES = {
    "greeting": [
        "Hi! I can help you plan trips to {count} amazing Nepal destinations — treks, heritage towns, and pilgrimage sites. What would you like to know?",
        "Hello! Ready to explore Nepal? Ask me about treks, permits, budgets, or best time to visit.",
        "Hey there! I cover {count} great Nepal destinations. What are you curious about?",
    ],
    "acknowledgment": [
        "You're welcome! Let me know if you have more questions about Nepal travel.",
        "Glad that helped! Feel free to ask about any other destination.",
        "Anytime! Let me know if there's anything else I can help with.",
        "Happy to help! Ask away if you have more questions.",
    ],
    "farewell": [
        "Safe travels! Feel free to come back if you need more trip planning help.",
        "Take care! Come back anytime for more Nepal travel tips.",
    ],
}


def classify_stage1(message: str) -> str | None:
    """Return 'greeting' / 'acknowledgment' / 'farewell' when the message is
    purely filler, otherwise None."""
    stripped = message.strip()

    for pattern, category in COMPILED_PATTERNS:
        if pattern.match(stripped):
            return category

    return None


def get_templated_reply(category: str) -> str:
    """Pick a random variant from the category's reply pool so repeated
    filler messages do not all receive the identical reply. The destination
    count is filled at call time, so an upsert is visible without a restart."""
    template = secrets.choice(TEMPLATED_REPLIES.get(category, ["Okay!"]))
    return template.format(count=len(get_destinations()))
