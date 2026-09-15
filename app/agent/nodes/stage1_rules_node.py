"""
Stage 1 — regex fast path for pure filler (greetings, acknowledgments,
farewells). Matches skip the LLM entirely and return a templated reply.

The patterns and reply pools live here on purpose: this is curated copy
that should not be duplicated anywhere else.
"""

import re
import secrets

GREETING_PATTERNS = [
    r"^hi+$",
    r"^hello+$",
    r"^hey+$",
    r"^namaste$",
    r"^good (morning|afternoon|evening)$",
    r"^yo+$",
    r"^h(ey|i|ello)\s+(bro|there|friend|man)$",
]
ACKNOWLEDGMENT_PATTERNS = [
    r"^(kk?|sure|alright|got it|cool|nice|great|thanks?( you)?|thank you)$"
]
FAREWELL_PATTERNS = [r"^(bye|goodbye|see you|take care)$"]

ALL_PATTERNS = GREETING_PATTERNS + ACKNOWLEDGMENT_PATTERNS + FAREWELL_PATTERNS
CATEGORIES = (
    ["greeting"] * len(GREETING_PATTERNS)
    + ["acknowledgment"] * len(ACKNOWLEDGMENT_PATTERNS)
    + ["farewell"] * len(FAREWELL_PATTERNS)
)
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ALL_PATTERNS]

TEMPLATED_REPLIES = {
    "greeting": [
        "Hi! I can help you plan trips to 10 amazing Nepal destinations — treks, heritage towns, and pilgrimage sites. What would you like to know?",
        "Hello! Ready to explore Nepal? Ask me about treks, permits, budgets, or best time to visit.",
        "Hey there! I cover 10 great Nepal destinations. What are you curious about?",
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

    for pattern, category in zip(COMPILED_PATTERNS, CATEGORIES):
        if pattern.match(stripped):
            return category

    return None


def get_templated_reply(category: str) -> str:
    """Pick a random variant from the category's reply pool so repeated
    filler messages do not all receive the identical reply."""
    return secrets.choice(TEMPLATED_REPLIES.get(category, ["Okay!"]))
