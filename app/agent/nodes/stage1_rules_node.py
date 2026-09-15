"""
Stage 1 — fast rule-based filter for pure filler messages (greetings,
acknowledgments, farewells). If matched, skip retrieval/generation entirely
and respond with a templated reply — zero LLM calls, zero tokens spent.

The regex patterns and reply pools below are deliberately hardcoded: this is
curated copy a human is meant to review, not data that should vary at
runtime. What WOULD be a real problem is this content getting duplicated in
another file — it should only ever live here.
"""

import random
import re

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
    """
    Returns 'greeting' / 'acknowledgment' / 'farewell' if the message is
    PURELY filler (nothing else in it), or None if it needs real processing.
    """
    stripped = message.strip()

    for pattern, category in zip(COMPILED_PATTERNS, CATEGORIES):
        if pattern.match(stripped):
            return category

    return None


def get_templated_reply(category: str) -> str:
    """Randomly picks a variant from the category's reply pool, so repeated
    filler messages in one session don't all get the identical reply."""
    return random.choice(TEMPLATED_REPLIES.get(category, ["Okay!"]))
