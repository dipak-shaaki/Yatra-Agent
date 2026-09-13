"""
Stage 2 — LLM fallback for ambiguous cases Stage 1's regex didn't cleanly
match (e.g. "yeah okay but what about permits"). One cheap Groq call
determines: new_question / filler / unclear.
"""
import json

from app.components.groq.client import get_groq_client
from app.utils.logger import log_event
from app.configs.agent_config import AGENT_MODEL

STAGE2_PROMPT = """Classify this message as exactly one of: "new_question", "filler", "unclear".

- new_question: contains a real question or request needing information
- filler: pure acknowledgment/greeting/farewell, even if phrased conversationally
- unclear: ambiguous, can't tell

Respond with ONLY a JSON object: {"category": "new_question" or "filler" or "unclear"}

Message: {message}"""


def classify_stage2(message: str) -> str:
    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": STAGE2_PROMPT.replace("{message}", message)}],
            temperature=0,
            max_tokens=300,
        )
        result = json.loads(response.choices[0].message.content)
        category = result.get("category", "new_question")
        log_event("stage2_classified", message=message[:50], category=category)
        return category
    except Exception as e:
        log_event("stage2_classify_failed", error=str(e))
        return "new_question"  