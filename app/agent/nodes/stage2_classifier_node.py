"""
Stage 2 — LLM fallback for ambiguous cases Stage 1's regex didn't cleanly
match. Includes the last assistant message as context, since short
follow-ups ("yes", "I wanna compare") are often direct responses to
something the assistant just asked — classifying them in isolation
misreads them as filler/unclear.
"""
import json

from app.components.groq.client import get_groq_client
from app.configs.agent_config import AGENT_MODEL
from app.utils.logger import log_event

STAGE2_PROMPT = """Classify the user's LATEST message as exactly one of: "new_question", "filler", "unclear".

- new_question: contains a real question or request needing information, OR is a genuine engaged \
response to something the assistant just asked — including short answers ("yes", "okay", "I wanna \
compare") that continue the conversation, AND expressions of uncertainty ("I'm not sure", "no idea").

- filler: pure acknowledgment/greeting/farewell with no real content, and NOT a response to \
something the assistant just asked (e.g. "ok", "thanks", "bye" with nothing else)
- unclear: genuinely nonsensical, garbled, or impossible to interpret even accounting for context \
(e.g. random characters, completely off-topic gibberish)

Respond with ONLY a JSON object: {"category": "new_question" or "filler" or "unclear"}

Assistant's last message (for context, may be empty if this is the first message): {last_assistant_message}

User's latest message: {message}"""

def classify_stage2(message: str, last_assistant_message: str = "") -> str:
    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[{
                "role": "user",
                "content": STAGE2_PROMPT
                    .replace("{message}", message)
                    .replace("{last_assistant_message}", last_assistant_message or "(none)"),
            }],
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