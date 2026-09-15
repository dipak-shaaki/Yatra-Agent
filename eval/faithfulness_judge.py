"""
LLM-as-judge faithfulness check: given the retrieved context and a final
answer, flags any claim in the answer NOT supported by that context.

This is the one eval metric that genuinely needs an LLM (vs. Precision@K/
Recall@K/tool-accuracy, which are deterministic) — catching a wrong number
or fabricated detail requires actually comparing claims, not just word
overlap, which is why we're not using the word-overlap approach common in
generic RAG eval guides.
"""

import json

from app.components.groq.client import get_groq_client
from app.configs.agent_config import AGENT_MODEL

JUDGE_PROMPT = """You are a strict fact-checker. Given RETRIEVED CONTEXT and an ANSWER, \
identify any factual claims in the ANSWER that are NOT supported by the CONTEXT.

Ignore stylistic differences, formatting, and reasonable inferences (e.g. summarizing or \
rephrasing). Only flag claims that state something as fact which the context does not contain \
or contradicts — this includes numbers, names, dates, and specific details.

Respond with ONLY a JSON object:
{
  "faithful": true or false,
  "unsupported_claims": ["claim 1", "claim 2", ...],
  "notes": "brief explanation"
}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}"""


def judge_faithfulness(context: str, answer: str) -> dict:
    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.replace("{context}", context).replace(
                        "{answer}", answer
                    ),
                }
            ],
            temperature=0,
            max_tokens=2000,
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "faithful": bool(result.get("faithful", False)),
            "unsupported_claims": result.get("unsupported_claims", []),
            "notes": result.get("notes", ""),
        }
    except Exception as e:
        return {
            "faithful": None,
            "unsupported_claims": [],
            "notes": f"Judge failed: {e}",
        }
