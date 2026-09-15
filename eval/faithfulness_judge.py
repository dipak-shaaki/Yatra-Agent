"""
LLM-as-judge faithfulness check: given retrieved context and a final answer,
flags any claim in the answer not supported by the context.

Needs an LLM because catching a wrong number or a fabricated detail requires
comparing claims, not word overlap.
"""

import json

from app.components.groq.client import get_async_groq_client
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


async def judge_faithfulness(context: str, answer: str) -> dict:
    client = get_async_groq_client()
    try:
        response = await client.chat.completions.create(
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
