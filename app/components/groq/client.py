"""Groq API client setup"""

from functools import lru_cache

from app.core.config import get_settings
from groq import AsyncGroq, Groq


@lru_cache
def get_groq_client() -> Groq:
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key)


@lru_cache
def get_async_groq_client() -> AsyncGroq:
    settings = get_settings()
    return AsyncGroq(api_key=settings.groq_api_key)


def reasoning_params(model: str, effort: str | None) -> dict:
    """Extra kwargs for chat.completions.create() controlling reasoning effort.

    Groq only accepts `reasoning_effort` on gpt-oss models; sending it to
    others is a 400. On gpt-oss, reasoning tokens count against max_tokens,
    so "low" keeps the budget available for actual content.
    """
    if model.startswith("openai/gpt-oss") and effort:
        return {"reasoning_effort": effort}
    return {}
