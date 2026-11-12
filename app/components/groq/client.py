"""Groq API client setup.

The key is validated on first use, not at import: tests, tooling and the
smoke test can import the app without a GROQ_API_KEY, and a missing key
produces one clear message instead of a pydantic traceback at startup.
"""

from functools import lru_cache

from app.core.config import get_settings
from groq import AsyncGroq, Groq

MISSING_KEY_MESSAGE = (
    "GROQ_API_KEY is not set. Add it to app/core/.env (or the environment) "
    "before sending chat requests."
)


def _api_key() -> str:
    key = get_settings().groq_api_key
    if not key:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    return key


@lru_cache
def get_groq_client() -> Groq:
    settings = get_settings()
    return Groq(
        api_key=_api_key(),
        timeout=settings.groq_timeout_seconds,
        max_retries=settings.groq_max_retries,
    )


@lru_cache
def get_async_groq_client() -> AsyncGroq:
    settings = get_settings()
    return AsyncGroq(
        api_key=_api_key(),
        timeout=settings.groq_timeout_seconds,
        max_retries=settings.groq_max_retries,
    )


def reasoning_params(model: str, effort: str | None) -> dict:
    """Extra kwargs for chat.completions.create() controlling reasoning effort.

    Groq only accepts `reasoning_effort` on gpt-oss models; sending it to
    others is a 400. On gpt-oss, reasoning tokens count against max_tokens,
    so "low" keeps the budget available for actual content.
    """
    if model.startswith("openai/gpt-oss") and effort:
        return {"reasoning_effort": effort}
    return {}
