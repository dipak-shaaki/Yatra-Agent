"""Groq API client setup — connection only, no business/prompt logic here."""

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
