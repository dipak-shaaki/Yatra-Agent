"""Groq API client setup — connection only, no business/prompt logic here."""
from functools import lru_cache

from app.core.config import get_settings
from groq import Groq


@lru_cache
def get_groq_client() -> Groq:
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key)