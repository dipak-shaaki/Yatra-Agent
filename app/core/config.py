"""Application settings.

Single source of truth for every runtime value that comes from the
environment. Default file-based paths are anchored to the project root
instead of the process CWD, so the app behaves the same no matter where
it is launched from.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / "app" / "core" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "Yatra"
    environment: str = "development"

    # Groq
    groq_api_key: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Chroma
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma_db")

    # Corpus (markdown destination documents)
    corpus_dir: str = str(PROJECT_ROOT / "data" / "corpus")

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
