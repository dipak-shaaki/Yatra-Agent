from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="app/core/.env", env_file_encoding="utf-8", extra="ignore")
    # App
    app_name: str = "Yatra"
    environment: str = "development"

    # Groq
    groq_api_key: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Chroma
    chroma_persist_dir: str = "./data/chroma_db"

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # Turn handler
    filler_loop_threshold: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()