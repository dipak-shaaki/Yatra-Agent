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
    # LLM judging eval answers must use a DIFFERENT model than the one
    # that generated them, or the judge inherits the generator's blind spots.
    judge_model: str = "llama-3.3-70b-versatile"
    agent_model: str = "openai/gpt-oss-20b"
    # gpt-oss reasoning tokens count against max_tokens on Groq; "low" keeps
    # the cap available for real output. None = don't send the parameter.
    reasoning_effort: str | None = "low"
    final_answer_max_tokens: int = 2000
    classifier_max_tokens: int = 2000

    # Agent loop
    max_tool_iterations: int = 5

    # Conversation memory
    conversation_history_limit: int = 12  # messages loaded into context per turn
    max_stored_history_messages: int = 20  # messages kept in Redis before trimming
    session_ttl_seconds: int = 60 * 60 * 2  # 2h of inactivity before expiry

    # Turn handler
    filler_loop_threshold: int = 3

    # Retrieval
    candidate_pool: int = 10  # per-ranker results fetched before RRF fusion
    rrf_k: int = 60  # standard RRF constant; dampens impact of high ranks

    # Chroma
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma_db")
    collection_name: str = "yatra_destinations"

    # Corpus (markdown destination documents)
    corpus_dir: str = str(PROJECT_ROOT / "data" / "corpus")

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    # Opt-in offline mode for airgapped deployments (sets HF_HUB_OFFLINE
    # before sentence-transformers loads). Off by default so first runs
    # can download the model.
    hf_offline: bool = False

    # Admin (mutating /documents and /debug endpoints)
    admin_api_key: str | None = None

    # Chat API
    chat_rate_limit_per_minute: int = 30
    # Exposes retrieved_context + tools_called in /chat responses (and the
    # final SSE event) for eval/debug; off in production.
    expose_eval_context: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
