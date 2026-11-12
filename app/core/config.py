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
    log_level: str = "INFO"

    # Groq
    # Optional at import time so unit tests / tooling work without a key;
    # the client raises a clear error on first use instead.
    groq_api_key: str | None = None
    # LLM judging eval answers must use a DIFFERENT model than the one
    # that generated them, or the judge inherits the generator's blind spots.
    judge_model: str = "llama-3.3-70b-versatile"
    agent_model: str = "openai/gpt-oss-20b"
    # Small, fast model for the 4-way turn classifier; a normal question
    # should not pay the reasoning model's latency just to be routed.
    router_model: str = "llama-3.1-8b-instant"
    # gpt-oss reasoning tokens count against max_tokens on Groq; "low" keeps
    # the cap available for real output. None = don't send the parameter.
    reasoning_effort: str | None = "low"
    final_answer_max_tokens: int = 2000
    classifier_max_tokens: int = 400
    groq_timeout_seconds: float = 30.0
    groq_max_retries: int = 2

    # Agent loop
    max_tool_iterations: int = 5
    turn_timeout_seconds: float = 45.0
    max_concurrent_llm_calls: int = 2
    max_concurrent_embeddings: int = 2

    # Conversation memory
    conversation_history_limit: int = 12  # messages loaded into context per turn
    max_stored_history_messages: int = 20  # messages kept in Redis before trimming
    session_ttl_seconds: int = 60 * 60 * 2  # 2h of inactivity before expiry

    # Turn handler
    filler_loop_threshold: int = 3

    # Retrieval
    candidate_pool: int = 10  # per-ranker results fetched before RRF fusion
    rrf_k: int = 60  # standard RRF constant; dampens impact of high ranks
    search_n_results: int = 5  # default chunks returned by the search tool
    filter_max_results: int = 10  # cap on filter_by_criteria matches

    # Chroma
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma_db")
    collection_name: str = "yatra_destinations"

    # Corpus (markdown destination documents)
    corpus_dir: str = str(PROJECT_ROOT / "data" / "corpus")

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    # Pin the commit, not `main`: an upstream retag would otherwise change the
    # weights under already-stored vectors (same name + dimension, so the
    # startup check cannot catch it). Stored in Chroma metadata and verified.
    embedding_revision: str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    # Opt-in offline mode for airgapped deployments (sets HF_HUB_OFFLINE
    # before sentence-transformers loads). Off by default so first runs
    # can download the model.
    hf_offline: bool = False

    # Admin (mutating /documents and /debug endpoints)
    admin_api_key: str | None = None

    # Chat API
    chat_rate_limit_per_minute: int = 30
    session_rate_limit_per_minute: int = 10  # per-IP token minting cap
    rate_limit_window_seconds: int = 61  # window expiry (+1s grace)
    max_query_length: int = 2000
    # Exposes retrieved_context + tools_called in /chat responses (and the
    # final SSE event) for eval/debug; off in production.
    expose_eval_context: bool = False

    # Metrics / cost estimate (USD per million tokens; edit to your model)
    input_cost_per_million: float = 0.075
    output_cost_per_million: float = 0.30

    # Redis
    redis_url: str = "redis://localhost:6379/0"


@lru_cache
def get_settings() -> Settings:
    # groq_api_key is required but arrives via the environment / .env file
    # (pydantic-settings), which static analysis cannot see — hence the
    # silenced call-arg diagnostic below.
    return Settings()  # type: ignore[call-arg]
