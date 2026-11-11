import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.components.redis.client import get_redis_client
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.db.chroma.client import get_collection
from app.retrieval.embeddings import get_embedding_model
from app.routers.chat_router import router as chat_router
from app.routers.documents_routers import router as documents_router

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    app.state.settings = settings

    # Preload embedding model + Chroma collection so the first real
    # user request doesn't pay cold-load latency.
    get_embedding_model()
    get_collection()

    # Warn-only Redis check — the app is happy to start; session memory
    # will return empty history until Redis becomes reachable.
    try:
        await asyncio.to_thread(get_redis_client().ping)
    except Exception as e:
        _logger.warning(
            "Redis unavailable at startup (%s) — session memory disabled", e
        )

    yield

    try:
        get_redis_client().close()
    except Exception as e:
        _logger.warning("Redis close failed: %s", e)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        return {"status": "ok", "app": settings.app_name}

    @app.get("/ready")
    async def ready():
        """Readiness for orchestration: 200 only when both backing stores
        answer. /health is liveness (always ok once booted); this catches a
        Redis/Chroma that went down after startup."""
        checks: dict[str, bool] = {}
        try:
            await asyncio.to_thread(get_redis_client().ping)
            checks["redis"] = True
        except Exception:
            checks["redis"] = False
        try:
            await asyncio.to_thread(get_collection)
            checks["chroma"] = True
        except Exception:
            checks["chroma"] = False
        if all(checks.values()):
            return {"status": "ready", "checks": checks}
        raise HTTPException(
            status_code=503, detail={"status": "not ready", "checks": checks}
        )

    return app


app = create_app()
