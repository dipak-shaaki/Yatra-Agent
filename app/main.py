from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.db.chroma.client import get_collection
from app.retrieval.embeddings import get_embedding_model
from app.routers.chat_router import router as chat_router
from app.routers.documents_routers import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    configure_logging()
    settings = get_settings()
    app.state.settings = settings

    # Preload embedding model + Chroma collection so the first real
    # user request doesn't pay cold-load latency.
    get_embedding_model()
    get_collection()
    # TODO: init redis client , once we want an explicit connection check

    yield

    # shutdown
    # TODO: close redis connection, persist chroma if needed


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    @app.get("/health")
    async def health():
        return {"status": "ok", "app": settings.app_name}

    return app


app = create_app()