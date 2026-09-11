from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    configure_logging()
    settings = get_settings()
    app.state.settings = settings
    # TODO: init redis client, chroma client, load/attach retriever here
    yield
    # shutdown
    # TODO: close redis connection, persist chroma if needed


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    # router registration
    # from app.routers.chat_router import router as chat_router
    # app.include_router(chat_router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        return {"status": "ok", "app": settings.app_name}

    return app


app = create_app()