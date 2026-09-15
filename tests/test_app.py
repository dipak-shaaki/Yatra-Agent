"""Smoke tests for the FastAPI app wiring (no external services started)."""

from starlette.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    # TestClient without the context manager skips lifespan, so no
    # embedding-model/Chroma warm-up happens during the smoke test.
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_docs_include_v1_routes() -> None:
    client = TestClient(app)
    schema = client.get("/openapi.json").json()

    assert "/api/v1/chat" in schema["paths"]
    assert "/api/v1/documents/upsert" in schema["paths"]
