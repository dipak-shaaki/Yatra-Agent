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


def test_documents_upsert_rejects_invalid_admin_token(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "secret-token")
    from app.core.config import get_settings
    get_settings.cache_clear()
    try:
        from app.main import create_app

        client = TestClient(create_app())

        # Missing header
        res = client.post(
            "/api/v1/documents/upsert", json={"filename": "x", "content": "x"}
        )
        assert res.status_code == 403

        # Wrong token
        res = client.post(
            "/api/v1/documents/upsert",
            headers={"X-Admin-Token": "wrong"},
            json={"filename": "x", "content": "x"},
        )
        assert res.status_code == 403
    finally:
        get_settings.cache_clear()
