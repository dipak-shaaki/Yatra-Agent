"""Smoke + auth tests for the FastAPI app wiring.

Session/rate-limit tests need a live Redis (CI provides a redis service;
locally `docker compose up redis`). Chat tests use stage1-regex filler
queries ("hi") so no Groq calls happen.
"""

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
    assert "/api/v1/session" in schema["paths"]
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


def test_debug_endpoints_require_admin_token(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "secret-token")
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        from app.main import create_app

        client = TestClient(create_app())

        res = client.post("/api/v1/debug/search", json={"query": "manaslu"})
        assert res.status_code == 403

        res = client.post(
            "/api/v1/debug/search",
            json={"query": "manaslu"},
            headers={"X-Admin-Token": "wrong"},
        )
        assert res.status_code == 403
    finally:
        get_settings.cache_clear()


def test_chat_without_session_token_is_rejected() -> None:
    client = TestClient(app)

    res = client.post("/api/v1/chat", json={"query": "hi"})
    assert res.status_code == 401

    res = client.post(
        "/api/v1/chat",
        json={"query": "hi"},
        headers={"X-Session-Token": "not-a-real-token"},
    )
    assert res.status_code == 401


def test_session_flow_and_history_isolation() -> None:
    """A minted token works, and one session cannot see another's history."""
    from app.components.redis.session_store import get_history

    client = TestClient(app)
    token_a = client.post("/api/v1/session").json()["session_token"]
    token_b = client.post("/api/v1/session").json()["session_token"]
    assert token_a and token_a != token_b

    res = client.post(
        "/api/v1/chat",
        json={"query": "hi"},
        headers={"X-Session-Token": token_a},
    )
    assert res.status_code == 200
    assert res.json()["answer"]

    # The reply was stored under token_a's history only.
    assert len(get_history(token_a)) == 2  # user + assistant
    assert get_history(token_b) == []


def test_chat_eval_context_gated_by_settings(monkeypatch) -> None:
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        from app.main import create_app

        token = TestClient(app).post("/api/v1/session").json()["session_token"]

        # Default: eval context hidden.
        client = TestClient(create_app())
        res = client.post(
            "/api/v1/chat", json={"query": "hi"}, headers={"X-Session-Token": token}
        )
        assert res.status_code == 200
        assert "retrieved_context" not in res.json()

        # Opt-in: exposed (empty for a filler reply, but present).
        monkeypatch.setenv("EXPOSE_EVAL_CONTEXT", "true")
        get_settings.cache_clear()
        client = TestClient(create_app())
        res = client.post(
            "/api/v1/chat", json={"query": "hi"}, headers={"X-Session-Token": token}
        )
        assert res.status_code == 200
        assert "retrieved_context" in res.json()
        assert "tools_called" in res.json()
    finally:
        monkeypatch.delenv("EXPOSE_EVAL_CONTEXT", raising=False)
        get_settings.cache_clear()


def test_chat_rate_limit_returns_429(monkeypatch) -> None:
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "1")
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        from app.main import create_app

        client = TestClient(create_app())
        token = client.post("/api/v1/session").json()["session_token"]
        headers = {"X-Session-Token": token}

        first = client.post("/api/v1/chat", json={"query": "hi"}, headers=headers)
        assert first.status_code == 200

        second = client.post("/api/v1/chat", json={"query": "hi"}, headers=headers)
        assert second.status_code == 429
        assert int(second.headers["Retry-After"]) >= 0
    finally:
        monkeypatch.delenv("CHAT_RATE_LIMIT_PER_MINUTE", raising=False)
        get_settings.cache_clear()


def test_session_mint_is_throttled_per_ip(monkeypatch) -> None:
    from app.routers import chat_router

    calls = {"n": 0}

    def _limit(ip: str) -> tuple[bool, int]:
        calls["n"] += 1
        return calls["n"] <= 1, 42

    monkeypatch.setattr(chat_router, "check_session_mint_limit", _limit)
    from app.main import create_app

    client = TestClient(create_app())
    assert client.post("/api/v1/session").status_code == 200
    second = client.post("/api/v1/session")
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "42"


def test_documents_upsert_returns_422_for_invalid_document(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "secret-token")
    from app.core.config import get_settings
    from app.routers import documents_routers

    def _reject(filename: str, content: str) -> dict:
        raise ValueError("Invalid document format, not written: bad frontmatter")

    monkeypatch.setattr(documents_routers, "upsert_document", _reject)
    get_settings.cache_clear()
    try:
        from app.main import create_app

        client = TestClient(create_app())
        res = client.post(
            "/api/v1/documents/upsert",
            headers={"X-Admin-Token": "secret-token"},
            json={"filename": "bad", "content": "x"},
        )
        assert res.status_code == 422
        assert "Invalid document" in res.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_stream_uses_typed_sse_events() -> None:
    client = TestClient(app)
    token = client.post("/api/v1/session").json()["session_token"]

    with client.stream(
        "POST",
        "/api/v1/chat",
        json={"query": "hi", "stream": True},
        headers={"X-Session-Token": token},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: token" in body
    assert '"type": "token"' in body
    assert "event: done" in body
    assert "event: meta" not in body  # eval context off by default
    assert "[DONE]" not in body


def test_stream_emits_error_event_on_unexpected_failure(monkeypatch) -> None:
    from app.routers import chat_router

    async def _boom(query: str, sender: str, turn_sink=None):
        raise RuntimeError("connection reset")
        yield  # pragma: no cover — keeps this an async generator

    monkeypatch.setattr(chat_router, "handle_turn_stream", _boom)
    client = TestClient(app)
    token = client.post("/api/v1/session").json()["session_token"]

    with client.stream(
        "POST",
        "/api/v1/chat",
        json={"query": "hi", "stream": True},
        headers={"X-Session-Token": token},
    ) as response:
        body = "".join(response.iter_text())

    assert "event: error" in body
    assert "Please try again" in body
    assert "event: done" in body
