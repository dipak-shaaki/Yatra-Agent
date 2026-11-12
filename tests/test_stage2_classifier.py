"""Stage-2 classifier: model choice, JSON mode, parsing, fail-open, log hygiene."""

import asyncio
from types import SimpleNamespace

import pytest

import app.agent.nodes.stage2_classifier_node as stage2
from app.core.config import get_settings


class _FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )


def _client(content=None, error=None):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(content, error))
    )


@pytest.mark.parametrize(
    "category", ["new_question", "filler", "unclear", "out_of_scope"]
)
def test_parses_each_category(monkeypatch, category) -> None:
    fake = _client(content=f'{{"category": "{category}"}}')
    monkeypatch.setattr(stage2, "get_async_groq_client", lambda: fake)

    assert asyncio.run(stage2.classify_stage2("hello there")) == category


def test_uses_router_model_and_json_mode(monkeypatch) -> None:
    fake = _client(content='{"category": "new_question"}')
    monkeypatch.setattr(stage2, "get_async_groq_client", lambda: fake)

    asyncio.run(stage2.classify_stage2("best time for Manaslu"))

    call = fake.chat.completions.calls[0]
    assert call["model"] == get_settings().router_model
    assert call["response_format"] == {"type": "json_object"}


def test_unknown_category_falls_back_to_new_question(monkeypatch) -> None:
    fake = _client(content='{"category": "banana"}')
    monkeypatch.setattr(stage2, "get_async_groq_client", lambda: fake)

    assert asyncio.run(stage2.classify_stage2("q")) == "new_question"


def test_provider_error_fails_open(monkeypatch) -> None:
    fake = _client(error=RuntimeError("boom"))
    monkeypatch.setattr(stage2, "get_async_groq_client", lambda: fake)

    assert asyncio.run(stage2.classify_stage2("q")) == "new_question"


def test_classified_log_omits_raw_user_text(monkeypatch) -> None:
    fake = _client(content='{"category": "filler"}')
    monkeypatch.setattr(stage2, "get_async_groq_client", lambda: fake)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        stage2, "log_event", lambda event, **fields: events.append((event, fields))
    )

    secret = "my-passport-number-is-X1234567"
    asyncio.run(stage2.classify_stage2(secret))

    classified = dict(events)["stage2_classified"]
    assert classified["message_length"] == len(secret)
    assert secret not in str(classified)
