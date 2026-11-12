"""Dynamic filler replies: small model, no tools, template fallback."""

import asyncio
import uuid
from types import SimpleNamespace

import pytest

import app.agent.nodes.filler_reply_node as filler
from app.agent.graph import Turn, TurnGraph
from app.agent.nodes.stage1_rules_node import TEMPLATED_REPLIES
from app.core.config import get_settings
from app.data_ingestion.corpus_meta import get_destinations


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
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
        )


def _client(content=None, error=None):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(content, error))
    )


def _fallback(category: str) -> set[str]:
    count = len(get_destinations())
    return {t.format(count=count) for t in TEMPLATED_REPLIES[category]}


def test_greeting_uses_router_model_without_tools(monkeypatch) -> None:
    fake = _client(content="Namaste! Ask me about treks, permits, or budgets.")
    monkeypatch.setattr(filler, "get_async_groq_client", lambda: fake)

    reply = asyncio.run(filler.generate_filler_reply("greeting", "hello"))

    assert reply == "Namaste! Ask me about treks, permits, or budgets."
    call = fake.chat.completions.calls[0]
    assert call["model"] == get_settings().router_model
    assert "tools" not in call
    assert call["max_tokens"] == get_settings().filler_reply_max_tokens


@pytest.mark.parametrize("category", ["greeting", "farewell"])
def test_falls_back_to_template_on_error(monkeypatch, category) -> None:
    fake = _client(error=RuntimeError("GROQ_API_KEY is not set"))
    monkeypatch.setattr(filler, "get_async_groq_client", lambda: fake)

    reply = asyncio.run(filler.generate_filler_reply(category, "hi"))

    assert reply in _fallback(category)


def test_falls_back_to_template_on_empty_content(monkeypatch) -> None:
    fake = _client(content="   ")
    monkeypatch.setattr(filler, "get_async_groq_client", lambda: fake)

    reply = asyncio.run(filler.generate_filler_reply("farewell", "bye"))

    assert reply in _fallback("farewell")


def test_stage1_greeting_is_dynamic_end_to_end(monkeypatch) -> None:
    fake = _client(content="Hello! Where in Nepal would you like to go?")
    monkeypatch.setattr(filler, "get_async_groq_client", lambda: fake)

    turn = Turn(query="hello", sender=f"filler-dynamic-{uuid.uuid4()}")
    asyncio.run(TurnGraph().run(turn))

    assert turn.short_circuited
    assert turn.reply == "Hello! Where in Nepal would you like to go?"
    assert turn.llm_calls == 1


def test_acknowledgment_stays_templated(monkeypatch) -> None:
    fake = _client(content="should not be used")
    monkeypatch.setattr(filler, "get_async_groq_client", lambda: fake)

    turn = Turn(query="thanks", sender=f"filler-dynamic-{uuid.uuid4()}")
    asyncio.run(TurnGraph().run(turn))

    assert turn.reply in _fallback("acknowledgment")
    assert turn.llm_calls == 0
    assert fake.chat.completions.calls == []
