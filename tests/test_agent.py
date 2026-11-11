"""
Agent-loop tests with a stubbed AsyncGroq client: tool-call reassembly
across streamed deltas, result capture into AgentResult, the empty-content
guard, and the max-iterations bound. No network, no Chroma; Redis is used
for history (like the app tests).
"""

import asyncio
import uuid
from types import SimpleNamespace

import app.agent.agent as agent_module
from app.agent.agent import AGENT_ERROR_REPLY, AgentResult, run_agent_stream


def _tc_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _chunk(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
        x_groq=None,
    )


class _RoundStream:
    """Async-iterable of chunks, like a streamed Groq completion."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeCompletions:
    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _RoundStream(self._rounds.pop(0))


def _fake_groq(rounds):
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(rounds)))


def _collect(gen):
    async def drain():
        return [chunk async for chunk in gen]

    return asyncio.run(drain())


def test_stream_reassembles_tool_calls_and_captures_result(monkeypatch) -> None:
    # Round 1: one tool call whose name and arguments arrive split across
    # chunks — the loop must buffer per tool_calls index and reassemble.
    # Round 2: plain content streamed in two deltas.
    fake = _fake_groq(
        [
            [
                _chunk(tool_calls=[_tc_delta(0, id="call_1", name="filter_by_")]),
                _chunk(
                    tool_calls=[
                        _tc_delta(0, name="criteria", arguments='{"province": "Ga')
                    ]
                ),
                _chunk(tool_calls=[_tc_delta(0, arguments='ndaki"}')]),
            ],
            [_chunk(content="Bandipur is "), _chunk(content="in Gandaki.")],
        ]
    )
    monkeypatch.setattr(agent_module, "get_async_groq_client", lambda: fake)

    result = AgentResult()
    chunks = _collect(run_agent_stream("any", f"agent-test-{uuid.uuid4()}", result))

    assert chunks == ["Bandipur is ", "in Gandaki."]
    assert result.answer == "Bandipur is in Gandaki."
    assert result.tools_called == ["filter_by_criteria"]
    assert result.llm_calls == 2  # tool-call round + final-answer round

    # The second LLM round received the tool's output as a tool message.
    second_round_messages = fake.chat.completions.calls[1]["messages"]
    tool_message = second_round_messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert "Bandipur" in tool_message["content"]


def test_empty_final_content_yields_error_reply(monkeypatch) -> None:
    # A final round with neither tool calls nor content (e.g. reasoning
    # tokens ate the budget) must not stream an empty answer.
    fake = _fake_groq([[_chunk(content=None), _chunk(content="")]])
    monkeypatch.setattr(agent_module, "get_async_groq_client", lambda: fake)

    result = AgentResult()
    chunks = _collect(run_agent_stream("any", f"agent-test-{uuid.uuid4()}", result))

    assert chunks == [AGENT_ERROR_REPLY]
    assert result.answer == AGENT_ERROR_REPLY
    assert result.tools_called == []
    assert result.llm_calls == 1


def test_max_iterations_ends_with_error_reply(monkeypatch) -> None:
    class _AlwaysToolCalls:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            return _RoundStream(
                [
                    _chunk(
                        tool_calls=[
                            _tc_delta(
                                0, id="c", name="filter_by_criteria", arguments="{}"
                            )
                        ]
                    )
                ]
            )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=_AlwaysToolCalls()))
    monkeypatch.setattr(agent_module, "get_async_groq_client", lambda: fake)

    result = AgentResult()
    chunks = _collect(run_agent_stream("any", f"agent-test-{uuid.uuid4()}", result))

    assert chunks == [AGENT_ERROR_REPLY]
    assert result.answer == AGENT_ERROR_REPLY
    assert (
        fake.chat.completions.calls == agent_module.get_settings().max_tool_iterations
    )
    assert result.llm_calls == agent_module.get_settings().max_tool_iterations
