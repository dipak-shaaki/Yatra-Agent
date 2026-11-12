"""Unit tests for the eval harness logic (no LLM, no server, no index)."""

import uuid

import httpx
import pytest

from eval.answer_judge import _build_prompt
from scripts import run_eval
from scripts.eval_answers import _percentile, _required_verdicts, _tool_selection_pass
from scripts.eval_paths import input_path
from scripts.eval_retrieval import bootstrap_ci, precision_recall_rr
from scripts.run_eval import _ApiOriginTransport, validate_api_url


def test_required_verdicts_follow_category_and_flags() -> None:
    factual = {"category": "factual", "retrieved_context": ["chunk"]}
    assert _required_verdicts(factual) == ["faithful", "correct"]

    refusal = {
        "category": "out_of_scope",
        "requires_refusal": True,
        "retrieved_context": [],
    }
    assert _required_verdicts(refusal) == ["refusal_correct"]

    partial = {
        "category": "adversarial",
        "requires_refusal": True,
        "retrieved_context": ["chunk"],
    }
    assert _required_verdicts(partial) == ["faithful", "correct", "refusal_correct"]

    hedged = {
        "category": "ambiguous_conflicting",
        "hedge": True,
        "retrieved_context": ["chunk"],
    }
    assert _required_verdicts(hedged) == ["faithful", "correct", "hedged"]


def test_faithful_not_required_without_retrieved_context() -> None:
    # Budget/discovery cases run metadata tools that capture no chunks.
    budget = {"category": "budget", "retrieved_context": []}
    assert _required_verdicts(budget) == ["correct"]


def test_tool_selection_is_exact_set() -> None:
    assert _tool_selection_pass(["compare_destinations"], ["compare_destinations"])
    # A missing expected tool fails...
    assert not _tool_selection_pass(
        ["search_destinations", "compare_destinations"], ["compare_destinations"]
    )
    # ...and so does an unneeded extra.
    assert not _tool_selection_pass(
        ["compare_destinations"], ["compare_destinations", "calculate_budget_estimate"]
    )
    assert not _tool_selection_pass(["search_destinations"], ["filter_by_criteria"])
    assert _tool_selection_pass([], [])
    assert not _tool_selection_pass([], ["search_destinations"])


def test_percentile_basic() -> None:
    assert _percentile([1, 2, 3, 4, 5], 50) == 3
    assert _percentile([1, 2, 3, 4, 5], 95) == 5
    assert _percentile([], 50) == 0.0


def test_bootstrap_ci_brackets_mean_and_is_reproducible() -> None:
    values = [0.2, 1.0, 0.5, 0.25, 1.0, 0.0]
    low, high = bootstrap_ci(values, samples=200, seed=7)
    mean = sum(values) / len(values)
    assert low <= mean
    assert mean <= high
    assert bootstrap_ci(values, samples=200, seed=7) == (low, high)
    assert bootstrap_ci([], samples=200, seed=7) == (0.0, 0.0)


def test_precision_recall_rr_scores_ranked_hits() -> None:
    m = precision_recall_rr(["a", "b", "c", "d", "e"], ["b"])
    assert m["precision"] == 0.2
    assert m["recall"] == 1.0
    assert m["rr"] == 0.5

    miss = precision_recall_rr(["a", "b", "c", "d", "e"], ["z"])
    assert miss["precision"] == 0.0
    assert miss["recall"] == 0.0
    assert miss["rr"] == 0.0


def test_judge_prompt_splices_via_delimited_blocks() -> None:
    # Placeholder-looking user text must land inside its block verbatim,
    # not be interpreted by the prompt builder.
    prompt = _build_prompt("{question}", "{expected}", "{context}", "{actual}")
    assert "<<<Q\n{question}\n>>>" in prompt
    assert "<<<A\n{actual}\n>>>" in prompt


def test_eval_input_must_remain_inside_eval_tree(tmp_path) -> None:
    with pytest.raises(ValueError, match="inside"):
        input_path(tmp_path)


@pytest.fixture
def outside_eval_symlink(tmp_path):
    link = run_eval.PROJECT_ROOT / "eval" / f"test-outside-{uuid.uuid4().hex}"
    link.symlink_to(tmp_path)
    yield link
    link.unlink(missing_ok=True)


def test_eval_input_rejects_symlinks(outside_eval_symlink) -> None:
    with pytest.raises(ValueError, match="symlinks"):
        input_path(outside_eval_symlink)


def test_api_url_requires_public_http_destination(monkeypatch) -> None:
    monkeypatch.setattr(
        run_eval.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 8000))],
    )
    with pytest.raises(ValueError, match="private, link-local, or reserved"):
        validate_api_url("http://example.test")
    target = validate_api_url("http://localhost:8000/api/v1", allow_loopback=True)
    assert target.host == "localhost"
    assert target.addresses[0].is_loopback


def test_api_url_rejects_non_http_schemes() -> None:
    with pytest.raises(ValueError, match="http://"):
        validate_api_url("file:///etc/passwd")


def test_api_url_rejects_metadata_hosts() -> None:
    with pytest.raises(ValueError, match="metadata"):
        validate_api_url("http://metadata.google.internal/latest/meta-data")


def test_api_url_validation_rejects_private_dns_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        run_eval.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 8000))],
    )

    with pytest.raises(ValueError, match="private, link-local, or reserved"):
        validate_api_url("http://example.test/api/v1")


def test_api_url_requires_exact_trusted_host_and_https(monkeypatch) -> None:
    monkeypatch.setattr(
        run_eval.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_api_url("https://api.example.test/api/v1")
    target = validate_api_url(
        "https://api.example.test/api/v1",
        allowed_hosts=frozenset({"api.example.test"}),
    )
    assert target.host == "api.example.test"
    with pytest.raises(ValueError, match="HTTPS"):
        validate_api_url(
            "http://api.example.test/api/v1",
            allowed_hosts=frozenset({"api.example.test"}),
        )


def test_private_lan_addresses_stay_blocked_for_local_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(
        run_eval.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("10.0.0.12", 8000))],
    )
    with pytest.raises(ValueError, match="private, link-local, or reserved"):
        validate_api_url("http://10.0.0.12:8000/api/v1", allow_loopback=True)


def test_api_origin_transport_routes_fixed_paths_to_validated_origin(
    monkeypatch,
) -> None:
    class CaptureTransport(httpx.BaseTransport):
        request: httpx.Request | None = None

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            self.request = request
            return httpx.Response(200, request=request)

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        run_eval.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )
    target = validate_api_url(
        "https://api.example.test/api/v1",
        allowed_hosts=frozenset({"api.example.test"}),
    )
    transport = _ApiOriginTransport(target, False)
    capture = CaptureTransport()
    transport._transport = capture

    request = httpx.Request("POST", "https://eval.invalid/api/v1/session")
    transport.handle_request(request)

    assert capture.request is not None
    assert str(capture.request.url) == "https://api.example.test/api/v1/session"
    assert capture.request.headers["host"] == "api.example.test"
