"""Unit tests for the eval harness logic (no LLM, no server, no index)."""

import pytest

from eval.answer_judge import _build_prompt
from scripts import run_eval
from scripts.eval_answers import _percentile, _required_verdicts, _tool_selection_pass
from scripts.eval_paths import input_path
from scripts.eval_retrieval import bootstrap_ci, precision_recall_rr
from scripts.run_eval import validate_api_url


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
    assert (low, high) == bootstrap_ci(values, samples=200, seed=7)
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


def test_eval_input_rejects_symlinks(tmp_path) -> None:
    link = run_eval.PROJECT_ROOT / "eval" / "test-outside-link"
    link.symlink_to(tmp_path)
    try:
        with pytest.raises(ValueError, match="symlinks"):
            input_path(link)
    finally:
        link.unlink()


def test_api_url_requires_public_http_destination(monkeypatch) -> None:
    monkeypatch.setattr(
        run_eval.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 8000))],
    )
    with pytest.raises(ValueError, match="non-public"):
        validate_api_url("http://example.test")
    assert validate_api_url("http://example.test", allow_private=True)


def test_api_url_rejects_non_http_schemes() -> None:
    with pytest.raises(ValueError, match="http://"):
        validate_api_url("file:///etc/passwd")
