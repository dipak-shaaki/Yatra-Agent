"""Unit tests for the eval harness logic (no LLM, no server, no index)."""

from eval.answer_judge import _build_prompt
from scripts.eval_answers import _percentile, _required_verdicts, _tool_selection_pass
from scripts.eval_retrieval import bootstrap_ci, precision_recall_rr


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
    assert low <= sum(values) / len(values) <= high
    assert (low, high) == bootstrap_ci(values, samples=200, seed=7)
    assert bootstrap_ci([], samples=200, seed=7) == (0.0, 0.0)


def test_precision_recall_rr_scores_ranked_hits() -> None:
    m = precision_recall_rr(["a", "b", "c", "d", "e"], ["b"])
    assert m == {"precision": 0.2, "recall": 1.0, "rr": 0.5}

    miss = precision_recall_rr(["a", "b", "c", "d", "e"], ["z"])
    assert miss == {"precision": 0.0, "recall": 0.0, "rr": 0.0}


def test_judge_prompt_splices_via_delimited_blocks() -> None:
    # Placeholder-looking user text must land inside its block verbatim,
    # not be interpreted by the prompt builder.
    prompt = _build_prompt("{question}", "{expected}", "{context}", "{actual}")
    assert "<<<Q\n{question}\n>>>" in prompt
    assert "<<<A\n{actual}\n>>>" in prompt
