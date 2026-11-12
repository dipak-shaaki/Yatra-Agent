"""
Judges eval/results.jsonl (written by scripts/run_eval.py) with the combined
answer judge, computes tool-selection accuracy and latency percentiles, and
writes eval/judged.jsonl.

Run as `python -m scripts.eval_answers` from the project root.

A case passes when every verdict required for its category is true:
faithful+correct for answer categories, correct alone for filler/unclear,
plus refusal_correct for cases flagged requires_refusal and hedged for
cases flagged hedge. Tool selection is exact: the set of expected tools must
equal the set of tools actually called (wrong extras fail).
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from scripts.eval_paths import input_path, output_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from eval.answer_judge import judge_answer

# Per-category verdicts every case must pass; refusal_correct/hedged are
# appended per case from the requires_refusal / hedge flags in the cases.
REQUIRED_VERDICTS = {
    "factual": ("faithful", "correct"),
    "comparison": ("faithful", "correct"),
    "discovery": ("faithful", "correct"),
    "budget": ("faithful", "correct"),
    "ambiguous_conflicting": ("faithful", "correct"),
    "adversarial": ("faithful", "correct"),
    "filler": ("correct",),
    "unclear": ("correct",),
    "out_of_scope": (),
}


def _required_verdicts(result: dict) -> list[str]:
    required = list(REQUIRED_VERDICTS.get(result["category"], ("correct",)))
    if result.get("requires_refusal"):
        required.append("refusal_correct")
    if result.get("hedge"):
        required.append("hedged")
    if not result.get("retrieved_context"):
        # Nothing was retrieved (metadata tools or short-circuit paths), so
        # there is no context to be faithful to — the judge returns null.
        required = [v for v in required if v != "faithful"]
    return required


def _tool_selection_pass(expected: list[str], actual: list[str]) -> bool:
    """Exact set match: a missing expected tool or an unneeded extra fails."""
    return set(expected) == set(actual)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round(p / 100 * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


async def _judge_result(result: dict) -> dict:
    if result.get("error") or not result.get("actual_answer"):
        return {**result, "verdicts": None, "passed": False}

    verdicts = await judge_answer(
        result["question"],
        result.get("expected_answer") or "",
        result["actual_answer"],
        result.get("retrieved_context") or [],
    )
    required = _required_verdicts(result)
    failed = [verdict for verdict in required if verdicts.get(verdict) is not True]
    return {
        **result,
        "verdicts": verdicts,
        "required_verdicts": required,
        "passed": not failed,
        "tool_selection_pass": _tool_selection_pass(
            result.get("expected_tools") or [], result.get("tools_called") or []
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "eval" / "results.jsonl",
        help="run_eval.py output to judge",
    )
    parser.add_argument(
        "--judged",
        type=Path,
        default=PROJECT_ROOT / "eval" / "judged.jsonl",
        help="Where to write judged results",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between judge calls (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        results_path = input_path(args.results)
        judged_path = output_path(args.judged)
    except ValueError as exc:
        parser.error(str(exc))

    with results_path.open(encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    if not results:
        print(f"No results in {args.results}", file=sys.stderr)
        return 1

    print(f"Judging {len(results)} results with the answer judge...\n")
    judged = []
    for result in results:
        judged_result = asyncio.run(_judge_result(result))
        judged.append(judged_result)
        verdicts = judged_result["verdicts"]
        if verdicts is None:
            print(f"[{result['id']}] ERROR — no answer to judge: {result.get('error')}")
            continue
        failed = [
            verdict
            for verdict in judged_result["required_verdicts"]
            if verdicts.get(verdict) is not True
        ]
        passed = judged_result["passed"]
        tool_pass = judged_result["tool_selection_pass"]

        print(
            f"[{result['id']}] {'PASS' if passed else 'FAIL'}"
            f" tools={'ok' if tool_pass else 'MISS'}"
            f" {result.get('time_secs')}s"
        )
        if failed:
            print(f"  failed verdicts: {failed} — {verdicts.get('notes', '')}")
        time.sleep(args.delay)

    judged_path.parent.mkdir(parents=True, exist_ok=True)
    with judged_path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(j) + "\n" for j in judged)

    scored = [j for j in judged if j["verdicts"] is not None]
    latencies = [j["time_secs"] for j in scored if j.get("time_secs")]
    llm_calls = [j["llm_calls"] for j in scored if j.get("llm_calls") is not None]
    by_category: dict[str, list[bool]] = {}
    for j in scored:
        by_category.setdefault(j["category"], []).append(j["passed"])

    print("\n" + "=" * 60)
    for category in sorted(by_category):
        outcomes = by_category[category]
        print(f"  {category}: {sum(outcomes)}/{len(outcomes)} passed")
    answer_pass = sum(j["passed"] for j in scored)
    tool_pass = sum(j["tool_selection_pass"] for j in scored)
    print(
        f"\n  answers:    {answer_pass}/{len(scored)} passed"
        f" ({answer_pass / len(scored):.0%})"
    )
    print(
        f"  tool use:   {tool_pass}/{len(scored)} passed"
        f" ({tool_pass / len(scored):.0%})"
    )
    print(
        f"  latency:    p50={_percentile(latencies, 50):.1f}s"
        f" p95={_percentile(latencies, 95):.1f}s"
    )
    if llm_calls:
        print(
            f"  llm calls:  mean={sum(llm_calls) / len(llm_calls):.1f}"
            f" max={max(llm_calls)} per turn"
        )
    errors = [j for j in judged if j["verdicts"] is None]
    if errors:
        print(f"\n  WARNING: {len(errors)} cases produced no answer at runtime")
    print(f"\nSaved to {judged_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
