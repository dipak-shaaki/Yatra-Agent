"""
Runs eval/cases/*.jsonl against the running system over HTTP, exercising the
full stack (Turn Handler, agent loop, tools).

Each case runs in its own session (fresh history) — POST /api/v1/session,
then POST /api/v1/chat with the X-Session-Token header. The server must run
with EXPOSE_EVAL_CONTEXT=true for retrieved_context to be captured.

Captures each case's actual output, retrieved context, tools called and
latency for the judge/metrics step to score. It does not auto-grade
expected_answer — that is free text and hard to compare exactly.

All paths are anchored to the project root, so the script behaves the same
no matter where it is launched from; every knob is a CLI flag.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_cases(eval_dir: Path) -> list[dict]:
    """Load all evaluation cases from JSONL files."""
    cases = []
    for jsonl_file in sorted(eval_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def run_case(case: dict, client: httpx.Client, base_url: str, timeout: float) -> dict:
    start = time.perf_counter()
    try:
        session_token = (
            client.post(f"{base_url}/session", timeout=timeout)
            .raise_for_status()
            .json()["session_token"]
        )

        response = client.post(
            f"{base_url}/chat",
            json={"query": case["question"], "stream": False},
            headers={"X-Session-Token": session_token},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        answer = body.get("answer")
        retrieved_context = body.get("retrieved_context", [])
        tools_called = body.get("tools_called", [])
        llm_calls = body.get("llm_calls")
        error = None
    except Exception as e:
        answer = None
        retrieved_context = []
        tools_called = []
        llm_calls = None
        error = str(e)
    elapsed_s = round(time.perf_counter() - start, 2)

    return {
        "id": case["id"],
        "category": case.get("category"),
        "difficulty": case.get("difficulty"),
        "question": case["question"],
        "expected_answer": case.get("expected_answer"),
        "expected_tools": case.get("expected_tools", []),
        "requires_refusal": case.get("requires_refusal", False),
        "hedge": case.get("hedge", False),
        "actual_answer": answer,
        "retrieved_context": retrieved_context,
        "tools_called": tools_called,
        "llm_calls": llm_calls,
        "error": error,
        "time_secs": elapsed_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1",
        help="Base URL of the running Yatra API (default: %(default)s)",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / "cases",
        help="Directory of *.jsonl eval cases",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "eval" / "results.jsonl",
        help="Output results file (JSONL)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between cases (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Per-request HTTP timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N cases"
    )
    args = parser.parse_args()

    cases = load_cases(args.eval_dir)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"No cases found in {args.eval_dir}", file=sys.stderr)
        return 1

    print(f"Running {len(cases)} eval cases against {args.api_url}...\n")

    results = []
    with httpx.Client() as client:
        for index, case in enumerate(cases):
            print(f"[{case['id']}] ({case.get('category')}) {case['question']}")

            result = run_case(case, client, args.api_url, args.timeout)
            results.append(result)

            if result["error"]:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  time={result['time_secs']}s tools={result['tools_called']}")
                print(f"  expected: {(result['expected_answer'] or '')[:100]}")
                print(f"  actual:   {(result['actual_answer'] or '')[:100]}")

            print()

            # Wait between cases to reduce API/rate-limit pressure.
            if index < len(cases) - 1 and args.delay > 0:
                time.sleep(args.delay)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Saved {len(results)} results to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
