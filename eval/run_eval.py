"""
Runs eval/cases/*.jsonl against the real running system (via HTTP, so it
exercises the full stack — Turn Handler, agent loop, tools — not just
individual functions in isolation).

This is a lightweight first pass: it captures actual system output per case
so a human (or a follow-up LLM-as-judge step) can score faithfulness/
correctness. It does NOT auto-grade "expected_answer" matches, since that's
free-text and hard to compare exactly — see the printed comparison instead.
"""
import json
import time
from pathlib import Path

import httpx

API_URL = "http://localhost:8000/api/v1/chat"
EVAL_DIR = Path("eval/cases")
RESULTS_PATH = Path("eval/results.jsonl")


def load_cases() -> list[dict]:
    cases = []
    for jsonl_file in sorted(EVAL_DIR.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def run_case(case: dict) -> dict:
    sender = f"eval_{case['id']}"
    start = time.perf_counter()
    try:
        response = httpx.post(API_URL, params={"query": case["question"], "sender": sender}, timeout=30)
        response.raise_for_status()
        answer = response.json()["answer"]
        error = None
    except Exception as e:
        answer = None
        error = str(e)
    elapsed_s = round(time.perf_counter() - start, 2)

    return {
        "id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "actual_answer": answer,
        "error": error,
        "time_secs": elapsed_s,
    }


def main():
    cases = load_cases()
    print(f"Running {len(cases)} eval cases...\n")

    results = []
    for case in cases:
        print(f"[{case['id']}] ({case['category']}) {case['question']}")
        result = run_case(case)
        results.append(result)

        if result["error"]:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  time={result['time_secs']}s")
            print(f"  expected: {result['expected_answer'][:100]}")
            print(f"  actual:   {(result['actual_answer'] or '')[:100]}")
        print()

    with open(RESULTS_PATH, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)

    print(f"Saved {len(results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()