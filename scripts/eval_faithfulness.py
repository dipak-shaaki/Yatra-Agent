"""
Runs the faithfulness judge against eval/results.jsonl, using the
retrieved_context each case actually captured during its real run —
not a separate re-retrieval, which previously caused inconsistent
verdicts on the same underlying facts.
"""
import json
import time
from pathlib import Path

from eval.faithfulness_judge import judge_faithfulness

RESULTS_PATH = Path("eval/results.jsonl")
FAITHFULNESS_RESULTS_PATH = Path("eval/faithfulness_results.jsonl")

SKIP_CATEGORIES = {"out_of_scope", "adversarial"}


def main():
    results = []
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Judging faithfulness for {len(results)} cases...\n")

    judged = []
    for r in results:
        if r.get("error") or not r.get("actual_answer"):
            print(f"[{r['id']}] skipped — no answer to judge")
            continue
        if r["category"] in SKIP_CATEGORIES:
            print(f"[{r['id']}] skipped — {r['category']} is a refusal case, not fact-grounded")
            continue

        context_chunks = r.get("retrieved_context", [])
        if not context_chunks:
            print(f"[{r['id']}] skipped — no retrieved_context captured (Turn Handler short-circuit or empty retrieval)")
            continue

        context = "\n\n".join(context_chunks)
        verdict = judge_faithfulness(context, r["actual_answer"])
        judged.append({**r, "faithfulness": verdict})

        print(f"[{r['id']}] faithful={verdict['faithful']}")
        if verdict["unsupported_claims"]:
            print(f"  unsupported: {verdict['unsupported_claims']}")
        print()

        time.sleep(2)

    with open(FAITHFULNESS_RESULTS_PATH, "w") as f:
        for j in judged:
            f.write(json.dumps(j) + "\n")

    faithful_count = sum(1 for j in judged if j["faithfulness"]["faithful"] is True)
    print(f"\n{faithful_count}/{len(judged)} judged faithful")
    print(f"Saved to {FAITHFULNESS_RESULTS_PATH}")


if __name__ == "__main__":
    main()