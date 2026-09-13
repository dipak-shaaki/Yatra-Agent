"""
Runs the faithfulness judge against eval/results.jsonl — re-retrieves
context per question (cheap, no LLM call) and judges the already-captured
answer against it (one LLM call per case).
"""
import json
import time
from pathlib import Path

from app.retrieval.hybrid_retriever import hybrid_search
from eval.faithfulness_judge import judge_faithfulness

RESULTS_PATH = Path("eval/results.jsonl")
FAITHFULNESS_RESULTS_PATH = Path("eval/faithfulness_results.jsonl")


def main():
    results = []
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Judging faithfulness for {len(results)} cases...\n")

    # Faithfulness only makes sense for cases where the answer is supposed to
    # draw on retrieved facts — refusals (out_of_scope, adversarial) don't
    # have a meaningful "correct context" to be judged against.
    SKIP_CATEGORIES = {"out_of_scope", "adversarial"}

    judged = []
    for r in results:
        if r.get("error") or not r.get("actual_answer"):
            print(f"[{r['id']}] skipped — no answer to judge")
            continue
        if r["category"] in SKIP_CATEGORIES:
            print(f"[{r['id']}] skipped — {r['category']} is a refusal case, not fact-grounded")
            continue

        retrieved = hybrid_search(r["question"], n_results=5)
        context = "\n\n".join(chunk["text"] for chunk in retrieved)

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