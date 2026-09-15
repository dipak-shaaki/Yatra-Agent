"""
Retrieval-level eval: Precision@K, Recall@K, MRR — computed directly
against hybrid_search(), no LLM calls, no server needed.

Deliberately skips word-overlap-based "groundedness"/"hallucination" scoring
(seen in generic RAG eval guides) since it can't distinguish a correct fact
from a wrong one using the same vocabulary — e.g. it would score "permits
cost NPR 50,000" as grounded even if the source says NPR 10,000, as long as
the words overlap. Faithfulness needs an LLM judge instead (built separately).
"""

import json
from pathlib import Path

from app.retrieval.hybrid_retriever import hybrid_search

EVAL_DIR = Path("eval/cases")
K_VALUES = [3, 5, 10]


def load_cases_with_source_docs() -> list[dict]:
    """Only cases with non-empty source_docs are meaningful for retrieval eval —
    out_of_scope/filler cases have none by design, so they're skipped here."""
    cases = []
    for jsonl_file in sorted(EVAL_DIR.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                if case.get("source_docs"):
                    cases.append(case)
    return cases


def precision_recall_mrr_at_k(
    retrieved_source_files: list[str], expected_docs: list[str], k: int
) -> dict:
    top_k = retrieved_source_files[:k]
    # retrieved chunk source_file looks like "manaslu_circuit.md" — expected_docs are "manaslu_circuit"
    top_k_stems = [f.replace(".md", "") for f in top_k]

    hits = [doc for doc in top_k_stems if doc in expected_docs]
    precision = len(hits) / k if k > 0 else 0
    recall = len(set(hits)) / len(set(expected_docs)) if expected_docs else 0

    rr = 0.0
    for rank, doc in enumerate(top_k_stems, start=1):
        if doc in expected_docs:
            rr = 1 / rank
            break

    return {"precision": precision, "recall": recall, "rr": rr}


def main():
    cases = load_cases_with_source_docs()
    print(f"Evaluating retrieval on {len(cases)} cases with labeled source_docs\n")

    results_by_k = {k: [] for k in K_VALUES}

    for case in cases:
        retrieved = hybrid_search(case["question"], n_results=max(K_VALUES))
        retrieved_source_files = [r["metadata"]["source_file"] for r in retrieved]

        print(f"[{case['id']}] {case['question']}")
        print(f"  expected: {case['source_docs']}")
        print(
            f"  retrieved (top 5): {[f.replace('.md', '') for f in retrieved_source_files[:5]]}"
        )

        for k in K_VALUES:
            metrics = precision_recall_mrr_at_k(
                retrieved_source_files, case["source_docs"], k
            )
            results_by_k[k].append(metrics)
            print(
                f"  @{k}: precision={metrics['precision']:.2f} recall={metrics['recall']:.2f} rr={metrics['rr']:.2f}"
            )
        print()

    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    for k in K_VALUES:
        avg_precision = sum(r["precision"] for r in results_by_k[k]) / len(
            results_by_k[k]
        )
        avg_recall = sum(r["recall"] for r in results_by_k[k]) / len(results_by_k[k])
        mrr = sum(r["rr"] for r in results_by_k[k]) / len(results_by_k[k])
        print(
            f"K={k}: Precision@{k}={avg_precision:.3f}  Recall@{k}={avg_recall:.3f}  MRR@{k}={mrr:.3f}"
        )


if __name__ == "__main__":
    main()
