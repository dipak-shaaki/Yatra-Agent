"""
Retrieval eval: chunk-level Precision@K / Recall@K / MRR of the retriever
against the labeled `source_chunks` of every eval case that has them.
No LLM calls, no server needed — this isolates retrieval quality.

Also doubles as an ablation harness (--retriever dense|bm25 shows what the
other half of hybrid buys) and as a CI gate: --gate-* thresholds exit
nonzero when missed. Run as `python -m scripts.eval_retrieval` from the
project root, so `app` imports resolve without sys.path hacks.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from app.retrieval.bm25_index import get_bm25_index
from app.retrieval.hybrid_retriever import hybrid_search

# The agent's tools retrieve n_results=5; score retrieval at that size.
K = 5


def _retrievers() -> dict[str, Callable[[str], list[str]]]:
    """Each retriever maps a query to its top-K chunk ids (ranked)."""
    from app.db.chroma.client import query_collection
    from app.retrieval.embeddings import embed_query

    def dense(query: str) -> list[str]:
        return query_collection(embed_query(query), n_results=K)["ids"][0]

    def bm25(query: str) -> list[str]:
        return [chunk_id for chunk_id, _ in get_bm25_index().search(query, K)]

    return {
        "hybrid": lambda q: [r["id"] for r in hybrid_search(q, n_results=K)],
        "dense": dense,
        "bm25": bm25,
    }


def load_cases_with_chunks(eval_dir: Path) -> list[dict]:
    """Only cases that are retrieval-scoreable: they must label source_chunks
    AND expect search_destinations — the only tool that runs hybrid_search.
    Discovery/budget questions go through the metadata tools instead, and
    out_of_scope/filler/unclear cases have no labels by design."""
    cases = []
    for jsonl_file in sorted(eval_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    if case.get("source_chunks") and "search_destinations" in case.get(
                        "expected_tools", []
                    ):
                        cases.append(case)
    return cases


def precision_recall_rr(retrieved: list[str], expected: list[str]) -> dict:
    top_k = retrieved[:K]
    hits = [chunk_id for chunk_id in top_k if chunk_id in expected]
    rr = 0.0
    for rank, chunk_id in enumerate(top_k, start=1):
        if chunk_id in expected:
            rr = 1 / rank
            break
    return {
        "precision": len(hits) / K,
        "recall": len(set(hits)) / len(set(expected)),
        "rr": rr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / "cases",
        help="Directory of *.jsonl eval cases",
    )
    parser.add_argument(
        "--retriever",
        choices=sorted(_retrievers()),
        default="hybrid",
        help="Retrieval path to score (default: %(default)s)",
    )
    parser.add_argument(
        "--gate-precision",
        type=float,
        default=0.15,
        help=f"Fail if mean Precision@{K} below this (default: %(default)s)",
    )
    parser.add_argument(
        "--gate-recall",
        type=float,
        default=0.85,
        help=f"Fail if mean Recall@{K} below this (default: %(default)s)",
    )
    parser.add_argument(
        "--gate-mrr",
        type=float,
        default=0.70,
        help=f"Fail if mean MRR@{K} below this (default: %(default)s)",
    )
    args = parser.parse_args()

    # Gate floors are calibrated ~10-15% below current hybrid numbers
    # (P=0.20 / R=0.96 / MRR=0.83). Precision tops out near |expected|/K
    # (usually 1 labeled chunk per question) — do not "fix" the low bar by
    # raising it without relabeling more chunks per case.
    cases = load_cases_with_chunks(args.eval_dir)
    retrieve = _retrievers()[args.retriever]
    print(f"Scoring {args.retriever} retrieval on {len(cases)} cases (K={K})\n")

    metrics = []
    misses = []
    for case in cases:
        retrieved = retrieve(case["question"])
        m = precision_recall_rr(retrieved, case["source_chunks"])
        metrics.append(m)
        marker = ""
        if m["rr"] == 0.0:  # none of the labeled chunks made the top K
            marker = "  <-- MISS"
            misses.append(case["id"])
        print(
            f"[{case['id']}] p={m['precision']:.2f} r={m['recall']:.2f} "
            f"rr={m['rr']:.2f}{marker}"
        )

    n = len(metrics)
    mean_precision = sum(m["precision"] for m in metrics) / n
    mean_recall = sum(m["recall"] for m in metrics) / n
    mean_rr = sum(m["rr"] for m in metrics) / n
    print("\n" + "=" * 60)
    print(
        f"{args.retriever}: Precision@{K}={mean_precision:.3f}  "
        f"Recall@{K}={mean_recall:.3f}  MRR@{K}={mean_rr:.3f}  "
        f"({len(misses)} full misses: {', '.join(misses) or 'none'})"
    )

    gates = {
        f"precision>={args.gate_precision}": mean_precision >= args.gate_precision,
        f"recall>={args.gate_recall}": mean_recall >= args.gate_recall,
        f"mrr>={args.gate_mrr}": mean_rr >= args.gate_mrr,
    }
    failed = [name for name, ok in gates.items() if not ok]
    for name, ok in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        print(f"\nRetrieval gate FAILED for {args.retriever}")
        return 1
    print(f"\nRetrieval gate passed for {args.retriever}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
