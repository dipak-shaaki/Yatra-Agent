# Evaluation baseline

Committed reference point for retrieval quality. Re-run the commands below
after changing the corpus, chunker, embedding model, fusion, or case labels,
and update the table (keep the old SHA in the git history rather than
silently editing numbers in place).

- **Code:** `3ad513e` (`main`)
- **Corpus tree:** `20871a8` (`git rev-parse 3ad513e:data/corpus`)
- **Date:** 2026-11-12
- **Hardware:** local Linux, CPU-only embeddings
- **Scored cases:** 176 (25 hand-written retrieval-routed + 145 generated
  from corpus frontmatter), K=5

## Settings

| Setting | Value |
| --- | --- |
| `embedding_model` | `BAAI/bge-base-en-v1.5` (768-dim, query prefix for retrieval) |
| Chroma | `1.5.9`, persistent local, collection `yatra_destinations`, cosine |
| `candidate_pool` | 10 per ranker before fusion |
| fusion | RRF, `rrf_k=60`, score-free |
| `search_n_results` | 5 |
| bootstrap | 1000 resamples, seed 42, 95% percentile CI |
| gate (CI, hybrid only) | P≥0.15 / R≥0.85 / MRR≥0.70 |

## Commands

```bash
python -m scripts.build_index                     # fresh Chroma from corpus
python -m scripts.eval_retrieval --retriever hybrid
python -m scripts.eval_retrieval --retriever dense
python -m scripts.eval_retrieval --retriever bm25
```

## Retrieval results

| Retriever | Precision@5 | Recall@5 | MRR@5 | Full misses |
| --- | --- | --- | --- | --- |
| **hybrid (gated)** | **0.193** [0.186, 0.199] | **0.955** [0.920, 0.983] | **0.751** [0.697, 0.798] | 8 |
| dense | 0.200 [0.195, 0.205] | 0.986 [0.969, 1.000] | 0.800 [0.755, 0.839] | 2 |
| BM25 | 0.164 [0.151, 0.176] | 0.812 [0.750, 0.869] | 0.617 [0.549, 0.679] | 33 |

Hybrid full misses: `discovery_008`, `retrieval_annapurna_base_camp_budget`,
`retrieval_badimalika_avoid_season`, `retrieval_badimalika_altitude_exact`,
`retrieval_bandipur_budget`, `retrieval_gorkha_duration`,
`retrieval_kori_altitude`, `retrieval_panauti_budget`.

Reading the numbers honestly: precision is capped near 1/K because most cases
label one chunk. Dense ranks best on this expanded set (the generated queries
share a lot of surface vocabulary with their target chunks), BM25 alone misses
the gate, and hybrid sits between them. Hybrid remains the served path and the
only gated one; the other rows are the committed ablation, not targets.

## Behavioral / LLM-judge half

Not included in this baseline: the full run needs a valid `GROQ_API_KEY` and
a running server. When available:

```bash
EXPOSE_EVAL_CONTEXT=true uvicorn app.main:app        # terminal 1
python -m scripts.run_eval --eval-dir eval/cases     # terminal 2
python -m scripts.eval_answers
```

Record afterwards: p50/p95 latency, per-category pass rates, answer pass rate,
exact-set tool-selection rate, and mean/max LLM calls per turn.
