from app.retrieval.hybrid_retriever import hybrid_search

TEST_QUERIES = [
    "TIMS card requirement",
    "5106m altitude pass",
    "easy trek near Pokhara",
]

for query in TEST_QUERIES:
    print(f"\n{'=' * 60}\nQuery: {query}\n{'=' * 60}")
    results = hybrid_search(query, n_results=3)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['id']}  (rrf_score={r['score']:.4f})")
        print(r["text"][:150])