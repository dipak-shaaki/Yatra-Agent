"""
Manual sanity check: embed a query, search Chroma, print top results.
"""
from app.db.chroma.client import query_collection
from app.retrieval.embeddings import embed_query

TEST_QUERIES = [
    "easy trek near Pokhara",
    "what permits do I need for Manaslu",
    "budget for a lake trip",
]

for query in TEST_QUERIES:
    print(f"\n{'=' * 60}\nQuery: {query}\n{'=' * 60}")
    embedding = embed_query(query)
    results = query_collection(embedding, n_results=3)

    for i, (doc_id, doc_text, distance) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["distances"][0])
    ):
        print(f"\n[{i+1}] {doc_id}  (distance={distance:.4f})")
        print(doc_text[:150])