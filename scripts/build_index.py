"""
Chunk the corpus, embed every chunk, and upsert into Chroma.
"""

from app.data_ingestion.chunker import chunk_corpus_dir
from app.db.chroma.client import reset_collection, upsert_chunks
from app.retrieval.embeddings import embed_documents


def main():
    print("Chunking corpus...")
    chunks = chunk_corpus_dir("data/corpus")
    print(f"  {len(chunks)} chunks from corpus")

    print("Embedding chunks (first run downloads the model, may take a bit)...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(texts)
    print(f"  embedded {len(embeddings)} chunks, dim={len(embeddings[0])}")

    print("Resetting and rebuilding Chroma collection...")
    reset_collection()
    upsert_chunks(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )
    print("Done. Index persisted to disk.")


if __name__ == "__main__":
    main()
