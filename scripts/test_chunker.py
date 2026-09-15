from app.core.config import get_settings
from app.data_ingestion.chunker import chunk_corpus_dir

chunks = chunk_corpus_dir(get_settings().corpus_dir)
print(f"Total chunks: {len(chunks)}")
print(f"From {len({c['metadata']['source_file'] for c in chunks})} files")
print()
print("Sample chunk:")
print(chunks[0]["id"])
print(chunks[0]["metadata"])
print(chunks[0]["text"][:200])
