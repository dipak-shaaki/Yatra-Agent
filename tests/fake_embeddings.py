"""Deterministic hash-based embedding stand-in for tests.

Lets real Chroma and hybrid retrieval be exercised without downloading
torch/sentence-transformers. Same text always maps to the same unit vector;
shared words give a little similarity signal.
"""

import hashlib
import math
import re

DIM = 32
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def fake_embedding(text: str) -> list[float]:
    vector = [0.0] * DIM
    for token in _TOKEN_RE.findall(text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        vector[digest[0] % DIM] += 1.0
        vector[digest[1] % DIM] += 0.5
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def fake_embed_documents(texts: list[str]) -> list[list[float]]:
    return [fake_embedding(text) for text in texts]
