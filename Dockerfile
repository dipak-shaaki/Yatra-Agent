# Yatra API server. The UI is not containerized (it is a dev frontend);
# run it on the host against this container's published port.

FROM python:3.13-slim

# uv pinned to the version the lockfile was resolved with
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

WORKDIR /app

# Dependencies first: this layer caches independently of code changes.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-build

COPY app/ app/
COPY scripts/build_index.py scripts/
COPY data/corpus/ data/corpus/

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface

EXPOSE 8000

# Rebuild the Chroma index from the baked-in corpus (idempotent — the
# embedding model and chroma db live in named volumes), then serve.
CMD ["sh", "-c", "python -m scripts.build_index && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
