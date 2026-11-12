"""
Corpus-level metadata derived from the destination frontmatter: canonical
names, aliases, controlled-vocabulary values, numeric budget/duration
ranges, and the section titles used for comparisons.

Everything here reads the corpus directory at call time, cached against the
shared corpus version (see corpus_version.py), so a document upsert in one
worker is visible to tools, prompts, and BM25 indexes in every worker on
their next call — no restart, no per-process staleness.
"""

import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.data_ingestion.chunker import SECTION_SPLIT_RE, chunk_corpus_dir
from app.data_ingestion.corpus_version import current_corpus_version
from app.data_ingestion.frontmatter import parse_frontmatter


@dataclass(frozen=True)
class Destination:
    name: str
    aliases: tuple[str, ...]
    province: str
    district: str
    tags: tuple[str, ...]
    difficulty: int  # 1 = easy ... 5 = very hard
    best_time_to_visit: tuple[str, ...]
    avoided_season: tuple[str, ...]
    route: str
    distance_from_kathmandu: str
    time_required: str
    days_min: int
    days_max: int
    max_altitude_m: int | None
    permits_required: tuple[str, ...]
    estimated_budget_npr: str
    budget_min_npr: int
    budget_max_npr: int
    budget_per_day: bool
    last_updated: str
    source_file: str
    sections: tuple[str, ...]


@dataclass(frozen=True)
class CorpusSnapshot:
    """One consistent view of the markdown corpus at a given version."""

    version: str
    destinations: tuple[Destination, ...]
    chunks: tuple[dict, ...]


def _load_snapshot(corpus_dir: str, version: str) -> CorpusSnapshot:
    destinations = []
    for md_file in sorted(Path(corpus_dir).glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(raw)
        data: dict[str, Any] = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in frontmatter.model_dump().items()
        }
        destinations.append(
            Destination(
                **data,
                source_file=md_file.name,
                sections=tuple(SECTION_SPLIT_RE.findall(body)),
            )
        )
    return CorpusSnapshot(
        version=version,
        destinations=tuple(destinations),
        chunks=tuple(chunk_corpus_dir(corpus_dir)),
    )


_lock = threading.Lock()
_cached: CorpusSnapshot | None = None


def get_corpus_snapshot() -> CorpusSnapshot:
    """The current corpus view, rebuilt when the shared version changes."""
    global _cached
    corpus_dir = get_settings().corpus_dir
    version = current_corpus_version()
    with _lock:
        if _cached is not None and _cached.version == version:
            return _cached
        _cached = _load_snapshot(corpus_dir, version)
        return _cached


def get_destinations() -> tuple[Destination, ...]:
    return get_corpus_snapshot().destinations


def get_corpus_chunks() -> tuple[dict, ...]:
    """Every corpus chunk as {"id", "text", "metadata"} — used by tools that
    need full section text without going through Chroma (and so cannot drift
    from an out-of-date index)."""
    return get_corpus_snapshot().chunks


def reset_corpus_meta() -> None:
    """Call after corpus files change on disk (e.g. document upsert)."""
    global _cached
    with _lock:
        _cached = None


def destination_names() -> tuple[str, ...]:
    return tuple(d.name for d in get_destinations())


def resolve_destination(raw_name: str) -> str | None:
    """Exact corpus name for a casual/raw name or alias, else None."""
    raw = raw_name.strip().lower()
    for dest in get_destinations():
        if raw == dest.name.lower() or raw in {a.lower() for a in dest.aliases}:
            return dest.name
    return None


def comparison_sections() -> tuple[str, ...]:
    """Section titles present in at least half the destinations — the
    schema-driven comparison vocabulary, replacing a hardcoded list."""
    destinations = get_destinations()
    threshold = max(1, len(destinations) // 2)
    counts = Counter(s for dest in destinations for s in dest.sections)
    return tuple(section for section, n in counts.items() if n >= threshold)
