"""
Parses a destination corpus markdown file (validated YAML frontmatter +
## sections) into a list of chunk dicts ready for embedding + Chroma storage.
"""

import re
from pathlib import Path
from typing import Any

from app.data_ingestion.frontmatter import parse_frontmatter

SECTION_SPLIT_RE = re.compile(r"^##[ \t]+([^\n]+?)[ \t]*$", re.MULTILINE)


def _flatten_metadata(
    frontmatter: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """Chroma metadata values must be str/int/float/bool — join lists into strings.

    None values are dropped entirely rather than stringified to "None",
    so absent attributes never pollute the stored metadata.
    """
    flat: dict[str, str | int | float | bool] = {}
    for key, value in frontmatter.items():
        if value is None:
            continue
        if isinstance(value, list):
            flat[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def _strip_sources_footer(body: str) -> str:
    """Drop a trailing '---\\n*Sources: ...*' footer if present."""
    return re.split(r"\n---[ \t]*\n\*Sources:[^\n]*(?:\n|$)", body, maxsplit=1)[
        0
    ].strip()


def _strip_title_line(body: str) -> str:
    """Drop the leading '# Title' line — it's document-level, not a section."""
    return re.sub(r"^#\s+.+?\n", "", body, count=1).strip()


def chunk_document(file_path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(file_path)
    return chunk_text(
        file_path.read_text(encoding="utf-8"),
        stem=file_path.stem,
        source_file=file_path.name,
    )


def chunk_text(raw_text: str, *, stem: str, source_file: str) -> list[dict[str, Any]]:
    """Chunk markdown from an in-memory string (no file write needed), so the
    upsert path can validate + embed before touching the corpus file."""
    frontmatter, body = parse_frontmatter(raw_text)
    body = _strip_sources_footer(body)
    body = _strip_title_line(body)

    dest_name = frontmatter.name
    metadata_base = _flatten_metadata(frontmatter.model_dump())

    chunks = []

    first_header_match = SECTION_SPLIT_RE.search(body)
    intro_text = (
        body[: first_header_match.start()].strip()
        if first_header_match
        else body.strip()
    )
    if intro_text:
        chunks.append(
            {
                "id": f"{stem}__overview",
                "text": f"{dest_name} — Overview:\n{intro_text}",
                "metadata": {
                    **metadata_base,
                    "section_title": "Overview",
                    "source_file": source_file,
                },
            }
        )

    headers = SECTION_SPLIT_RE.findall(body)
    sections = SECTION_SPLIT_RE.split(body)[1:]

    for i, header in enumerate(headers):
        section_text = sections[2 * i + 1].strip() if len(sections) > 2 * i + 1 else ""
        if not section_text:
            continue
        chunk_id = f"{stem}__{header.lower().replace(' ', '_').replace('/', '_')}"
        chunks.append(
            {
                "id": chunk_id,
                "text": f"{dest_name} — {header}:\n{section_text}",
                "metadata": {
                    **metadata_base,
                    "section_title": header,
                    "source_file": source_file,
                },
            }
        )

    return chunks


def chunk_corpus_dir(corpus_dir: str | Path) -> list[dict[str, Any]]:
    """Chunk every .md file in the corpus directory."""
    corpus_dir = Path(corpus_dir)
    all_chunks = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        all_chunks.extend(chunk_document(md_file))
    return all_chunks
