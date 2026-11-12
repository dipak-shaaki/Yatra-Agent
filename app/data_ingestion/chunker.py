"""
Parses a destination corpus markdown file (validated YAML frontmatter +
## sections) into a list of chunk dicts ready for embedding + Chroma storage.
"""

from pathlib import Path
from typing import Any

from app.data_ingestion.frontmatter import parse_frontmatter


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
    lines = body.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip() == "---" and lines[index + 1].startswith("*Sources:"):
            return "\n".join(lines[:index]).strip()
    return body.strip()


def _strip_title_line(body: str) -> str:
    """Drop the leading '# Title' line — it's document-level, not a section."""
    title, separator, remainder = body.partition("\n")
    if (
        separator
        and title.startswith("#")
        and len(title) > 1
        and title[1].isspace()
        and title[1:].strip()
    ):
        return remainder.strip()
    return body.strip()


def _split_sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Return intro text and level-two sections without regex backtracking."""
    lines = body.splitlines(keepends=True)
    headings: list[tuple[int, int, str]] = []
    offset = 0
    for line in lines:
        heading_line = line.rstrip("\r\n")
        if (
            heading_line.startswith("##")
            and len(heading_line) > 2
            and heading_line[2].isspace()
        ):
            title = heading_line[2:].strip()
            if title:
                headings.append((offset, offset + len(line), title))
        offset += len(line)

    if not headings:
        return body.strip(), []

    first_start = headings[0][0]
    sections = []
    for index, (_, content_start, title) in enumerate(headings):
        content_end = headings[index + 1][0] if index + 1 < len(headings) else len(body)
        sections.append((title, body[content_start:content_end].strip()))
    return body[:first_start].strip(), sections


def section_titles(body: str) -> list[str]:
    """Return normalized level-two section titles from markdown text."""
    _, sections = _split_sections(body)
    return [title for title, _ in sections]


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

    intro_text, sections = _split_sections(body)
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

    for header, section_text in sections:
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
