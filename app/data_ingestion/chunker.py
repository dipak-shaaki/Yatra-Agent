"""
Parses a destination corpus markdown file (YAML frontmatter + ## sections)
into a list of chunk dicts ready for embedding + Chroma storage.
"""

import re
from pathlib import Path
from typing import Any

import yaml

SECTION_SPLIT_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _flatten_metadata(
    frontmatter: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """Chroma metadata values must be str/int/float/bool — join lists into strings.

    None values are dropped entirely rather than stringified to "None",
    so absent attributes never pollute the stored metadata.
    """
    flat = {}
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


def _parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """Split leading YAML frontmatter (between --- markers) from the rest of the doc."""
    if not raw_text.startswith("---"):
        raise ValueError("Document missing YAML frontmatter (must start with '---')")

    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Malformed frontmatter — expected opening and closing '---'")

    frontmatter = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return frontmatter, body


def _strip_sources_footer(body: str) -> str:
    """Drop a trailing '---\\n*Sources: ...*' footer if present."""
    return re.split(r"\n---\s*\n\*Sources:.*", body, flags=re.DOTALL)[0].strip()


def _strip_title_line(body: str) -> str:
    """Drop the leading '# Title' line — it's document-level, not a section."""
    return re.sub(r"^#\s+.+?\n", "", body, count=1).strip()


def chunk_document(file_path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding="utf-8")

    frontmatter, body = _parse_frontmatter(raw_text)
    body = _strip_sources_footer(body)
    body = _strip_title_line(body)

    dest_name = frontmatter.get("name", file_path.stem)
    metadata_base = _flatten_metadata(frontmatter)

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
                "id": f"{file_path.stem}__overview",
                "text": f"{dest_name} — Overview:\n{intro_text}",
                "metadata": {
                    **metadata_base,
                    "section_title": "Overview",
                    "source_file": file_path.name,
                },
            }
        )

    headers = SECTION_SPLIT_RE.findall(body)
    sections = SECTION_SPLIT_RE.split(body)[1:]

    for i, header in enumerate(headers):
        section_text = sections[2 * i + 1].strip() if len(sections) > 2 * i + 1 else ""
        if not section_text:
            continue
        chunk_id = (
            f"{file_path.stem}__{header.lower().replace(' ', '_').replace('/', '_')}"
        )
        chunks.append(
            {
                "id": chunk_id,
                "text": f"{dest_name} — {header}:\n{section_text}",
                "metadata": {
                    **metadata_base,
                    "section_title": header,
                    "source_file": file_path.name,
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
