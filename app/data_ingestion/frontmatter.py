"""
Validated YAML frontmatter for destination corpus documents.

Single source of truth for the corpus data contract: controlled vocabularies
(provinces, tags), numeric budget/duration fields, and the alias list used
for name resolution. `chunk_document()` and the `/documents/upsert`
endpoint both parse through here, so invalid frontmatter cannot reach the
index.
"""

from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Controlled vocabularies — extend these (and the corpus) to add values.
PROVINCES = ("Bagmati", "Gandaki", "Karnali", "Lumbini", "Sudurpashchim")
ALLOWED_TAGS = (
    "trek",
    "pilgrimage",
    "hill town",
    "heritage village",
    "historic town",
    "lake",
)


class DestinationFrontmatter(BaseModel):
    """Every field a corpus document must declare. `extra="forbid"` so a
    typo'd key fails at ingestion instead of silently vanishing."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    aliases: list[str] = []
    province: str
    district: str
    tags: list[str] = Field(min_length=1)
    difficulty: int = Field(ge=1, le=5)  # 1 = easy ... 5 = very hard
    best_time_to_visit: list[str]
    avoided_season: list[str]
    route: str
    distance_from_kathmandu: str
    time_required: str
    days_min: int = Field(ge=1)
    days_max: int = Field(ge=1)
    max_altitude_m: int | None = None
    permits_required: list[str] = []
    estimated_budget_npr: str = Field(min_length=1)
    budget_min_npr: int = Field(ge=0)
    budget_max_npr: int = Field(ge=0)
    budget_per_day: bool = False
    last_updated: str = Field(min_length=1)

    @field_validator("province")
    @classmethod
    def _province_in_vocab(cls, v: str) -> str:
        if v not in PROVINCES:
            raise ValueError(f"province must be one of {PROVINCES}, got {v!r}")
        return v

    @field_validator("tags")
    @classmethod
    def _tags_in_vocab(cls, v: list[str]) -> list[str]:
        unknown = [t for t in v if t not in ALLOWED_TAGS]
        if unknown:
            raise ValueError(f"unknown tags {unknown}; allowed: {ALLOWED_TAGS}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate tags are not allowed")
        return v

    @model_validator(mode="after")
    def _ranges_consistent(self) -> "DestinationFrontmatter":
        if self.days_min > self.days_max:
            raise ValueError("days_min must be <= days_max")
        if self.budget_min_npr > self.budget_max_npr:
            raise ValueError("budget_min_npr must be <= budget_max_npr")
        return self


def parse_frontmatter(raw_text: str) -> tuple[DestinationFrontmatter, str]:
    """Split and validate leading YAML frontmatter; returns (model, body).

    Raises ValueError on missing/malformed frontmatter or schema violations.
    """
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Document missing YAML frontmatter (must start with '---')")

    end = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = idx
            break
    if end is None:
        raise ValueError("Malformed frontmatter — expected opening and closing '---'")

    try:
        data: dict[str, Any] = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Frontmatter is not valid YAML: {e}") from e

    try:
        frontmatter = DestinationFrontmatter.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid frontmatter: {e}") from e

    body = "\n".join(lines[end + 1 :]).strip()
    return frontmatter, body
