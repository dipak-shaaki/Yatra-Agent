from pydantic import BaseModel, field_validator


class DocumentUpsertRequest(BaseModel):
    filename: str  # e.g. "manaslu_circuit" (no .md extension)
    content: str   # full markdown content: YAML frontmatter + ## sections

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("filename must be alphanumeric (with _ or -), no extension or path characters")
        return v