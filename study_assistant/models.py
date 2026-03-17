from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MaterialType(str, Enum):
    LECTURE_NOTES = "lecture_notes"
    EXERCISES = "exercises"
    SOLUTIONS = "solutions"


class TutorMode(str, Enum):
    HINT = "Hint"
    GUIDED = "Guided"
    FULL_SOLUTION = "Full solution"


@dataclass
class Course:
    course_id: str
    name: str
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Course":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Material:
    material_id: str
    course_id: str
    material_type: str
    filename: str
    stored_path: str
    checksum: str
    ingest_status: str
    chunk_count: int
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Material":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MistakePattern:
    pattern_id: str
    course_id: str
    concept: str
    mistake_label: str
    explanation: str
    evidence: str
    recurrence_count: int
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MistakePattern":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceSnippet:
    source: str
    page: int | None
    material_type: str
    excerpt: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceSnippet":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TutorResponse:
    response_type: str
    plain_language_explanation: str
    concepts: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    diagnosis: str = ""
    mistake_label: str = ""
    next_step: str = ""
    confidence_note: str = ""
    sources: list[SourceSnippet] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TutorResponse":
        source_dicts = data.get("sources", [])
        return cls(
            response_type=data.get("response_type", "mixed"),
            plain_language_explanation=data.get("plain_language_explanation", ""),
            concepts=list(data.get("concepts", [])),
            prerequisites=list(data.get("prerequisites", [])),
            steps=list(data.get("steps", [])),
            diagnosis=data.get("diagnosis", ""),
            mistake_label=data.get("mistake_label", ""),
            next_step=data.get("next_step", ""),
            confidence_note=data.get("confidence_note", ""),
            sources=[SourceSnippet.from_dict(item) for item in source_dicts],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [source.to_dict() for source in self.sources]
        return payload


@dataclass
class IngestResult:
    status: str
    indexed_chunk_count: int
    material: Material | None = None
    message: str = ""
