"""Data models for knowledge-graph retrieval."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentic_rag.kg.schema import REQUIRED_RESULT_ALIASES, SCHEMA_VERSION


class KGQueryStatus(str, Enum):
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    GENERATION_ERROR = "generation_error"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"


class KGSectionResult(BaseModel):
    """Validated Section returned by a KG retrieval query."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    section_uid: str
    document_id: str
    section_id: str | None = None
    title: str | None = None
    text: str
    matched_concepts: list[str] = Field(default_factory=list)
    score: float | None = None
    rank: int | None = Field(default=None, ge=1)

    @field_validator("section_uid", "document_id", "text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Value must be a non-empty string")
        return normalized

    @field_validator("section_id", "title")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()
        return normalized or None

    @field_validator("matched_concepts", mode="before")
    @classmethod
    def normalize_matched_concepts(cls, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            values = [value]
        else:
            values = list(value)

        normalized: list[str] = []
        seen: set[str] = set()

        for item in values:
            if item is None:
                continue

            concept = str(item).strip()
            if not concept:
                continue

            key = concept.casefold()
            if key in seen:
                continue

            seen.add(key)
            normalized.append(concept)

        return normalized

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        rank: int | None = None,
    ) -> "KGSectionResult":
        """Create a result from a Neo4j record or mapping."""

        data = dict(record)

        missing_aliases = [
            alias
            for alias in REQUIRED_RESULT_ALIASES
            if alias not in data
        ]
        if missing_aliases:
            raise ValueError(
                "KG query result is missing required aliases: "
                + ", ".join(missing_aliases)
            )

        return cls(
            section_uid=data["section_uid"],
            document_id=data["document_id"],
            section_id=data["section_id"],
            title=data["title"],
            text=data["text"],
            matched_concepts=data["matched_concepts"],
            score=data["score"],
            rank=rank,
        )

    @property
    def heading(self) -> str:
        """Return a readable section heading."""

        parts = [
            part
            for part in (self.section_id, self.title)
            if part
        ]
        return " ".join(parts).strip()

    @property
    def page_content(self) -> str:
        """Return the future LangChain Document text."""

        if self.heading:
            return f"{self.heading}\n\n{self.text}"
        return self.text


class KGQueryTrace(BaseModel):
    """Diagnostic record for one natural-language KG retrieval query."""

    model_config = ConfigDict(extra="forbid")

    question: str
    generated_cypher: str | None = None
    status: KGQueryStatus
    results: list[KGSectionResult] = Field(default_factory=list)

    latency_ms: float = Field(ge=0)
    top_k: int = Field(ge=1)

    model_name: str = "gpt-4.1-mini"
    schema_version: str = SCHEMA_VERSION

    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question", "model_name")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Value must be a non-empty string")
        return normalized

    @field_validator("generated_cypher", "error")
    @classmethod
    def normalize_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "KGQueryTrace":
        failure_statuses = {
            KGQueryStatus.GENERATION_ERROR,
            KGQueryStatus.VALIDATION_ERROR,
            KGQueryStatus.EXECUTION_ERROR,
        }

        if self.status in failure_statuses and not self.error:
            raise ValueError(
                f"Status '{self.status.value}' requires an error message"
            )

        if self.status in {
            KGQueryStatus.SUCCESS,
            KGQueryStatus.NO_RESULTS,
        } and self.error:
            raise ValueError(
                f"Status '{self.status.value}' cannot contain an error"
            )

        if self.status == KGQueryStatus.SUCCESS and not self.results:
            raise ValueError(
                "A successful query must contain at least one result"
            )

        if self.status == KGQueryStatus.NO_RESULTS and self.results:
            raise ValueError(
                "A no_results query cannot contain results"
            )

        return self

    @property
    def returned_count(self) -> int:
        return len(self.results)

