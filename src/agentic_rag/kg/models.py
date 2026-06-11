"""Data models for knowledge-graph retrieval."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentic_rag.kg.schema import REQUIRED_RESULT_ALIASES, SCHEMA_VERSION


KGRankingMode = Literal["concept_match", "weighted_match"]


class KGQueryStatus(str, Enum):
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    GENERATION_ERROR = "generation_error"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"


class KGRetrievalScores(BaseModel):
    """Alternative deterministic scores computed for one Section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_match: float = 0.0
    weighted_match: float = 0.0


class KGMatchDiagnostic(BaseModel):
    """Diagnostic information explaining why a concept matched."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_term: str
    concept_name: str | None = None
    matched_value: str | None = None
    match_type: str
    weight: float

    @field_validator("query_term", "match_type")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Value must be a non-empty string")
        return normalized

    @field_validator("concept_name", "matched_value")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class KGSectionResult(BaseModel):
    """Validated Section returned by a KG retrieval query."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    section_uid: str
    document_id: str
    section_id: str | None = None
    printed_section_id: str | None = None
    title: str | None = None
    level: int | None = None
    text: str
    page_start: int | None = None
    page_end: int | None = None
    part_index: int | None = None
    part_count: int | None = None
    matched_concepts: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    score: float | None = None
    score_type: KGRankingMode | None = None
    scores: KGRetrievalScores | None = None
    match_diagnostics: list[KGMatchDiagnostic] = Field(default_factory=list)
    rank: int | None = Field(default=None, ge=1)

    @field_validator("section_uid", "document_id")
    @classmethod
    def validate_required_identifier(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Value must be a non-empty string")
        return normalized

    @field_validator("text")
    @classmethod
    def validate_text_preserving_content(cls, value: str) -> str:
        original = str(value)
        if not original.strip():
            raise ValueError("Section text must be non-empty")
        return original

    @field_validator("section_id", "printed_section_id", "title")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()
        return normalized or None

    @field_validator("matched_concepts", "matched_terms", mode="before")
    @classmethod
    def normalize_unique_strings(cls, value: Any) -> list[str]:
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

            text = str(item).strip()
            if not text:
                continue

            key = text.casefold()
            if key in seen:
                continue

            seen.add(key)
            normalized.append(text)

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

        scores = data.get("scores")
        if scores is None:
            concept_score = data.get("concept_match_score")
            weighted_score = data.get("weighted_match_score")
            if concept_score is not None or weighted_score is not None:
                scores = {
                    "concept_match": float(concept_score or 0.0),
                    "weighted_match": float(weighted_score or 0.0),
                }

        return cls(
            section_uid=data["section_uid"],
            document_id=data["document_id"],
            section_id=data.get("section_id"),
            printed_section_id=data.get("printed_section_id"),
            title=data.get("title"),
            level=data.get("level"),
            text=data["text"],
            page_start=data.get("page_start"),
            page_end=data.get("page_end"),
            part_index=data.get("part_index"),
            part_count=data.get("part_count"),
            matched_concepts=data["matched_concepts"],
            matched_terms=data.get("matched_terms", []),
            score=data["score"],
            score_type=data.get("score_type"),
            scores=scores,
            match_diagnostics=data.get("match_diagnostics", []),
            rank=rank,
        )

    @property
    def heading(self) -> str:
        """Return a readable section heading."""

        displayed_section_id = self.printed_section_id or self.section_id
        parts = [
            part
            for part in (displayed_section_id, self.title)
            if part
        ]
        return " ".join(parts).strip()

    @property
    def page_content(self) -> str:
        """Return title plus exact Section text for a future LangChain Document."""

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
