"""Modular candidate generators for knowledge-graph retrieval.

Candidate generation is deliberately separated from expansion and reranking.
This makes simple MENTIONS-only retrieval directly measurable and allows more
advanced graph strategies to be added as controlled ablations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_rag.kg.models import KGSectionResult, KGRankingMode


KGCandidateSource = Literal["mentions", "title", "descendant"]


class KGCandidate(BaseModel):
    """One Section candidate plus provenance from the retrieval pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section: KGSectionResult
    source: KGCandidateSource
    source_rank: int = Field(ge=1)
    final_rank: int | None = Field(default=None, ge=1)

    direct: bool = True
    seed_uid: str | None = None
    seed_rank: int | None = Field(default=None, ge=1)
    graph_distance: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("seed_uid")
    @classmethod
    def normalize_seed_uid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @property
    def section_uid(self) -> str:
        return self.section.section_uid

    @property
    def document_id(self) -> str:
        return self.section.document_id

    @property
    def printed_section_id(self) -> str | None:
        return self.section.printed_section_id

    @property
    def title(self) -> str | None:
        return self.section.title


class KGSectionSearchProtocol(Protocol):
    """Subset of ``KGSectionTools`` needed by candidate generators."""

    def search_sections_by_concepts(
        self,
        concepts: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
        top_k: int = 10,
        require_all: bool = False,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> list[KGSectionResult]: ...

    def search_sections_by_title(
        self,
        title_terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
        top_k: int = 10,
        require_all: bool = False,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> list[KGSectionResult]: ...


class CandidateGeneratorProtocol(Protocol):
    """Common interface for Section candidate generators."""

    name: str

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]: ...


class MentionsCandidateGenerator:
    """Generate Section candidates through ``Section-[:MENTIONS]->Concept``.

    With ``ranking_mode='concept_match'`` this is the pure graph baseline:
    candidates and their order depend only on MENTIONS concept matches.
    ``weighted_match`` keeps the same candidate set but applies the existing
    lexical match weights and title bonus implemented by ``KGSectionTools``.
    """

    name = "mentions"

    def __init__(
        self,
        tools: KGSectionSearchProtocol,
        *,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> None:
        self.tools = tools
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]:
        normalized_terms = _normalize_terms(terms)
        validated_top_k = _validate_top_k(top_k)

        results = self.tools.search_sections_by_concepts(
            normalized_terms,
            document_ids=document_ids,
            top_k=validated_top_k,
            require_all=bool(require_all),
            ranking_mode=self.ranking_mode,
            exclude_summary_sections=self.exclude_summary_sections,
        )
        return _wrap_results(results, source="mentions")


class TitleCandidateGenerator:
    """Generate Section candidates by matching section titles.

    This generator is retained for the advanced role-aware pipeline. It is not
    required by the MENTIONS-only baseline.
    """

    name = "title"

    def __init__(
        self,
        tools: KGSectionSearchProtocol,
        *,
        ranking_mode: KGRankingMode = "weighted_match",
        exclude_summary_sections: bool = True,
    ) -> None:
        self.tools = tools
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]:
        normalized_terms = _normalize_terms(terms)
        validated_top_k = _validate_top_k(top_k)

        results = self.tools.search_sections_by_title(
            normalized_terms,
            document_ids=document_ids,
            top_k=validated_top_k,
            require_all=bool(require_all),
            ranking_mode=self.ranking_mode,
            exclude_summary_sections=self.exclude_summary_sections,
        )
        return _wrap_results(results, source="title")


def deduplicate_candidates(
    candidates: Sequence[KGCandidate],
) -> list[KGCandidate]:
    """Keep the first occurrence of each canonical Section node."""

    output: list[KGCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.section_uid in seen:
            continue
        seen.add(candidate.section_uid)
        output.append(candidate)
    return output


def _wrap_results(
    results: Sequence[KGSectionResult],
    *,
    source: Literal["mentions", "title"],
) -> list[KGCandidate]:
    candidates: list[KGCandidate] = []
    seen: set[str] = set()

    for fallback_rank, result in enumerate(results, start=1):
        if result.section_uid in seen:
            continue
        seen.add(result.section_uid)
        source_rank = result.rank or fallback_rank
        candidates.append(
            KGCandidate(
                section=result,
                source=source,
                source_rank=source_rank,
                direct=True,
                seed_uid=result.section_uid,
                seed_rank=source_rank,
                graph_distance=0,
            )
        )
    return candidates


def _normalize_terms(values: Sequence[str] | str) -> list[str]:
    raw_values = [values] if isinstance(values, str) else list(values)
    output: list[str] = []
    seen: set[str] = set()

    for value in raw_values:
        term = str(value).strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(term)

    if not output:
        raise ValueError("At least one non-empty retrieval term is required")
    return output


def _validate_top_k(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc
    if normalized < 1 or normalized > 100:
        raise ValueError("top_k must be between 1 and 100")
    return normalized


def _validate_ranking_mode(value: str) -> KGRankingMode:
    normalized = str(value).strip().lower()
    if normalized not in {"concept_match", "weighted_match"}:
        raise ValueError(
            "ranking_mode must be 'concept_match' or 'weighted_match'"
        )
    return normalized  # type: ignore[return-value]
