"""Orchestration layer for parameterized knowledge-graph retrieval.

This module connects the structured LLM router to the deterministic Neo4j
Section tools. It intentionally does not calculate evaluation metrics and does
not accept gold document or section identifiers.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentic_rag.agent.output import KGRetrievalPlan, KGToolCall
from agentic_rag.kg.models import KGSectionResult, KGRankingMode


KGRetrievalStatus = Literal[
    "success",
    "partial_success",
    "no_results",
    "router_error",
    "execution_error",
]

KGToolExecutionStatus = Literal[
    "success",
    "no_results",
    "execution_error",
]


class KGRouterProtocol(Protocol):
    """Minimal router interface required by ``KGParameterizedRetriever``."""

    def route(
        self,
        question: str,
        *,
        config: Any | None = None,
    ) -> KGRetrievalPlan:
        ...


class KGSectionToolsProtocol(Protocol):
    """Minimal deterministic-tool interface required by the retriever."""

    def search_sections_by_concepts(
        self,
        concepts: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
        top_k: int = 10,
        require_all: bool = False,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> list[KGSectionResult]:
        ...

    def search_sections_by_title(
        self,
        title_terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
        top_k: int = 10,
        require_all: bool = False,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> list[KGSectionResult]:
        ...


class KGRRFContribution(BaseModel):
    """One tool-ranking contribution to a fused Section result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_index: int = Field(ge=0)
    tool: Literal[
        "search_sections_by_concepts",
        "search_sections_by_title",
    ]
    source_rank: int = Field(ge=1)
    reciprocal_rank_score: float = Field(gt=0)


class KGFusedSectionResult(KGSectionResult):
    """Section result enriched with deterministic RRF diagnostics."""

    fusion_method: Literal["rrf"] = "rrf"
    fusion_score: float = Field(gt=0)
    best_source_rank: int = Field(ge=1)
    contributions: list[KGRRFContribution] = Field(min_length=1)


class KGToolExecution(BaseModel):
    """Execution trace for one tool call produced by the router."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_index: int = Field(ge=0)
    call: KGToolCall
    status: KGToolExecutionStatus
    requested_k: int = Field(ge=1)
    ranking_mode: KGRankingMode
    latency_ms: float = Field(ge=0)
    results: list[KGSectionResult] = Field(default_factory=list)
    error: str | None = None

    @field_validator("error")
    @classmethod
    def normalize_error(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "KGToolExecution":
        if self.status == "success":
            if not self.results:
                raise ValueError(
                    "A successful tool execution must contain results"
                )
            if self.error:
                raise ValueError(
                    "A successful tool execution cannot contain an error"
                )

        elif self.status == "no_results":
            if self.results:
                raise ValueError(
                    "A no_results tool execution cannot contain results"
                )
            if self.error:
                raise ValueError(
                    "A no_results tool execution cannot contain an error"
                )

        elif self.status == "execution_error":
            if self.results:
                raise ValueError(
                    "A failed tool execution cannot contain results"
                )
            if not self.error:
                raise ValueError(
                    "A failed tool execution requires an error message"
                )

        return self

    @property
    def returned_count(self) -> int:
        return len(self.results)


class KGRetrievalRun(BaseModel):
    """Complete trace for one routed KG retrieval request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    status: KGRetrievalStatus
    plan: KGRetrievalPlan | None = None
    tool_executions: list[KGToolExecution] = Field(default_factory=list)
    results: list[KGFusedSectionResult] = Field(default_factory=list)

    candidate_k: int = Field(ge=1)
    final_k: int = Field(ge=1)
    ranking_mode: KGRankingMode
    rrf_k: int = Field(ge=1)
    exclude_summary_sections: bool = True

    latency_ms: float = Field(ge=0)
    error: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("question must be a non-empty string")
        return normalized

    @field_validator("error")
    @classmethod
    def normalize_error(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "KGRetrievalRun":
        if self.status == "router_error":
            if self.plan is not None:
                raise ValueError("router_error cannot contain a plan")
            if self.tool_executions or self.results:
                raise ValueError(
                    "router_error cannot contain tool executions or results"
                )
            if not self.error:
                raise ValueError("router_error requires an error message")
            return self

        if self.plan is None:
            raise ValueError(
                f"Status {self.status!r} requires a retrieval plan"
            )

        failed_calls = [
            execution
            for execution in self.tool_executions
            if execution.status == "execution_error"
        ]

        if self.status == "success":
            if not self.results:
                raise ValueError("success requires at least one result")
            if failed_calls or self.error:
                raise ValueError(
                    "success cannot contain failed calls or an error"
                )

        elif self.status == "partial_success":
            if not failed_calls:
                raise ValueError(
                    "partial_success requires at least one failed call"
                )
            if not self.error:
                raise ValueError(
                    "partial_success requires an error summary"
                )

        elif self.status == "no_results":
            if self.results:
                raise ValueError("no_results cannot contain results")
            if failed_calls or self.error:
                raise ValueError(
                    "no_results cannot contain failed calls or an error"
                )

        elif self.status == "execution_error":
            if self.results:
                raise ValueError("execution_error cannot contain results")
            if not failed_calls:
                raise ValueError(
                    "execution_error requires at least one failed call"
                )
            if not self.error:
                raise ValueError(
                    "execution_error requires an error summary"
                )

        return self

    @property
    def returned_count(self) -> int:
        return len(self.results)


class KGParameterizedRetriever:
    """Route one question, execute controlled tools, and fuse their rankings."""

    def __init__(
        self,
        router: KGRouterProtocol,
        tools: KGSectionToolsProtocol,
        *,
        candidate_k: int = 15,
        final_k: int = 10,
        ranking_mode: KGRankingMode = "weighted_match",
        rrf_k: int = 60,
        exclude_summary_sections: bool = True,
    ) -> None:
        self.router = router
        self.tools = tools
        self.candidate_k = _validate_positive_int(
            candidate_k,
            field_name="candidate_k",
            maximum=100,
        )
        self.final_k = _validate_positive_int(
            final_k,
            field_name="final_k",
            maximum=100,
        )
        self.rrf_k = _validate_positive_int(
            rrf_k,
            field_name="rrf_k",
        )
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)

    def retrieve(
        self,
        question: str,
        *,
        router_config: Any | None = None,
    ) -> KGRetrievalRun:
        """Execute the complete retrieval path for one question.

        Gold document and section identifiers are deliberately unavailable to
        this method. Every tool call searches the full graph using
        ``document_ids=None``.
        """

        normalized_question = _validate_question(question)
        started = time.perf_counter()

        try:
            plan = self.router.route(
                normalized_question,
                config=router_config,
            )
        except Exception as exc:
            return KGRetrievalRun(
                question=normalized_question,
                status="router_error",
                plan=None,
                tool_executions=[],
                results=[],
                candidate_k=self.candidate_k,
                final_k=self.final_k,
                ranking_mode=self.ranking_mode,
                rrf_k=self.rrf_k,
                exclude_summary_sections=self.exclude_summary_sections,
                latency_ms=_elapsed_ms(started),
                error=_format_exception(exc),
            )

        executions = [
            self._execute_call(call, call_index=index)
            for index, call in enumerate(plan.calls)
        ]

        fused_results = reciprocal_rank_fusion(
            executions,
            rrf_k=self.rrf_k,
            top_k=self.final_k,
        )

        failed_executions = [
            execution
            for execution in executions
            if execution.status == "execution_error"
        ]

        if failed_executions:
            error_summary = "; ".join(
                f"call {execution.call_index} "
                f"({execution.call.tool}): {execution.error}"
                for execution in failed_executions
            )

            status: KGRetrievalStatus = (
                "partial_success"
                if len(failed_executions) < len(executions)
                else "execution_error"
            )
        elif fused_results:
            status = "success"
            error_summary = None
        else:
            status = "no_results"
            error_summary = None

        return KGRetrievalRun(
            question=normalized_question,
            status=status,
            plan=plan,
            tool_executions=executions,
            results=fused_results,
            candidate_k=self.candidate_k,
            final_k=self.final_k,
            ranking_mode=self.ranking_mode,
            rrf_k=self.rrf_k,
            exclude_summary_sections=self.exclude_summary_sections,
            latency_ms=_elapsed_ms(started),
            error=error_summary,
        )

    def retrieve_dict(
        self,
        question: str,
        *,
        router_config: Any | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-serializable retrieval trace."""

        return self.retrieve(
            question,
            router_config=router_config,
        ).model_dump(mode="json")

    def _execute_call(
        self,
        call: KGToolCall,
        *,
        call_index: int,
    ) -> KGToolExecution:
        started = time.perf_counter()

        try:
            if call.tool == "search_sections_by_concepts":
                results = self.tools.search_sections_by_concepts(
                    call.terms,
                    document_ids=None,
                    top_k=self.candidate_k,
                    require_all=call.require_all,
                    ranking_mode=self.ranking_mode,
                    exclude_summary_sections=self.exclude_summary_sections,
                )

            elif call.tool == "search_sections_by_title":
                results = self.tools.search_sections_by_title(
                    call.terms,
                    document_ids=None,
                    top_k=self.candidate_k,
                    require_all=call.require_all,
                    ranking_mode=self.ranking_mode,
                    exclude_summary_sections=self.exclude_summary_sections,
                )

            else:
                raise ValueError(f"Unsupported KG tool: {call.tool!r}")

            status: KGToolExecutionStatus = (
                "success" if results else "no_results"
            )
            error = None

        except Exception as exc:
            results = []
            status = "execution_error"
            error = _format_exception(exc)

        return KGToolExecution(
            call_index=call_index,
            call=call,
            status=status,
            requested_k=self.candidate_k,
            ranking_mode=self.ranking_mode,
            latency_ms=_elapsed_ms(started),
            results=results,
            error=error,
        )


def reciprocal_rank_fusion(
    executions: Sequence[KGToolExecution],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[KGFusedSectionResult]:
    """Fuse successful tool rankings using Reciprocal Rank Fusion.

    Results are deduplicated by the graph-level ``section_uid``. The original
    per-tool results remain available inside ``KGToolExecution``; the fused
    result adds a separate ``fusion_score`` and contribution trace.
    """

    validated_rrf_k = _validate_positive_int(
        rrf_k,
        field_name="rrf_k",
    )
    validated_top_k = (
        None
        if top_k is None
        else _validate_positive_int(
            top_k,
            field_name="top_k",
            maximum=100,
        )
    )

    fusion_scores: dict[str, float] = {}
    contributions: dict[str, list[KGRRFContribution]] = {}
    representative_results: dict[str, KGSectionResult] = {}
    representative_keys: dict[str, tuple[int, int]] = {}

    for execution in executions:
        if execution.status != "success":
            continue

        seen_in_call: set[str] = set()

        for fallback_rank, result in enumerate(
            execution.results,
            start=1,
        ):
            section_uid = result.section_uid
            if section_uid in seen_in_call:
                continue
            seen_in_call.add(section_uid)

            source_rank = result.rank or fallback_rank
            contribution_score = 1.0 / (
                validated_rrf_k + source_rank
            )

            contribution = KGRRFContribution(
                call_index=execution.call_index,
                tool=execution.call.tool,
                source_rank=source_rank,
                reciprocal_rank_score=contribution_score,
            )

            fusion_scores[section_uid] = (
                fusion_scores.get(section_uid, 0.0)
                + contribution_score
            )
            contributions.setdefault(section_uid, []).append(
                contribution
            )

            representative_key = (
                source_rank,
                execution.call_index,
            )
            current_key = representative_keys.get(section_uid)

            if current_key is None or representative_key < current_key:
                representative_keys[section_uid] = representative_key
                representative_results[section_uid] = result

    ordered_uids = sorted(
        fusion_scores,
        key=lambda section_uid: (
            -fusion_scores[section_uid],
            representative_keys[section_uid][0],
            representative_keys[section_uid][1],
            section_uid,
        ),
    )

    if validated_top_k is not None:
        ordered_uids = ordered_uids[:validated_top_k]

    fused_results: list[KGFusedSectionResult] = []

    for fused_rank, section_uid in enumerate(ordered_uids, start=1):
        representative = representative_results[section_uid]
        section_contributions = sorted(
            contributions[section_uid],
            key=lambda contribution: (
                contribution.call_index,
                contribution.source_rank,
            ),
        )

        payload = representative.model_dump(mode="python")
        payload.update(
            {
                "rank": fused_rank,
                "fusion_method": "rrf",
                "fusion_score": fusion_scores[section_uid],
                "best_source_rank": min(
                    contribution.source_rank
                    for contribution in section_contributions
                ),
                "contributions": section_contributions,
            }
        )

        fused_results.append(
            KGFusedSectionResult.model_validate(payload)
        )

    return fused_results


def _validate_question(question: str) -> str:
    normalized = str(question).strip()
    if not normalized:
        raise ValueError("question must be a non-empty string")
    return normalized


def _validate_positive_int(
    value: int,
    *,
    field_name: str,
    maximum: int | None = None,
) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc

    if normalized < 1:
        raise ValueError(f"{field_name} must be at least 1")

    if maximum is not None and normalized > maximum:
        raise ValueError(
            f"{field_name} must not exceed {maximum}"
        )

    return normalized


def _validate_ranking_mode(value: str) -> KGRankingMode:
    normalized = str(value).strip().lower()
    if normalized not in {"concept_match", "weighted_match"}:
        raise ValueError(
            "ranking_mode must be 'concept_match' or 'weighted_match'"
        )
    return normalized  # type: ignore[return-value]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
