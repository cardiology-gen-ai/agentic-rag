"""Orchestration layer for parameterized knowledge-graph retrieval.

The module connects the structured LLM router to deterministic Neo4j Section
retrieval tools. It supports direct retrieval, hierarchy-aware composition,
facet-preserving composition, and RRF for genuinely alternative rankings.
It does not calculate evaluation metrics and never accepts gold identifiers.
"""

from __future__ import annotations

import re
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

KGCombinationMethod = Literal[
    "direct",
    "hierarchical_context",
    "facet_preserving",
    "context_aware_facet",
    "same_section_anchor_fallback",
    "same_section_anchor_rescue",
    "rrf",
]


class KGRouterProtocol(Protocol):
    def route(
        self,
        question: str,
        *,
        config: Any | None = None,
    ) -> KGRetrievalPlan:
        ...


class KGSectionToolsProtocol(Protocol):
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

    def find_hierarchical_context_matches(
        self,
        anchor_uids: Sequence[str] | str,
        context_uids: Sequence[str] | str,
        *,
        max_depth: int = 6,
    ) -> list[dict[str, Any]]:
        ...


class KGResultContribution(BaseModel):
    """One source-ranking contribution to a combined Section result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_index: int = Field(ge=0)
    tool: Literal[
        "search_sections_by_concepts",
        "search_sections_by_title",
    ]
    role: Literal[
        "anchor",
        "context",
        "facet",
        "alternative",
    ]
    source_rank: int = Field(ge=1)
    reciprocal_rank_score: float | None = Field(default=None, gt=0)


class KGHierarchyContextMatch(BaseModel):
    """Structural support connecting a context Section to an anchor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_uid: str
    context_document_id: str | None = None
    context_section_id: str | None = None
    context_printed_section_id: str | None = None
    context_title: str | None = None
    context_call_index: int = Field(ge=0)
    context_rank: int = Field(ge=1)
    hierarchy_distance: int = Field(ge=0)


class KGCombinedSectionResult(KGSectionResult):
    """Section result enriched with deterministic combination diagnostics."""

    combination_method: KGCombinationMethod
    combination_score: float | None = None
    best_source_rank: int = Field(ge=1)
    contributions: list[KGResultContribution] = Field(min_length=1)
    context_supported: bool = False
    context_matches: list[KGHierarchyContextMatch] = Field(default_factory=list)
    covered_facets: list[str] = Field(default_factory=list)


# Backward-compatible import name used by the first RRF prototype.
KGFusedSectionResult = KGCombinedSectionResult


_ANCHOR_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
    "approach",
    "guideline",
    "guidelines",
    "patient",
    "patients",
    "recommended",
    "recommendation",
    "recommendations",
    "section",
    "sections",
}

_EXCLUDED_FALLBACK_TITLE_PATTERNS = (
    "what to do",
    "what not to do",
    "key messages",
    "gaps in evidence",
    "references",
    "bibliography",
)

_RESCUE_MIN_SCORE = 25.0


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
    results: list[KGCombinedSectionResult] = Field(default_factory=list)

    candidate_k: int = Field(ge=1)
    final_k: int = Field(ge=1)
    ranking_mode: KGRankingMode
    rrf_k: int = Field(ge=1)
    hierarchy_max_depth: int = Field(ge=0)
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
            if not self.results:
                raise ValueError(
                    "partial_success requires at least one result"
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
            if not self.error:
                raise ValueError(
                    "execution_error requires an error summary"
                )

        return self

    @property
    def returned_count(self) -> int:
        return len(self.results)


class KGParameterizedRetriever:
    """Route one question, execute controlled tools, and combine results."""

    def __init__(
        self,
        router: KGRouterProtocol,
        tools: KGSectionToolsProtocol,
        *,
        candidate_k: int = 15,
        final_k: int = 10,
        ranking_mode: KGRankingMode = "weighted_match",
        rrf_k: int = 60,
        hierarchy_max_depth: int = 6,
        exclude_summary_sections: bool = True,
        multiple_facets_context_aware_merge: bool = False,
        same_section_anchor_fallback: bool = False,
        same_section_anchor_rescue: bool = False,
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
        self.hierarchy_max_depth = _validate_non_negative_int(
            hierarchy_max_depth,
            field_name="hierarchy_max_depth",
            maximum=8,
        )
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)
        self.multiple_facets_context_aware_merge = bool(
            multiple_facets_context_aware_merge
        )
        self.same_section_anchor_fallback = bool(
            same_section_anchor_fallback
        )
        self.same_section_anchor_rescue = bool(same_section_anchor_rescue)

    def retrieve(
        self,
        question: str,
        *,
        router_config: Any | None = None,
    ) -> KGRetrievalRun:
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
                hierarchy_max_depth=self.hierarchy_max_depth,
                exclude_summary_sections=self.exclude_summary_sections,
                latency_ms=_elapsed_ms(started),
                error=_format_exception(exc),
            )

        executions = [
            self._execute_call(call, call_index=index)
            for index, call in enumerate(plan.calls)
        ]

        combination_error: str | None = None
        try:
            combined_results = self._combine(plan, executions)
        except Exception as exc:
            combination_error = (
                "Combination failed; used RRF fallback: "
                + _format_exception(exc)
            )
            combined_results = reciprocal_rank_fusion(
                executions,
                rrf_k=self.rrf_k,
                top_k=self.final_k,
            )

        failed_executions = [
            execution
            for execution in executions
            if execution.status == "execution_error"
        ]

        errors: list[str] = []
        if failed_executions:
            errors.extend(
                f"call {execution.call_index} "
                f"({execution.call.tool}): {execution.error}"
                for execution in failed_executions
            )
        if combination_error:
            errors.append(combination_error)

        error_summary = "; ".join(errors) or None

        if combined_results and error_summary:
            status: KGRetrievalStatus = "partial_success"
        elif combined_results:
            status = "success"
        elif failed_executions:
            status = "execution_error"
        else:
            status = "no_results"

        return KGRetrievalRun(
            question=normalized_question,
            status=status,
            plan=plan,
            tool_executions=executions,
            results=combined_results,
            candidate_k=self.candidate_k,
            final_k=self.final_k,
            ranking_mode=self.ranking_mode,
            rrf_k=self.rrf_k,
            hierarchy_max_depth=self.hierarchy_max_depth,
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
        return self.retrieve(
            question,
            router_config=router_config,
        ).model_dump(mode="json")

    def _combine(
        self,
        plan: KGRetrievalPlan,
        executions: Sequence[KGToolExecution],
    ) -> list[KGCombinedSectionResult]:
        if plan.combination_mode == "direct":
            return direct_results(executions, top_k=self.final_k)

        if plan.combination_mode == "same_section":
            results = hierarchical_context_rerank(
                executions,
                tools=self.tools,
                top_k=self.final_k,
                max_depth=self.hierarchy_max_depth,
            )

            if self.same_section_anchor_rescue:
                return same_section_anchor_rescue_merge(
                    executions,
                    base_results=results,
                    top_k=self.final_k,
                )

            if results:
                return results

            if self.same_section_anchor_fallback:
                return same_section_anchor_fallback_merge(
                    executions,
                    top_k=self.final_k,
                )

            return []

        if plan.combination_mode == "multiple_facets":
            if self.multiple_facets_context_aware_merge:
                return context_aware_facet_merge(
                    executions,
                    rrf_k=self.rrf_k,
                    top_k=self.final_k,
                )

            return facet_preserving_merge(
                executions,
                top_k=self.final_k,
            )

        if plan.combination_mode == "alternative_retrieval":
            return reciprocal_rank_fusion(
                executions,
                rrf_k=self.rrf_k,
                top_k=self.final_k,
            )

        raise ValueError(
            f"Unsupported combination mode: {plan.combination_mode!r}"
        )

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


def direct_results(
    executions: Sequence[KGToolExecution],
    *,
    top_k: int | None = None,
) -> list[KGCombinedSectionResult]:
    execution = next(
        (
            item
            for item in executions
            if item.status == "success" and item.call.role == "anchor"
        ),
        None,
    )
    if execution is None:
        return []

    results: list[KGCombinedSectionResult] = []
    seen: set[str] = set()

    for fallback_rank, result in enumerate(execution.results, start=1):
        if result.section_uid in seen:
            continue
        seen.add(result.section_uid)
        source_rank = result.rank or fallback_rank
        results.append(
            _combined_result(
                result,
                rank=len(results) + 1,
                method="direct",
                best_source_rank=source_rank,
                contributions=[
                    _source_contribution(execution, source_rank)
                ],
            )
        )

        if top_k is not None and len(results) >= top_k:
            break

    return results


def hierarchical_context_rerank(
    executions: Sequence[KGToolExecution],
    *,
    tools: KGSectionToolsProtocol,
    top_k: int | None = None,
    max_depth: int = 6,
) -> list[KGCombinedSectionResult]:
    anchor_execution = _execution_by_role(executions, "anchor")
    context_execution = _execution_by_role(executions, "context")

    if anchor_execution is None or anchor_execution.status != "success":
        return []

    anchors = _deduplicate_results(anchor_execution.results)
    context_results = (
        _deduplicate_results(context_execution.results)
        if context_execution is not None
        and context_execution.status == "success"
        else []
    )

    context_by_uid = {
        item.section_uid: (item.rank or index)
        for index, item in enumerate(context_results, start=1)
    }

    support_by_anchor: dict[str, list[KGHierarchyContextMatch]] = {}

    if context_results:
        rows = tools.find_hierarchical_context_matches(
            [item.section_uid for item in anchors],
            [item.section_uid for item in context_results],
            max_depth=max_depth,
        )

        for row in rows:
            anchor_uid = str(row.get("anchor_uid") or "").strip()
            context_uid = str(row.get("context_uid") or "").strip()
            if not anchor_uid or context_uid not in context_by_uid:
                continue

            match = KGHierarchyContextMatch(
                context_uid=context_uid,
                context_document_id=row.get("context_document_id"),
                context_section_id=row.get("context_section_id"),
                context_printed_section_id=row.get(
                    "context_printed_section_id"
                ),
                context_title=row.get("context_title"),
                context_call_index=context_execution.call_index,
                context_rank=context_by_uid[context_uid],
                hierarchy_distance=int(row.get("hierarchy_distance", 0)),
            )
            support_by_anchor.setdefault(anchor_uid, []).append(match)

    sortable: list[
        tuple[tuple[int, int, int, int, str], KGCombinedSectionResult]
    ] = []

    for fallback_rank, anchor in enumerate(anchors, start=1):
        anchor_rank = anchor.rank or fallback_rank
        matches = sorted(
            support_by_anchor.get(anchor.section_uid, []),
            key=lambda item: (
                item.context_rank,
                item.hierarchy_distance,
                item.context_uid,
            ),
        )
        supported = bool(matches)
        best_context_rank = matches[0].context_rank if matches else 10**9
        best_distance = matches[0].hierarchy_distance if matches else 10**9

        contributions = [
            _source_contribution(anchor_execution, anchor_rank)
        ]
        if context_execution is not None:
            contributions.extend(
                _source_contribution(
                    context_execution,
                    match.context_rank,
                )
                for match in matches
            )

        combined = _combined_result(
            anchor,
            rank=1,
            method="hierarchical_context",
            best_source_rank=min(
                [anchor_rank]
                + [match.context_rank for match in matches]
            ),
            contributions=contributions,
            context_supported=supported,
            context_matches=matches,
        )

        sort_key = (
            0 if supported else 1,
            best_context_rank,
            best_distance,
            anchor_rank,
            anchor.section_uid,
        )
        sortable.append((sort_key, combined))

    sortable.sort(key=lambda item: item[0])
    combined_results = [item[1] for item in sortable]

    if top_k is not None:
        combined_results = combined_results[:top_k]

    return [
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(combined_results, start=1)
    ]


def same_section_anchor_fallback_merge(
    executions: Sequence[KGToolExecution],
    *,
    top_k: int | None = None,
) -> list[KGCombinedSectionResult]:
    """Fallback for same_section when anchor/context hierarchy matching fails.

    This fallback uses successful context results, but only keeps candidates
    that lexically match the anchor terms in the title or text. It is intended
    for cases where the anchor title search is too specific, while the context
    call already retrieved plausible target sections.
    """
    candidates = _same_section_anchor_fallback_candidates(executions)
    if top_k is not None:
        candidates = candidates[:top_k]

    return [
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(candidates, start=1)
    ]


def same_section_anchor_rescue_merge(
    executions: Sequence[KGToolExecution],
    *,
    base_results: Sequence[KGCombinedSectionResult],
    top_k: int | None = None,
) -> list[KGCombinedSectionResult]:
    """Merge same_section results with strong anchor-sensitive context hits."""
    rescue_candidates = [
        candidate
        for candidate in _same_section_anchor_fallback_candidates(executions)
        if candidate.combination_score is not None
        and candidate.combination_score >= _RESCUE_MIN_SCORE
    ]

    if not rescue_candidates:
        output = list(base_results)
        if top_k is not None:
            output = output[:top_k]
        return [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(output, start=1)
        ]

    if not base_results:
        output = rescue_candidates
        if top_k is not None:
            output = output[:top_k]
        return [
            item.model_copy(
                update={
                    "rank": rank,
                    "combination_method": "same_section_anchor_rescue",
                }
            )
            for rank, item in enumerate(output, start=1)
        ]

    sortable: list[
        tuple[float, int, int, str, KGCombinedSectionResult]
    ] = []

    for fallback_rank, result in enumerate(base_results, start=1):
        source_rank = result.best_source_rank or result.rank or fallback_rank
        result_rank = result.rank or fallback_rank
        score = 50.0 - result_rank * 0.01
        sortable.append(
            (
                -score,
                1,
                source_rank,
                result.section_uid,
                result.model_copy(
                    update={
                        "combination_method": "same_section_anchor_rescue",
                        "combination_score": score,
                    }
                ),
            )
        )

    for fallback_rank, result in enumerate(rescue_candidates, start=1):
        score = result.combination_score or 0.0
        source_rank = result.best_source_rank or result.rank or fallback_rank
        sortable.append(
            (
                -score,
                0,
                source_rank,
                result.section_uid,
                result.model_copy(
                    update={
                        "combination_method": "same_section_anchor_rescue",
                    }
                ),
            )
        )

    sortable.sort(key=lambda item: item[:4])

    output: list[KGCombinedSectionResult] = []
    seen: set[str] = set()
    for _score, _priority, _source_rank, section_uid, result in sortable:
        if section_uid in seen:
            continue
        seen.add(section_uid)
        output.append(result)
        if top_k is not None and len(output) >= top_k:
            break

    return [
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(output, start=1)
    ]


def _same_section_anchor_fallback_candidates(
    executions: Sequence[KGToolExecution],
) -> list[KGCombinedSectionResult]:
    anchor_terms: list[str] = []
    for execution in executions:
        if execution.call.role == "anchor":
            anchor_terms.extend(execution.call.terms)

    anchor_tokens: set[str] = set()
    for term in anchor_terms:
        anchor_tokens.update(_anchor_fallback_tokens(term))

    anchor_phrases = _anchor_fallback_phrases(anchor_terms)

    if not anchor_tokens and not anchor_phrases:
        return []

    context_executions = [
        execution
        for execution in executions
        if execution.status == "success"
        and execution.call.role == "context"
    ]

    candidates: list[
        tuple[float, int, str, KGCombinedSectionResult]
    ] = []

    for execution in context_executions:
        for fallback_rank, result in enumerate(execution.results, start=1):
            if _is_excluded_fallback_title(result.title):
                continue

            source_rank = result.rank or fallback_rank
            title_norm = _normalize_anchor_fallback_text(result.title)
            text_norm = _normalize_anchor_fallback_text(result.text)
            title_tokens = _anchor_fallback_tokens(result.title)
            text_tokens = _anchor_fallback_tokens(result.text)

            title_overlap = anchor_tokens & title_tokens
            text_overlap = anchor_tokens & text_tokens

            has_title_phrase = any(
                phrase and phrase in title_norm
                for phrase in anchor_phrases
            )
            has_text_phrase = any(
                phrase and phrase in text_norm
                for phrase in anchor_phrases
            )

            accepted = (
                has_title_phrase
                or len(title_overlap) >= 2
                or (len(title_overlap) >= 1 and has_text_phrase)
                or len(text_overlap) >= 3
            )

            if not accepted:
                continue

            score = 0.0
            if has_title_phrase:
                score += 100.0
            score += 25.0 * len(title_overlap)

            if has_text_phrase:
                score += 20.0
            score += min(len(text_overlap), 8) * 2.0
            score -= source_rank * 0.01

            combined = _combined_result(
                result,
                rank=1,
                method="same_section_anchor_fallback",
                combination_score=score,
                best_source_rank=source_rank,
                contributions=[
                    _source_contribution(execution, source_rank)
                ],
                context_supported=True,
                covered_facets=[],
            )
            candidates.append(
                (-score, source_rank, result.section_uid, combined)
            )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    output: list[KGCombinedSectionResult] = []
    seen: set[str] = set()
    for _negative_score, _source_rank, section_uid, combined in candidates:
        if section_uid in seen:
            continue
        seen.add(section_uid)
        output.append(combined)

    return output


def facet_preserving_merge(
    executions: Sequence[KGToolExecution],
    *,
    top_k: int | None = None,
) -> list[KGCombinedSectionResult]:
    facet_executions = [
        item
        for item in executions
        if item.status == "success" and item.call.role == "facet"
    ]
    if not facet_executions:
        return []

    facet_terms: list[str] = []
    seen_terms: set[str] = set()
    for execution in facet_executions:
        for term in execution.call.terms:
            key = term.casefold()
            if key not in seen_terms:
                seen_terms.add(key)
                facet_terms.append(term)

    candidates: list[tuple[KGToolExecution, KGSectionResult, int]] = []
    seen_candidate_keys: set[tuple[int, str]] = set()

    for execution in facet_executions:
        for fallback_rank, result in enumerate(execution.results, start=1):
            key = (execution.call_index, result.section_uid)
            if key in seen_candidate_keys:
                continue
            seen_candidate_keys.add(key)
            candidates.append(
                (execution, result, result.rank or fallback_rank)
            )

    selected_uids: list[str] = []
    coverage: dict[str, list[str]] = {}
    representative: dict[
        str, tuple[KGToolExecution, KGSectionResult, int]
    ] = {}

    for term in facet_terms:
        normalized_term = term.casefold()
        for execution, result, source_rank in candidates:
            matched_terms = {
                item.casefold() for item in result.matched_terms
            }
            if normalized_term not in matched_terms:
                continue

            coverage.setdefault(result.section_uid, []).append(term)
            representative.setdefault(
                result.section_uid,
                (execution, result, source_rank),
            )
            if result.section_uid not in selected_uids:
                selected_uids.append(result.section_uid)
            break

    for execution, result, source_rank in candidates:
        representative.setdefault(
            result.section_uid,
            (execution, result, source_rank),
        )
        if result.section_uid not in selected_uids:
            selected_uids.append(result.section_uid)

    if top_k is not None:
        selected_uids = selected_uids[:top_k]

    results: list[KGCombinedSectionResult] = []
    for rank, uid in enumerate(selected_uids, start=1):
        execution, result, source_rank = representative[uid]
        results.append(
            _combined_result(
                result,
                rank=rank,
                method="facet_preserving",
                best_source_rank=source_rank,
                contributions=[
                    _source_contribution(execution, source_rank)
                ],
                covered_facets=coverage.get(uid, []),
            )
        )

    return results


def context_aware_facet_merge(
    executions: Sequence[KGToolExecution],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[KGCombinedSectionResult]:
    """Rerank facet results using document-level support from context calls.

    This is an ablation-friendly variant of facet_preserving_merge. It keeps
    the facet-generated candidates, but promotes candidates whose document is
    supported by successful context calls. If facet retrieval returns no
    results, successful context calls are used as a conservative fallback.
    """

    facet_results = facet_preserving_merge(
        executions,
        top_k=None,
    )

    context_executions = [
        execution
        for execution in executions
        if execution.status == "success"
        and execution.call.role == "context"
    ]

    if not facet_results:
        if not context_executions:
            return []
        return reciprocal_rank_fusion(
            context_executions,
            rrf_k=rrf_k,
            top_k=top_k,
        )

    context_results: list[KGSectionResult] = []
    for execution in context_executions:
        context_results.extend(execution.results)

    if not context_results:
        output = facet_results[:top_k] if top_k is not None else facet_results
        return [
            item.model_copy(
                update={
                    "rank": rank,
                    "combination_method": "context_aware_facet",
                }
            )
            for rank, item in enumerate(output, start=1)
        ]

    context_rank_by_document: dict[str, int] = {}
    for fallback_rank, result in enumerate(context_results, start=1):
        document_id = result.document_id
        if not document_id:
            continue
        rank = result.rank or fallback_rank
        current = context_rank_by_document.get(document_id)
        if current is None or rank < current:
            context_rank_by_document[document_id] = rank

    sortable: list[
        tuple[tuple[int, int, int, str], KGCombinedSectionResult]
    ] = []

    for fallback_rank, result in enumerate(facet_results, start=1):
        document_id = result.document_id
        source_rank = result.best_source_rank or fallback_rank
        context_document_rank = context_rank_by_document.get(
            document_id,
            10**9,
        )
        supported_by_context_document = document_id in context_rank_by_document

        sort_key = (
            0 if supported_by_context_document else 1,
            context_document_rank,
            source_rank,
            result.section_uid,
        )

        sortable.append(
            (
                sort_key,
                result.model_copy(
                    update={
                        "combination_method": "context_aware_facet",
                        "context_supported": supported_by_context_document,
                    }
                ),
            )
        )

    sortable.sort(key=lambda item: item[0])
    output = [item[1] for item in sortable]

    if top_k is not None:
        output = output[:top_k]

    return [
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(output, start=1)
    ]


def reciprocal_rank_fusion(
    executions: Sequence[KGToolExecution],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[KGCombinedSectionResult]:
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
    contributions: dict[str, list[KGResultContribution]] = {}
    representative_results: dict[str, KGSectionResult] = {}
    representative_keys: dict[str, tuple[int, int]] = {}

    for execution in executions:
        if execution.status != "success":
            continue

        seen_in_call: set[str] = set()
        for fallback_rank, result in enumerate(execution.results, start=1):
            section_uid = result.section_uid
            if section_uid in seen_in_call:
                continue
            seen_in_call.add(section_uid)

            source_rank = result.rank or fallback_rank
            reciprocal_score = 1.0 / (validated_rrf_k + source_rank)
            contribution = _source_contribution(
                execution,
                source_rank,
                reciprocal_rank_score=reciprocal_score,
            )

            fusion_scores[section_uid] = (
                fusion_scores.get(section_uid, 0.0)
                + reciprocal_score
            )
            contributions.setdefault(section_uid, []).append(
                contribution
            )

            representative_key = (source_rank, execution.call_index)
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

    results: list[KGCombinedSectionResult] = []
    for rank, section_uid in enumerate(ordered_uids, start=1):
        section_contributions = sorted(
            contributions[section_uid],
            key=lambda item: (item.call_index, item.source_rank),
        )
        results.append(
            _combined_result(
                representative_results[section_uid],
                rank=rank,
                method="rrf",
                combination_score=fusion_scores[section_uid],
                best_source_rank=min(
                    item.source_rank for item in section_contributions
                ),
                contributions=section_contributions,
            )
        )

    return results


def _execution_by_role(
    executions: Sequence[KGToolExecution],
    role: str,
) -> KGToolExecution | None:
    return next(
        (item for item in executions if item.call.role == role),
        None,
    )


def _source_contribution(
    execution: KGToolExecution,
    source_rank: int,
    *,
    reciprocal_rank_score: float | None = None,
) -> KGResultContribution:
    return KGResultContribution(
        call_index=execution.call_index,
        tool=execution.call.tool,
        role=execution.call.role,
        source_rank=source_rank,
        reciprocal_rank_score=reciprocal_rank_score,
    )


def _combined_result(
    result: KGSectionResult,
    *,
    rank: int,
    method: KGCombinationMethod,
    best_source_rank: int,
    contributions: Sequence[KGResultContribution],
    combination_score: float | None = None,
    context_supported: bool = False,
    context_matches: Sequence[KGHierarchyContextMatch] = (),
    covered_facets: Sequence[str] = (),
) -> KGCombinedSectionResult:
    payload = result.model_dump(mode="python")
    payload.update(
        {
            "rank": rank,
            "combination_method": method,
            "combination_score": combination_score,
            "best_source_rank": best_source_rank,
            "contributions": list(contributions),
            "context_supported": context_supported,
            "context_matches": list(context_matches),
            "covered_facets": list(covered_facets),
        }
    )
    return KGCombinedSectionResult.model_validate(payload)


def _normalize_anchor_fallback_text(value: str | None) -> str:
    text = str(value or "").casefold()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _anchor_fallback_tokens(value: str | None) -> set[str]:
    normalized = _normalize_anchor_fallback_text(value)
    tokens: set[str] = set()
    for token in normalized.split():
        if len(token) < 3:
            continue
        if token in _ANCHOR_FALLBACK_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _anchor_fallback_phrases(terms: Sequence[str]) -> list[str]:
    phrases: list[str] = []
    for term in terms:
        normalized = _normalize_anchor_fallback_text(term)
        if not normalized:
            continue
        phrase_tokens = [
            token
            for token in normalized.split()
            if len(token) >= 3
            and token not in _ANCHOR_FALLBACK_STOPWORDS
        ]
        if len(phrase_tokens) >= 2:
            phrases.append(" ".join(phrase_tokens))
    return phrases


def _is_excluded_fallback_title(title: str | None) -> bool:
    normalized = _normalize_anchor_fallback_text(title)
    return any(
        pattern in normalized
        for pattern in _EXCLUDED_FALLBACK_TITLE_PATTERNS
    )


def _deduplicate_results(
    results: Sequence[KGSectionResult],
) -> list[KGSectionResult]:
    output: list[KGSectionResult] = []
    seen: set[str] = set()
    for item in results:
        if item.section_uid in seen:
            continue
        seen.add(item.section_uid)
        output.append(item)
    return output


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


def _validate_non_negative_int(
    value: int,
    *,
    field_name: str,
    maximum: int | None = None,
) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc

    if normalized < 0:
        raise ValueError(f"{field_name} must be at least 0")

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
