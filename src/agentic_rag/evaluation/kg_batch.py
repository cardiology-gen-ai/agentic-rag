"""Pure helpers for batch evaluation of KG retrieval runs.

The canonical retrieval metrics live in :mod:`agentic_rag.evaluation.metrics`
and are shared with the classic RAG evaluation path.  This module adapts KG
``Section`` results to the same coverage-aware ``RetrievedEvidence`` contract
without changing the metric implementation itself.

``SectionKey`` remains useful for validating human-readable gold annotations
against Neo4j sections.  Retrieval scoring, however, is based on original
section coverage so that one hierarchical retrieval unit can correctly cover
multiple gold sections while still occupying one rank position.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from agentic_rag.evaluation.evidence import EvidenceSection, RetrievedEvidence
from agentic_rag.evaluation.metrics import CoverageMetrics, compute_coverage_metrics
from agentic_rag.evaluation.retrieval import (
    SectionKey,
    deduplicate_section_keys,
    normalize_document_id,
    normalize_printed_section_id,
    section_key_from_gold,
    section_key_from_result,
)
from agentic_rag.kg.models import KGSectionResult


DEFAULT_CUTOFFS: tuple[int, ...] = (1, 3, 5, 10, 20)
_DOCUMENT_SENTINEL_SECTION_ID = "__document__"


def load_retrieval_dataset(path: Path) -> dict[str, Any]:
    """Load a schema-v2 retrieval dataset."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset root must be a JSON object")
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Dataset must contain a 'questions' list")
    return payload


def index_dataset_questions(
    dataset: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Index dataset records by unique question ID."""

    questions = dataset.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Dataset must contain a 'questions' list")

    indexed: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(questions):
        if not isinstance(record, Mapping):
            raise ValueError(f"Question at index {position} must be an object")
        question_id = str(record.get("id") or "").strip()
        question = str(record.get("question") or "").strip()
        if not question_id:
            raise ValueError(f"Question at index {position} has no id")
        if not question:
            raise ValueError(f"Question {question_id!r} has empty text")
        if question_id in indexed:
            raise ValueError(f"Duplicate question id: {question_id!r}")
        indexed[question_id] = record
    return indexed


def gold_section_keys(question: Mapping[str, Any]) -> list[SectionKey]:
    """Return the deduplicated, title-validated section gold set."""

    sources = question.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Question must contain a non-empty 'sources' list")

    keys: list[SectionKey] = []
    for source_index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"Source {source_index} must be an object")
        document_id = str(source.get("document_id") or "").strip()
        sections = source.get("sections")
        if not document_id:
            raise ValueError(f"Source {source_index} has no document_id")
        if not isinstance(sections, list) or not sections:
            raise ValueError(f"Source {source_index} has no section labels")
        for label in sections:
            keys.append(section_key_from_gold(document_id, str(label)))

    return deduplicate_section_keys(keys)


def gold_document_ids(question: Mapping[str, Any]) -> list[str]:
    """Return normalized, deduplicated gold document IDs."""

    sources = question.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Question must contain a non-empty 'sources' list")

    values: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("Every source must be an object")
        document_id = str(source.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("Every source must contain document_id")
        normalized = normalize_document_id(document_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def section_keys_from_results(
    results: Iterable[KGSectionResult],
) -> list[SectionKey]:
    """Convert retrieved KG sections to canonical keys for exact validation.

    This helper intentionally ignores hierarchical coverage.  Use
    :func:`kg_results_to_evidence` for retrieval metrics.
    """

    return deduplicate_section_keys(
        section_key_from_result(result) for result in results
    )


def gold_evidence_sections(
    gold_sections: Iterable[SectionKey],
) -> frozenset[EvidenceSection]:
    """Project validated gold keys to the coverage-aware metric identity."""

    return frozenset(
        EvidenceSection(
            document_id=normalize_document_id(section.document_id),
            section_id=normalize_printed_section_id(section.printed_section_id),
        )
        for section in gold_sections
    )


def kg_results_to_evidence(
    results: Sequence[KGSectionResult],
) -> list[RetrievedEvidence]:
    """Adapt KG Section-view results to coverage-aware evidence units.

    ``represented_section_ids`` is the authoritative coverage set for a
    Section-view retrieval unit.  Older/non-aggregated results that do not
    expose it fall back to their printed section id (then ``section_id``).
    Duplicate retrieval units are collapsed at their first rank.
    """

    evidence: list[RetrievedEvidence] = []
    seen_units: set[tuple[str, str]] = set()

    for raw_rank, result in enumerate(results, start=1):
        document_id = normalize_document_id(result.document_id)
        retrieval_unit_id = (result.retrieval_unit_id or result.section_uid).strip()
        unit_key = (document_id, retrieval_unit_id)
        if unit_key in seen_units:
            continue
        seen_units.add(unit_key)

        represented_ids = list(result.represented_section_ids)
        if not represented_ids:
            fallback = result.printed_section_id or result.section_id
            if fallback is None:
                raise ValueError(
                    f"KG result {result.section_uid!r} has neither "
                    "represented_section_ids nor a section identifier"
                )
            represented_ids = [fallback]

        covered_sections = frozenset(
            EvidenceSection(
                document_id=document_id,
                section_id=normalize_printed_section_id(section_id),
            )
            for section_id in represented_ids
        )

        evidence.append(
            RetrievedEvidence(
                document_id=document_id,
                retrieval_unit_id=retrieval_unit_id,
                covered_sections=covered_sections,
                raw_rank=raw_rank,
                source_record_ids=(result.section_uid,),
                source_type="kg_section_view",
                raw_score=result.score,
            )
        )

    return evidence


def document_ranking_from_sections(
    section_ranking: Iterable[SectionKey],
) -> list[str]:
    """Project a SectionKey ranking to first-occurrence document ranking."""

    ranking: list[str] = []
    seen: set[str] = set()
    for section in section_ranking:
        document_id = normalize_document_id(section.document_id)
        if document_id in seen:
            continue
        seen.add(document_id)
        ranking.append(document_id)
    return ranking


def document_ranking_from_evidence(
    evidence: Iterable[RetrievedEvidence],
) -> list[str]:
    """Project evidence units to first-occurrence document ranking."""

    ranking: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        document_id = normalize_document_id(item.document_id)
        if document_id in seen:
            continue
        seen.add(document_id)
        ranking.append(document_id)
    return ranking


def candidate_diagnostics(
    gold: Sequence[SectionKey],
    ranking: Sequence[SectionKey | KGSectionResult | RetrievedEvidence],
) -> dict[str, Any]:
    """Describe gold-section coverage and first covering ranks.

    For hierarchical KG results, one retrieval unit may therefore contribute
    the same rank to multiple gold sections.
    """

    evidence = _coerce_ranking_to_evidence(ranking)
    gold_by_key = {
        EvidenceSection(
            normalize_document_id(item.document_id),
            normalize_printed_section_id(item.printed_section_id),
        ): item
        for item in gold
    }

    first_positions: dict[EvidenceSection, int] = {}
    for rank, item in enumerate(evidence, start=1):
        for covered in item.covered_sections:
            if covered in gold_by_key and covered not in first_positions:
                first_positions[covered] = rank

    found_keys = [key for key in gold_by_key if key in first_positions]
    missing_keys = [key for key in gold_by_key if key not in first_positions]
    found_ranks = [first_positions[key] for key in found_keys]

    found = [gold_by_key[key] for key in found_keys]
    missing = [gold_by_key[key] for key in missing_keys]

    return {
        "gold_count": len(gold_by_key),
        "found_count": len(found),
        "missing_count": len(missing),
        "recall": len(found) / len(gold_by_key) if gold_by_key else 0.0,
        "any_gold_found": bool(found),
        "all_gold_found": len(found) == len(gold_by_key),
        "best_gold_rank": min(found_ranks) if found_ranks else None,
        "worst_gold_rank": max(found_ranks) if found_ranks else None,
        "found_gold": [item.to_dict() for item in found],
        "missing_gold": [item.to_dict() for item in missing],
    }


def evaluate_rankings(
    *,
    gold_sections: Sequence[SectionKey],
    section_ranking: Sequence[SectionKey | KGSectionResult | RetrievedEvidence],
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Calculate coverage-aware section and document metrics for one query.

    Passing ``KGSectionResult`` values preserves Section-view coverage through
    ``represented_section_ids``.  ``SectionKey`` rankings remain supported for
    legacy exact-section tests/callers, where each key becomes one evidence
    unit covering exactly one original section.
    """

    normalized_cutoffs = tuple(int(k) for k in cutoffs)
    evidence = _coerce_ranking_to_evidence(section_ranking)
    gold_evidence = gold_evidence_sections(gold_sections)

    section_metrics = compute_coverage_metrics(
        evidence,
        gold_evidence,
        cutoffs=normalized_cutoffs,
    )

    gold_documents = _deduplicate_strings(
        section.document_id for section in gold_evidence
    )
    document_ranking = document_ranking_from_evidence(evidence)
    document_metrics = compute_coverage_metrics(
        _document_evidence(document_ranking),
        {
            EvidenceSection(document_id, _DOCUMENT_SENTINEL_SECTION_ID)
            for document_id in gold_documents
        },
        cutoffs=normalized_cutoffs,
    )

    return {
        "section": _flatten_coverage_metrics(section_metrics),
        "document": _flatten_coverage_metrics(document_metrics),
        "section_ranking": [_evidence_to_dict(item) for item in evidence],
        "document_ranking": document_ranking,
    }


def load_coverage_evaluation_sets(
    path: Path,
) -> dict[str, set[str]]:
    """Load clean and end-to-end question sets from an enriched artifact."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Coverage artifact not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Coverage artifact root must be an object")

    return {
        "section_clean": set(
            _get_evaluation_question_ids(payload, level="section", view="clean")
        ),
        "section_end_to_end": set(
            _get_evaluation_question_ids(
                payload,
                level="section",
                view="end_to_end",
            )
        ),
        "document_clean": set(
            _get_evaluation_question_ids(payload, level="document", view="clean")
        ),
        "document_end_to_end": set(
            _get_evaluation_question_ids(
                payload,
                level="document",
                view="end_to_end",
            )
        ),
    }


def build_metric_rows(
    query_records: Sequence[Mapping[str, Any]],
    evaluation_sets: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    """Expand query records into level/view metric rows."""

    rows: list[dict[str, Any]] = []
    for record in query_records:
        question_id = str(record["question_id"])
        mode = str(record["mode"])
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            continue

        for level in ("section", "document"):
            level_metrics = metrics.get(level)
            if not isinstance(level_metrics, Mapping):
                continue
            for view in ("clean", "end_to_end"):
                set_key = f"{level}_{view}"
                if question_id not in evaluation_sets.get(set_key, set()):
                    continue
                row: dict[str, Any] = {
                    "question_id": question_id,
                    "mode": mode,
                    "level": level,
                    "view": view,
                    "status": record.get("status"),
                    "latency_ms": record.get("latency_ms"),
                    "returned_count": record.get("returned_count"),
                }
                row.update(
                    {name: float(value) for name, value in level_metrics.items()}
                )
                rows.append(row)
    return rows


def aggregate_metric_rows(
    metric_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Macro-aggregate rows by mode, level, and evaluation view."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in metric_rows:
        key = (str(row["mode"]), str(row["level"]), str(row["view"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    metadata_keys = {
        "question_id",
        "mode",
        "level",
        "view",
        "status",
        "latency_ms",
        "returned_count",
    }
    for (mode, level, view), rows in sorted(grouped.items()):
        metric_payloads = [
            {
                key: float(value)
                for key, value in row.items()
                if key not in metadata_keys
            }
            for row in rows
        ]
        aggregated = _aggregate_query_metrics(metric_payloads)
        latencies = [
            float(row["latency_ms"])
            for row in rows
            if row.get("latency_ms") is not None
        ]
        output.append(
            {
                "mode": mode,
                "level": level,
                "view": view,
                "query_count": aggregated["query_count"],
                "means": aggregated["means"],
                "statistics": aggregated["statistics"],
                "mean_latency_ms": (
                    sum(latencies) / len(latencies) if latencies else None
                ),
            }
        )
    return output


def _coerce_ranking_to_evidence(
    ranking: Sequence[SectionKey | KGSectionResult | RetrievedEvidence],
) -> list[RetrievedEvidence]:
    if not ranking:
        return []

    if all(isinstance(item, RetrievedEvidence) for item in ranking):
        return list(ranking)  # type: ignore[arg-type]

    if all(isinstance(item, KGSectionResult) for item in ranking):
        return kg_results_to_evidence(ranking)  # type: ignore[arg-type]

    if all(isinstance(item, SectionKey) for item in ranking):
        return _section_keys_to_evidence(ranking)  # type: ignore[arg-type]

    raise TypeError(
        "section_ranking must contain only SectionKey, KGSectionResult, "
        "or RetrievedEvidence values"
    )


def _section_keys_to_evidence(
    ranking: Sequence[SectionKey],
) -> list[RetrievedEvidence]:
    evidence: list[RetrievedEvidence] = []
    seen: set[tuple[str, str, str]] = set()

    for raw_rank, section in enumerate(ranking, start=1):
        document_id = normalize_document_id(section.document_id)
        section_id = normalize_printed_section_id(section.printed_section_id)
        key = (document_id, section_id, section.title)
        if key in seen:
            continue
        seen.add(key)
        retrieval_unit_id = f"{document_id}::{section_id}::{section.title}"
        evidence.append(
            RetrievedEvidence(
                document_id=document_id,
                retrieval_unit_id=retrieval_unit_id,
                covered_sections=frozenset(
                    {EvidenceSection(document_id, section_id)}
                ),
                raw_rank=raw_rank,
                source_record_ids=(retrieval_unit_id,),
                source_type="kg_section_key",
            )
        )

    return evidence


def _document_evidence(
    document_ranking: Sequence[str],
) -> list[RetrievedEvidence]:
    evidence: list[RetrievedEvidence] = []
    for raw_rank, document_id in enumerate(document_ranking, start=1):
        normalized = normalize_document_id(document_id)
        unit_id = f"document::{normalized}"
        evidence.append(
            RetrievedEvidence(
                document_id=normalized,
                retrieval_unit_id=unit_id,
                covered_sections=frozenset(
                    {
                        EvidenceSection(
                            normalized,
                            _DOCUMENT_SENTINEL_SECTION_ID,
                        )
                    }
                ),
                raw_rank=raw_rank,
                source_record_ids=(unit_id,),
                source_type="kg_document",
            )
        )
    return evidence


def _flatten_coverage_metrics(metrics: CoverageMetrics) -> dict[str, float]:
    output: dict[str, float] = {}
    for item in metrics.cutoffs:
        output[f"hit@{item.k}"] = item.hit
        output[f"precision@{item.k}"] = item.precision
        output[f"recall@{item.k}"] = item.recall
        output[f"complete_recall@{item.k}"] = item.complete_recall
        output[f"reciprocal_rank@{item.k}"] = item.reciprocal_rank
    return output


def _evidence_to_dict(item: RetrievedEvidence) -> dict[str, Any]:
    return {
        "document_id": item.document_id,
        "retrieval_unit_id": item.retrieval_unit_id,
        "covered_section_ids": sorted(item.covered_section_ids),
        "raw_rank": item.raw_rank,
        "source_record_ids": list(item.source_record_ids),
        "source_type": item.source_type,
        "raw_score": item.raw_score,
    }


def _get_evaluation_question_ids(
    artifact: Mapping[str, Any],
    *,
    level: str,
    view: str,
) -> list[str]:
    if level not in {"section", "document"}:
        raise ValueError("level must be 'section' or 'document'")
    if view not in {"clean", "end_to_end"}:
        raise ValueError("view must be 'clean' or 'end_to_end'")

    summary = artifact.get("coverage_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("Coverage artifact has no coverage_summary object")
    evaluation_sets = summary.get("evaluation_sets")
    if not isinstance(evaluation_sets, Mapping):
        raise ValueError("coverage_summary has no evaluation_sets object")

    key = f"{level}_level_{view}_question_ids"
    values = evaluation_sets.get(key)
    if not isinstance(values, list):
        raise ValueError(f"evaluation_sets.{key} must be a list")

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        question_id = str(value).strip()
        if not question_id or question_id in seen:
            continue
        seen.add(question_id)
        output.append(question_id)
    return output


def _aggregate_query_metrics(
    metric_payloads: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    if not metric_payloads:
        return {"query_count": 0, "means": {}, "statistics": {}}

    metric_names = sorted(
        {name for payload in metric_payloads for name in payload}
    )
    means: dict[str, float] = {}
    statistics: dict[str, dict[str, float]] = {}

    for name in metric_names:
        values = [float(payload[name]) for payload in metric_payloads if name in payload]
        if not values:
            continue
        mean_value = fmean(values)
        means[name] = mean_value
        statistics[name] = {
            "mean": mean_value,
            "population_std": pstdev(values),
            "min": min(values),
            "max": max(values),
        }

        if name.startswith("reciprocal_rank@"):
            cutoff = name.split("@", 1)[1]
            alias = f"mrr@{cutoff}"
            means[alias] = mean_value
            statistics[alias] = dict(statistics[name])

    return {
        "query_count": len(metric_payloads),
        "means": means,
        "statistics": statistics,
    }


def _deduplicate_strings(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_document_id(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


CLINICAL_MAIN_EVALUATION_GROUPS = frozenset(
    {
        "single_section",
        "multi_section",
        "reasoning_multi_hop",
    }
)


def build_question_group_membership(
    dataset: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return evaluation-group memberships for each dataset question.

    Each question belongs to its explicit dataset evaluation_group, to the
    aggregate all-question group, and, when applicable, to clinical_main.
    Graph-hop diagnostic questions are intentionally excluded from
    clinical_main.
    """
    indexed = index_dataset_questions(dataset)
    memberships: dict[str, tuple[str, ...]] = {}

    for question_id, record in indexed.items():
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(
                f"Question {question_id!r} has no metadata object"
            )

        evaluation_group = str(
            metadata.get("evaluation_group") or ""
        ).strip()
        if not evaluation_group:
            raise ValueError(
                f"Question {question_id!r} has no metadata.evaluation_group"
            )

        groups = [evaluation_group]

        if evaluation_group in CLINICAL_MAIN_EVALUATION_GROUPS:
            groups.append("clinical_main")

        groups.append("all_including_graph_diagnostic")

        memberships[question_id] = tuple(groups)

    return memberships


def aggregate_metric_rows_by_group(
    metric_rows: Sequence[Mapping[str, Any]],
    question_groups: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Macro-aggregate metric rows by dataset evaluation group.

    Base metric rows are left unchanged. This produces an additional reporting
    view layered on top of the existing mode/level/view aggregation.
    """
    group_names = sorted(
        {
            group
            for groups in question_groups.values()
            for group in groups
        }
    )

    output: list[dict[str, Any]] = []

    for group in group_names:
        selected_rows = [
            row
            for row in metric_rows
            if group
            in question_groups.get(
                str(row.get("question_id") or ""),
                (),
            )
        ]

        for aggregate in aggregate_metric_rows(selected_rows):
            output.append(
                {
                    "group": group,
                    **aggregate,
                }
            )

    return output
