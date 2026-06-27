"""Pure helpers for batch evaluation of KG retrieval runs.

The functions in this module are backend-independent once a ranked list of
``KGSectionResult`` objects has been produced. They convert retrieved sections
and dataset annotations to canonical ``SectionKey`` values, calculate section-
and document-level metrics, and expose candidate-generation diagnostics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agentic_rag.evaluation.metrics import (
    DEFAULT_CUTOFFS,
    aggregate_query_metrics,
    compute_query_metrics,
    deduplicate_ranking,
    get_evaluation_question_ids,
)
from agentic_rag.evaluation.retrieval import (
    SectionKey,
    deduplicate_section_keys,
    normalize_document_id,
    section_key_from_gold,
    section_key_from_result,
)
from agentic_rag.kg.models import KGSectionResult


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
    """Return the deduplicated section gold set for one question."""

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
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("Every source must be an object")
        document_id = str(source.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("Every source must contain document_id")
        values.append(normalize_document_id(document_id))
    return deduplicate_ranking(values)


def section_keys_from_results(
    results: Iterable[KGSectionResult],
) -> list[SectionKey]:
    """Convert a retrieved Section ranking to canonical keys."""

    return deduplicate_section_keys(
        section_key_from_result(result) for result in results
    )


def document_ranking_from_sections(
    section_ranking: Iterable[SectionKey],
) -> list[str]:
    """Project a Section ranking to first-occurrence document ranking."""

    return deduplicate_ranking(
        section.document_id for section in section_ranking
    )


def candidate_diagnostics(
    gold: Sequence[SectionKey],
    ranking: Sequence[SectionKey],
) -> dict[str, Any]:
    """Describe whether and where gold sections occur in one ranking."""

    gold_set = set(gold)
    positions: dict[SectionKey, int] = {
        item: rank for rank, item in enumerate(ranking, start=1)
    }
    found = [item for item in gold if item in positions]
    missing = [item for item in gold if item not in positions]
    found_ranks = [positions[item] for item in found]

    return {
        "gold_count": len(gold),
        "found_count": len(found),
        "missing_count": len(missing),
        "recall": len(found) / len(gold) if gold else 0.0,
        "any_gold_found": bool(found),
        "all_gold_found": len(found) == len(gold),
        "best_gold_rank": min(found_ranks) if found_ranks else None,
        "worst_gold_rank": max(found_ranks) if found_ranks else None,
        "found_gold": [item.to_dict() for item in found],
        "missing_gold": [item.to_dict() for item in missing],
    }


def evaluate_rankings(
    *,
    gold_sections: Sequence[SectionKey],
    section_ranking: Sequence[SectionKey],
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Calculate section- and document-level metrics for one query."""

    normalized_section_ranking = deduplicate_section_keys(section_ranking)
    gold_documents = deduplicate_ranking(
        section.document_id for section in gold_sections
    )
    document_ranking = document_ranking_from_sections(
        normalized_section_ranking
    )

    return {
        "section": compute_query_metrics(
            gold_sections,
            normalized_section_ranking,
            cutoffs=cutoffs,
        ),
        "document": compute_query_metrics(
            gold_documents,
            document_ranking,
            cutoffs=cutoffs,
        ),
        "section_ranking": [
            item.to_dict() for item in normalized_section_ranking
        ],
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
            get_evaluation_question_ids(
                payload,
                level="section",
                view="clean",
            )
        ),
        "section_end_to_end": set(
            get_evaluation_question_ids(
                payload,
                level="section",
                view="end_to_end",
            )
        ),
        "document_clean": set(
            get_evaluation_question_ids(
                payload,
                level="document",
                view="clean",
            )
        ),
        "document_end_to_end": set(
            get_evaluation_question_ids(
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
        aggregated = aggregate_query_metrics(metric_payloads)
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
