"""Add corpus-coverage and evaluation-eligibility metadata to a KG gold artifact.

This script is a pure post-processing step. It reads an existing
``gold_resolution.json`` produced by ``validate_kg_gold.py`` and writes an
enriched copy without querying Neo4j or rerunning retrieval.

The enriched artifact separates two evaluation views:

- clean retrieval evaluation: include only questions whose complete gold set is
  represented in the backend;
- end-to-end evaluation: include every question and keep unavailable gold
  sections in the metric denominator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from agentic_rag.evaluation.retrieval import normalize_document_id


DEFAULT_OUTPUT_NAME = "gold_resolution_enriched.json"
RESOLVED_STATUS = "resolved"


def load_gold_resolution(path: Path) -> dict[str, Any]:
    """Load and minimally validate an existing gold-resolution artifact."""

    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Gold-resolution artifact not found: {resolved_path}")

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gold-resolution artifact root must be a JSON object")

    summary = payload.get("summary")
    resolutions = payload.get("resolutions")

    if not isinstance(summary, Mapping):
        raise ValueError("Artifact must contain a 'summary' object")
    if not isinstance(resolutions, list):
        raise ValueError("Artifact must contain a 'resolutions' list")

    return payload


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalized_document_set(values: Sequence[Any]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        if value is None:
            continue
        normalized.add(normalize_document_id(str(value)))
    return normalized


def _serialize_unavailable_resolution(
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    source = resolution.get("source")
    source_payload = dict(source) if isinstance(source, Mapping) else {}

    parsed_key = resolution.get("parsed_key")
    parsed_payload = dict(parsed_key) if isinstance(parsed_key, Mapping) else {}

    candidates = resolution.get("same_number_candidates")
    candidate_payload = candidates if isinstance(candidates, list) else []

    return {
        "gold_id": resolution.get("gold_id"),
        "gold_label": resolution.get("gold_label"),
        "status": resolution.get("status"),
        "source": source_payload,
        "parsed_key": parsed_payload,
        "same_number_candidates": candidate_payload,
    }


def build_coverage_metadata(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build aggregate and per-question coverage metadata."""

    resolutions = artifact.get("resolutions")
    summary = artifact.get("summary")
    if not isinstance(resolutions, list):
        raise ValueError("Artifact must contain a 'resolutions' list")
    if not isinstance(summary, Mapping):
        raise ValueError("Artifact must contain a 'summary' object")

    graph_documents = _normalized_document_set(
        list(summary.get("graph_document_ids_found") or [])
    )

    grouped: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
    for index, raw_resolution in enumerate(resolutions):
        if not isinstance(raw_resolution, Mapping):
            raise ValueError(f"Resolution at index {index} must be an object")

        question_id = str(raw_resolution.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Resolution at index {index} has no question_id")

        grouped.setdefault(question_id, []).append(raw_resolution)

    question_coverage: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    clean_document_ids: list[str] = []
    clean_section_ids: list[str] = []
    end_to_end_ids: list[str] = []

    total_annotations = 0
    available_annotations = 0

    for question_id, items in grouped.items():
        question = str(items[0].get("question") or "")
        total = len(items)
        resolved_items = [
            item for item in items if item.get("status") == RESOLVED_STATUS
        ]
        unavailable_items = [
            item for item in items if item.get("status") != RESOLVED_STATUS
        ]

        total_annotations += total
        available_annotations += len(resolved_items)

        source_document_ids = _ordered_unique(
            [
                str((item.get("source") or {}).get("document_id") or "").strip()
                for item in items
                if isinstance(item.get("source"), Mapping)
                and str((item.get("source") or {}).get("document_id") or "").strip()
            ]
        )
        normalized_source_documents = {
            normalize_document_id(document_id)
            for document_id in source_document_ids
        }
        missing_document_ids = [
            document_id
            for document_id in source_document_ids
            if normalize_document_id(document_id) not in graph_documents
        ]

        all_documents_available = (
            bool(normalized_source_documents)
            and not missing_document_ids
        )
        full_section_coverage = not unavailable_items

        if full_section_coverage:
            coverage_status = "full"
        elif resolved_items:
            coverage_status = "partial"
        else:
            coverage_status = "none"
        status_counts[coverage_status] += 1

        exclusion_reasons: list[str] = []
        if not all_documents_available:
            exclusion_reasons.append("missing_gold_document_in_kg")
        if not full_section_coverage:
            exclusion_reasons.append("incomplete_gold_section_coverage_in_kg")

        if all_documents_available:
            clean_document_ids.append(question_id)
        if full_section_coverage:
            clean_section_ids.append(question_id)
        end_to_end_ids.append(question_id)

        question_coverage.append(
            {
                "question_id": question_id,
                "question": question,
                "source_document_ids": source_document_ids,
                "missing_document_ids": missing_document_ids,
                "total_gold_sections": total,
                "available_gold_sections": len(resolved_items),
                "unavailable_gold_sections": len(unavailable_items),
                "section_coverage_ratio": (
                    len(resolved_items) / total if total else 0.0
                ),
                "coverage_status": coverage_status,
                "full_coverage": full_section_coverage,
                "unavailable_gold_details": [
                    _serialize_unavailable_resolution(item)
                    for item in unavailable_items
                ],
                "evaluation_eligibility": {
                    "document_level_clean": all_documents_available,
                    "document_level_end_to_end": True,
                    "section_level_clean": full_section_coverage,
                    "section_level_end_to_end": True,
                },
                "clean_evaluation_exclusion_reasons": exclusion_reasons,
            }
        )

    total_questions = len(question_coverage)
    unavailable_annotations = total_annotations - available_annotations
    full_questions = status_counts.get("full", 0)

    return {
        "coverage_policy": {
            "clean_retrieval_evaluation": (
                "A question is eligible only when its complete gold target set "
                "is represented in the evaluated backend."
            ),
            "end_to_end_evaluation": (
                "All questions remain included; unavailable gold targets stay "
                "in the metric denominator and therefore count as failures."
            ),
            "document_level_clean_rule": (
                "All gold source documents for the question exist in the KG."
            ),
            "section_level_clean_rule": (
                "Every gold section annotation for the question has status "
                "'resolved'."
            ),
        },
        "coverage_summary": {
            "section_annotations": {
                "total": total_annotations,
                "available": available_annotations,
                "unavailable": unavailable_annotations,
                "coverage_ratio": (
                    available_annotations / total_annotations
                    if total_annotations
                    else 0.0
                ),
            },
            "questions": {
                "total": total_questions,
                "full_coverage": full_questions,
                "partial_coverage": status_counts.get("partial", 0),
                "no_coverage": status_counts.get("none", 0),
                "full_coverage_ratio": (
                    full_questions / total_questions if total_questions else 0.0
                ),
            },
            "evaluation_sets": {
                "document_level_clean_question_ids": clean_document_ids,
                "document_level_end_to_end_question_ids": end_to_end_ids,
                "section_level_clean_question_ids": clean_section_ids,
                "section_level_end_to_end_question_ids": end_to_end_ids,
            },
        },
        "question_coverage": question_coverage,
    }


def enrich_artifact(
    artifact: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Return a copy of the original artifact with coverage metadata added."""

    enriched = deepcopy(dict(artifact))
    coverage = build_coverage_metadata(artifact)

    enriched["schema_version"] = "1.1"
    enriched["coverage_generated_at"] = datetime.now(timezone.utc).isoformat()
    if source_path is not None:
        enriched["coverage_source_artifact"] = str(
            source_path.expanduser().resolve()
        )

    enriched.update(coverage)
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add corpus-coverage and evaluation-eligibility metadata to an "
            "existing KG gold-resolution artifact without rerunning Neo4j."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Existing gold_resolution.json artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. Defaults to gold_resolution_enriched.json in the "
            "same directory as the input artifact."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def print_summary(enriched: Mapping[str, Any], output_path: Path) -> None:
    coverage = enriched["coverage_summary"]
    sections = coverage["section_annotations"]
    questions = coverage["questions"]
    evaluation_sets = coverage["evaluation_sets"]

    excluded = sorted(
        set(evaluation_sets["section_level_end_to_end_question_ids"])
        - set(evaluation_sets["section_level_clean_question_ids"])
    )

    print()
    print("KG gold coverage augmentation")
    print(
        "Section annotations available: "
        f"{sections['available']}/{sections['total']} "
        f"({sections['coverage_ratio']:.1%})"
    )
    print(
        "Questions with full section coverage: "
        f"{questions['full_coverage']}/{questions['total']} "
        f"({questions['full_coverage_ratio']:.1%})"
    )
    print(
        "Clean section-level evaluation questions: "
        f"{len(evaluation_sets['section_level_clean_question_ids'])}"
    )
    print(
        "End-to-end section-level evaluation questions: "
        f"{len(evaluation_sets['section_level_end_to_end_question_ids'])}"
    )
    print(f"Excluded from clean section-level metrics: {excluded or 'none'}")
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_name(DEFAULT_OUTPUT_NAME)
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    artifact = load_gold_resolution(input_path)
    enriched = enrich_artifact(artifact, source_path=input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print_summary(enriched, output_path)


if __name__ == "__main__":
    main()
