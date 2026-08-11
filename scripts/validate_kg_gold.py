"""Validate section-level gold annotations against the Neo4j KG.

The gold dataset remains independent from Neo4j-internal UIDs. Each gold
annotation is resolved through the exact, human-readable key:

    document_id + printed_section_id + normalized section title

The script writes a complete ``gold_resolution.json`` artifact before
optionally failing when unresolved, mismatched, or ambiguous annotations are
found.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from agentic_rag.evaluation.retrieval import (
    SectionKey,
    normalize_document_id,
    normalize_printed_section_id,
    normalize_section_title,
    section_key_from_gold,
)


DEFAULT_DATASET_PATH = Path("tests/data/subset_test_en_cm_syn.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/kg_retrieval")

_SECTION_LOOKUP_CYPHER = """
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
WHERE d.doc_id IN $document_ids
RETURN
    s.uid AS section_uid,
    d.doc_id AS document_id,
    s.section_id AS section_id,
    s.printed_section_id AS printed_section_id,
    s.title AS title,
    s.level AS level,
    s.page_start AS page_start,
    s.page_end AS page_end,
    s.part_index AS part_index,
    s.part_count AS part_count,
    s.section_view_role AS section_view_role,
    s.retrieval_unit_id AS retrieval_unit_id,
    s.represented_section_ids AS represented_section_ids,
    s.absorbed_section_ids AS absorbed_section_ids,
    s.is_aggregated AS is_aggregated
ORDER BY
    document_id,
    printed_section_id,
    part_index,
    section_uid
"""


class ReadClient(Protocol):
    """Minimal interface required from the read-only KG client."""

    def run_read(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        ...


def load_dataset(dataset_path: Path) -> dict[str, Any]:
    """Load and minimally validate a schema-v2 retrieval dataset."""

    resolved_path = dataset_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {resolved_path}")

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset root must be a JSON object")

    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Dataset must contain a 'questions' list")

    return payload


def collect_gold_annotations(
    dataset: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten all question sources and section labels into gold records."""

    annotations: list[dict[str, Any]] = []
    seen_gold_ids: set[str] = set()

    for question_index, question in enumerate(dataset["questions"]):
        if not isinstance(question, Mapping):
            raise ValueError(
                f"Question at index {question_index} must be an object"
            )

        question_id = str(question.get("id") or "").strip()
        question_text = str(question.get("question") or "").strip()
        if not question_id:
            raise ValueError(
                f"Question at index {question_index} has no non-empty id"
            )
        if not question_text:
            raise ValueError(f"Question {question_id!r} has no text")

        sources = question.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Question {question_id!r} has no sources")

        for source_index, source in enumerate(sources):
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"Source {source_index} of question {question_id!r} "
                    "must be an object"
                )

            document_id = str(source.get("document_id") or "").strip()
            if not document_id:
                raise ValueError(
                    f"Source {source_index} of question {question_id!r} "
                    "has no document_id"
                )

            sections = source.get("sections")
            if not isinstance(sections, list) or not sections:
                raise ValueError(
                    f"Source {source_index} of question {question_id!r} "
                    "has no section labels"
                )

            for section_index, section_label_raw in enumerate(sections):
                section_label = str(section_label_raw or "").strip()
                key = section_key_from_gold(document_id, section_label)
                gold_id = (
                    f"{question_id}::source_{source_index}::"
                    f"section_{section_index}"
                )

                if gold_id in seen_gold_ids:
                    raise ValueError(f"Duplicate generated gold_id: {gold_id}")
                seen_gold_ids.add(gold_id)

                annotations.append(
                    {
                        "gold_id": gold_id,
                        "question_id": question_id,
                        "question": question_text,
                        "source_index": source_index,
                        "section_index": section_index,
                        "document_id": document_id,
                        "filename": source.get("filename"),
                        "guideline_name": source.get("guideline_name"),
                        "gold_label": section_label,
                        "key": key,
                    }
                )

    return annotations


def fetch_graph_sections(
    client: ReadClient,
    document_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Load all Section metadata needed for exact gold resolution."""

    unique_document_ids = list(dict.fromkeys(document_ids))
    if not unique_document_ids:
        return []

    return client.run_read(
        _SECTION_LOOKUP_CYPHER,
        {"document_ids": unique_document_ids},
    )


def _safe_normalize(value: Any, normalizer) -> str | None:
    if value is None:
        return None
    try:
        return normalizer(str(value))
    except ValueError:
        return None


def serialize_graph_section(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable raw and normalized metadata for one Section node."""

    return {
        "section_uid": row.get("section_uid"),
        "document_id": row.get("document_id"),
        "section_id": row.get("section_id"),
        "printed_section_id": row.get("printed_section_id"),
        "title": row.get("title"),
        "level": row.get("level"),
        "page_start": row.get("page_start"),
        "page_end": row.get("page_end"),
        "part_index": row.get("part_index"),
        "part_count": row.get("part_count"),
        "section_view_role": row.get("section_view_role"),
        "retrieval_unit_id": row.get("retrieval_unit_id"),
        "represented_section_ids": list(
            row.get("represented_section_ids") or []
        ),
        "absorbed_section_ids": list(
            row.get("absorbed_section_ids") or []
        ),
        "is_aggregated": bool(row.get("is_aggregated")),
        "normalized": {
            "document_id": _safe_normalize(
                row.get("document_id"), normalize_document_id
            ),
            "printed_section_id": _safe_normalize(
                row.get("printed_section_id"),
                normalize_printed_section_id,
            ),
            "title": _safe_normalize(
                row.get("title"), normalize_section_title
            ),
        },
    }


def build_section_number_index(
    graph_sections: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Index graph nodes by normalized document and printed section number."""

    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for raw_row in graph_sections:
        row = serialize_graph_section(raw_row)
        normalized = row["normalized"]
        document_id = normalized["document_id"]
        printed_section_id = normalized["printed_section_id"]

        if document_id is None or printed_section_id is None:
            continue

        index[(document_id, printed_section_id)].append(row)

    return dict(index)


def build_represented_section_index(
    graph_sections: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Index retrieval units by every original section they represent.

    Fine-grained original sections may be absorbed by a hierarchical
    Section View and therefore no longer exist as independent Section nodes.
    Only actual retrieval units are allowed to resolve such gold targets.
    """

    index: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for raw_row in graph_sections:
        row = serialize_graph_section(raw_row)
        normalized = row["normalized"]

        document_id = normalized["document_id"]
        if document_id is None:
            continue

        if (
            str(row.get("section_view_role") or "")
            .strip()
            .casefold()
            != "retrieval"
        ):
            continue

        if not row.get("retrieval_unit_id"):
            continue

        for raw_section_id in row.get(
            "represented_section_ids"
        ) or []:
            represented_id = _safe_normalize(
                raw_section_id,
                normalize_printed_section_id,
            )
            if represented_id is None:
                continue

            index[(document_id, represented_id)].append(row)

    return dict(index)


def _is_valid_multi_part_group(
    candidates: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether duplicate exact matches are technical parts of one section."""

    if len(candidates) < 2:
        return False

    part_counts = {candidate.get("part_count") for candidate in candidates}
    part_indices = [candidate.get("part_index") for candidate in candidates]

    if len(part_counts) != 1:
        return False

    part_count = next(iter(part_counts))
    if not isinstance(part_count, int) or part_count <= 1:
        return False

    if not all(isinstance(index, int) for index in part_indices):
        return False

    if len(set(part_indices)) != len(part_indices):
        return False

    return all(0 <= index < part_count for index in part_indices)


def resolve_gold_annotation(
    annotation: Mapping[str, Any],
    section_index: Mapping[
        tuple[str, str],
        list[dict[str, Any]],
    ],
    represented_section_index: Mapping[
        tuple[str, str],
        list[dict[str, Any]],
    ] | None = None,
) -> dict[str, Any]:
    """Resolve one gold label against the KG Section View.

    Resolution order:

    1. Prefer the original exact contract:
       document + printed section number + normalized title.
    2. If no node with that section number exists, allow a unique retrieval
       unit whose represented_section_ids contains the original gold section.

    The retrieval-owner title is deliberately not compared with the gold
    title in case 2, because an aggregated parent naturally has a different
    title from the absorbed child section.
    """

    key = annotation["key"]
    if not isinstance(key, SectionKey):
        raise TypeError(
            "annotation['key'] must be a SectionKey"
        )

    number_candidates = list(
        section_index.get(
            (
                key.document_id,
                key.printed_section_id,
            ),
            [],
        )
    )

    exact_candidates = [
        candidate
        for candidate in number_candidates
        if candidate["normalized"]["title"] == key.title
    ]

    represented_candidates = list(
        (represented_section_index or {}).get(
            (
                key.document_id,
                key.printed_section_id,
            ),
            [],
        )
    )

    if not number_candidates:
        if len(represented_candidates) == 1:
            status = "resolved"
            resolution_kind = (
                "represented_by_retrieval_unit"
            )
            matched_nodes = represented_candidates

        elif len(represented_candidates) > 1:
            # One original section should have one retrieval owner.
            status = "ambiguous"
            resolution_kind = None
            matched_nodes = represented_candidates

        else:
            status = "unresolved"
            resolution_kind = None
            matched_nodes = []

    elif not exact_candidates:
        # If the numbered node actually exists, preserve the strict
        # historical title-matching contract instead of hiding a mismatch
        # behind Section View provenance.
        status = "title_mismatch"
        resolution_kind = None
        matched_nodes = []

    elif len(exact_candidates) == 1:
        status = "resolved"
        resolution_kind = "single_node"
        matched_nodes = exact_candidates

    elif _is_valid_multi_part_group(exact_candidates):
        status = "resolved"
        resolution_kind = "multi_part_section"
        matched_nodes = sorted(
            exact_candidates,
            key=lambda item: (
                item.get("part_index") is None,
                item.get("part_index"),
                str(item.get("section_uid") or ""),
            ),
        )

    else:
        status = "ambiguous"
        resolution_kind = None
        matched_nodes = exact_candidates

    return {
        "gold_id": annotation["gold_id"],
        "question_id": annotation["question_id"],
        "question": annotation["question"],
        "source_index": annotation["source_index"],
        "section_index": annotation["section_index"],
        "gold_label": annotation["gold_label"],
        "source": {
            "document_id": annotation["document_id"],
            "filename": annotation.get("filename"),
            "guideline_name": annotation.get(
                "guideline_name"
            ),
        },
        "parsed_key": key.to_dict(),
        "status": status,
        "resolution_kind": resolution_kind,
        "matched_node_count": len(matched_nodes),
        "matched_nodes": matched_nodes,
        "same_number_candidates": number_candidates,
        "represented_section_candidates": (
            represented_candidates
        ),
    }

def validate_dataset_gold(
    client: ReadClient,
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve every gold annotation in a loaded dataset against Neo4j."""

    annotations = collect_gold_annotations(dataset)

    requested_document_ids = list(
        dict.fromkeys(
            annotation["document_id"]
            for annotation in annotations
        )
    )

    graph_sections = fetch_graph_sections(
        client,
        requested_document_ids,
    )

    section_index = build_section_number_index(
        graph_sections
    )

    represented_section_index = (
        build_represented_section_index(
            graph_sections
        )
    )

    resolutions = [
        resolve_gold_annotation(
            annotation,
            section_index,
            represented_section_index,
        )
        for annotation in annotations
    ]

    status_counts = Counter(
        item["status"] for item in resolutions
    )

    resolution_kind_counts = Counter(
        item["resolution_kind"]
        for item in resolutions
        if item["resolution_kind"] is not None
    )

    graph_document_ids = sorted(
        {
            str(row.get("document_id"))
            for row in graph_sections
            if row.get("document_id") is not None
        }
    )

    return {
        "summary": {
            "question_count": len(
                dataset["questions"]
            ),
            "gold_annotation_count": len(
                annotations
            ),
            "requested_document_ids": (
                requested_document_ids
            ),
            "graph_document_ids_found": (
                graph_document_ids
            ),
            "graph_section_nodes_loaded": len(
                graph_sections
            ),
            "status_counts": {
                status: status_counts.get(
                    status,
                    0,
                )
                for status in (
                    "resolved",
                    "unresolved",
                    "title_mismatch",
                    "ambiguous",
                )
            },
            "resolution_kind_counts": dict(
                sorted(
                    resolution_kind_counts.items()
                )
            ),
            "all_resolved": all(
                item["status"] == "resolved"
                for item in resolutions
            ),
        },
        "resolutions": resolutions,
    }

def build_artifact(
    *,
    run_id: str,
    dataset_path: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete JSON artifact for one validation run."""

    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_type": "kg_gold_preflight",
        "matching_policy": {
            "document_id": "normalized exact match",
            "printed_section_id": "normalized exact match",
            "title": (
                "normalized exact match for direct Section nodes; "
                "not compared when an absorbed gold section is resolved "
                "through represented_section_ids"
            ),
            "section_view_fallback": (
                "when no direct numbered node exists, resolve uniquely "
                "through a retrieval unit whose represented_section_ids "
                "contains the gold section"
            ),
            "fuzzy_matching": False,
            "neo4j_uid_required_in_gold": False,
        },
        "dataset": str(dataset_path.expanduser().resolve()),
        **validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate section-level retrieval gold annotations against "
            "Neo4j Section nodes."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Environment file containing Neo4j credentials (default: .env)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Schema-v2 retrieval dataset "
            f"(default: {DEFAULT_DATASET_PATH})"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory for validation artifacts "
            f"(default: {DEFAULT_OUTPUT_ROOT})"
        ),
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help=(
            "Exit with status 1 after saving the artifact if any gold "
            "annotation is not resolved."
        ),
    )
    return parser.parse_args()


def print_summary(artifact: Mapping[str, Any], output_path: Path) -> None:
    summary = artifact["summary"]
    counts = summary["status_counts"]

    print()
    print("KG gold preflight")
    print(f"Questions: {summary['question_count']}")
    print(f"Gold annotations: {summary['gold_annotation_count']}")
    print(f"Graph Section nodes loaded: {summary['graph_section_nodes_loaded']}")
    print(
        "Statuses: "
        f"resolved={counts['resolved']}, "
        f"unresolved={counts['unresolved']}, "
        f"title_mismatch={counts['title_mismatch']}, "
        f"ambiguous={counts['ambiguous']}"
    )
    print(f"All resolved: {summary['all_resolved']}")
    print(f"Saved: {output_path}")

    issues = [
        item
        for item in artifact["resolutions"]
        if item["status"] != "resolved"
    ]
    if issues:
        print()
        print("Validation issues:")
        for item in issues:
            candidate_titles = [
                candidate.get("title")
                for candidate in item["same_number_candidates"]
            ]
            print(
                f"- {item['question_id']} | {item['gold_label']} | "
                f"{item['status']} | candidates={candidate_titles}"
            )


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    dataset = load_dataset(dataset_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"kg_gold_validation_{timestamp}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "gold_resolution.json"

    try:
        from agentic_rag.kg.client import Neo4jKGClient

        with Neo4jKGClient.from_env(env_path=args.env_file) as client:
            validation = validate_dataset_gold(client, dataset)

        artifact = build_artifact(
            run_id=run_id,
            dataset_path=dataset_path,
            validation=validation,
        )
        output_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise

    print_summary(artifact, output_path)

    if args.fail_on_issues and not artifact["summary"]["all_resolved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
