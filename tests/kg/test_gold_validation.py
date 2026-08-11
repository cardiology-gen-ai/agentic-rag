from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.validate_kg_gold import (
    build_represented_section_index,
    build_section_number_index,
    collect_gold_annotations,
    resolve_gold_annotation,
    validate_dataset_gold,
)


class FakeClient:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def run_read(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "cypher": cypher,
                "parameters": dict(parameters or {}),
                "timeout_seconds": timeout_seconds,
            }
        )
        requested = set((parameters or {}).get("document_ids", []))
        return [
            row for row in self.rows if row["document_id"] in requested
        ]


def make_row(
    *,
    uid: str = "Cardiomyopathies_2023::7.1.1.1",
    document_id: str = "Cardiomyopathies_2023",
    section_id: str = "7.1.1.1",
    printed_section_id: str = "7.1.1.1",
    title: str = "Diagnostic criteria",
    part_index: int = 0,
    part_count: int = 1,
    section_view_role: str = "retrieval",
    retrieval_unit_id: str | None = None,
    represented_section_ids: list[str] | None = None,
    absorbed_section_ids: list[str] | None = None,
    is_aggregated: bool = False,
) -> dict[str, Any]:
    if retrieval_unit_id is None:
        retrieval_unit_id = (
            f"{document_id}:{section_id}::retrieval"
        )

    if represented_section_ids is None:
        represented_section_ids = [
            printed_section_id
        ]

    if absorbed_section_ids is None:
        absorbed_section_ids = []

    return {
        "section_uid": uid,
        "document_id": document_id,
        "section_id": section_id,
        "printed_section_id": printed_section_id,
        "title": title,
        "level": 5,
        "page_start": 48,
        "page_end": 48,
        "part_index": part_index,
        "part_count": part_count,
        "section_view_role": section_view_role,
        "retrieval_unit_id": retrieval_unit_id,
        "represented_section_ids": (
            represented_section_ids
        ),
        "absorbed_section_ids": (
            absorbed_section_ids
        ),
        "is_aggregated": is_aggregated,
    }

def make_dataset(
    section_label: str = "7.1.1.1. Diagnostic criteria",
) -> dict[str, Any]:
    return {
        "metadata": {"schema_version": "2.0"},
        "questions": [
            {
                "id": "CM_01",
                "question": "What are the diagnostic criteria?",
                "sources": [
                    {
                        "document_id": "Cardiomyopathies_2023",
                        "filename": "Cardiomyopathies_2023.md",
                        "guideline_name": "Cardiomyopathies",
                        "sections": [section_label],
                    }
                ],
            }
        ],
    }


def test_collect_gold_annotations_builds_canonical_key() -> None:
    annotations = collect_gold_annotations(make_dataset())

    assert len(annotations) == 1
    assert annotations[0]["key"].document_id == "cardiomyopathies_2023"
    assert annotations[0]["key"].printed_section_id == "7.1.1.1"
    assert annotations[0]["key"].title == "diagnostic criteria"


def test_exact_section_is_resolved() -> None:
    annotation = collect_gold_annotations(make_dataset())[0]
    index = build_section_number_index([make_row()])

    resolution = resolve_gold_annotation(annotation, index)

    assert resolution["status"] == "resolved"
    assert resolution["resolution_kind"] == "single_node"
    assert resolution["matched_node_count"] == 1


def test_absorbed_gold_section_resolves_to_retrieval_unit() -> None:
    annotation = collect_gold_annotations(
        make_dataset(
            "7.1.4.1.2. Drug therapy"
        )
    )[0]

    owner = make_row(
        uid=(
            "Cardiomyopathies_2023::"
            "7.1.4.1"
        ),
        section_id="7.1.4.1",
        printed_section_id="7.1.4.1",
        title=(
            "Management of left ventricular "
            "outflow tract obstruction"
        ),
        retrieval_unit_id=(
            "Cardiomyopathies_2023:"
            "7.1.4.1:0::retrieval::"
            "max_level_4"
        ),
        represented_section_ids=[
            "7.1.4.1",
            "7.1.4.1.1",
            "7.1.4.1.2",
            "7.1.4.1.3",
        ],
        absorbed_section_ids=[
            "7.1.4.1.1",
            "7.1.4.1.2",
            "7.1.4.1.3",
        ],
        is_aggregated=True,
    )

    direct_index = build_section_number_index(
        [owner]
    )
    represented_index = (
        build_represented_section_index(
            [owner]
        )
    )

    resolution = resolve_gold_annotation(
        annotation,
        direct_index,
        represented_index,
    )

    assert resolution["status"] == "resolved"
    assert (
        resolution["resolution_kind"]
        == "represented_by_retrieval_unit"
    )
    assert resolution["matched_node_count"] == 1
    assert resolution["same_number_candidates"] == []

    matched = resolution["matched_nodes"][0]

    assert (
        matched["printed_section_id"]
        == "7.1.4.1"
    )
    assert (
        "7.1.4.1.2"
        in matched["represented_section_ids"]
    )


def test_wrong_title_is_reported_as_title_mismatch() -> None:
    annotation = collect_gold_annotations(make_dataset())[0]
    index = build_section_number_index(
        [make_row(title="Management")]
    )

    resolution = resolve_gold_annotation(annotation, index)

    assert resolution["status"] == "title_mismatch"
    assert resolution["matched_node_count"] == 0
    assert resolution["same_number_candidates"][0]["title"] == "Management"


def test_missing_number_is_reported_as_unresolved() -> None:
    annotation = collect_gold_annotations(make_dataset())[0]
    index = build_section_number_index(
        [make_row(printed_section_id="7.1.1.2")]
    )

    resolution = resolve_gold_annotation(annotation, index)

    assert resolution["status"] == "unresolved"
    assert resolution["same_number_candidates"] == []


def test_valid_multiple_parts_are_one_resolved_section() -> None:
    annotation = collect_gold_annotations(make_dataset())[0]
    rows = [
        make_row(
            uid="Cardiomyopathies_2023::7.1.1.1::part_0",
            section_id="7.1.1.1::part_0",
            part_index=0,
            part_count=2,
        ),
        make_row(
            uid="Cardiomyopathies_2023::7.1.1.1::part_1",
            section_id="7.1.1.1::part_1",
            part_index=1,
            part_count=2,
        ),
    ]
    index = build_section_number_index(rows)

    resolution = resolve_gold_annotation(annotation, index)

    assert resolution["status"] == "resolved"
    assert resolution["resolution_kind"] == "multi_part_section"
    assert resolution["matched_node_count"] == 2


def test_duplicate_non_part_nodes_are_ambiguous() -> None:
    annotation = collect_gold_annotations(make_dataset())[0]
    rows = [
        make_row(uid="Cardiomyopathies_2023::duplicate_a"),
        make_row(uid="Cardiomyopathies_2023::duplicate_b"),
    ]
    index = build_section_number_index(rows)

    resolution = resolve_gold_annotation(annotation, index)

    assert resolution["status"] == "ambiguous"
    assert resolution["matched_node_count"] == 2


def test_validate_dataset_gold_queries_only_gold_documents() -> None:
    client = FakeClient([make_row()])

    validation = validate_dataset_gold(client, make_dataset())

    assert validation["summary"]["all_resolved"] is True
    assert validation["summary"]["status_counts"]["resolved"] == 1
    assert client.calls[0]["parameters"] == {
        "document_ids": ["Cardiomyopathies_2023"]
    }


def test_real_twenty_question_dataset_is_collectable() -> None:
    dataset_path = Path("tests/data/subset_test_en_cm_syn.json")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))

    annotations = collect_gold_annotations(payload)

    assert len(payload["questions"]) == 20
    assert len(annotations) > 20
    assert {item["question_id"] for item in annotations} == {
        item["id"] for item in payload["questions"]
    }
