from __future__ import annotations

from scripts.augment_kg_gold_coverage import (
    build_coverage_metadata,
    enrich_artifact,
)


def make_resolution(
    *,
    question_id: str,
    gold_id: str,
    gold_label: str,
    status: str,
    document_id: str = "Syncope_2018",
) -> dict:
    return {
        "gold_id": gold_id,
        "question_id": question_id,
        "question": f"Question {question_id}",
        "gold_label": gold_label,
        "status": status,
        "source": {
            "document_id": document_id,
            "filename": f"{document_id}.md",
        },
        "parsed_key": {
            "document_id": document_id.casefold(),
            "printed_section_id": gold_label.split(".", 1)[0],
            "title": gold_label.casefold(),
        },
        "same_number_candidates": [],
    }


def make_artifact() -> dict:
    return {
        "schema_version": "1.0",
        "summary": {
            "graph_document_ids_found": [
                "Cardiomyopathies_2023",
                "Syncope_2018",
            ]
        },
        "resolutions": [
            make_resolution(
                question_id="CM_01",
                gold_id="CM_01::0",
                gold_label="7.1.1.1. Diagnostic criteria",
                status="resolved",
                document_id="Cardiomyopathies_2023",
            ),
            make_resolution(
                question_id="SYN_02",
                gold_id="SYN_02::0",
                gold_label="4.1. Initial evaluation",
                status="resolved",
            ),
            make_resolution(
                question_id="SYN_02",
                gold_id="SYN_02::1",
                gold_label="4.1.2. Management",
                status="unresolved",
            ),
        ],
    }


def test_build_coverage_metadata_separates_clean_and_end_to_end_sets() -> None:
    coverage = build_coverage_metadata(make_artifact())
    summary = coverage["coverage_summary"]

    assert summary["section_annotations"] == {
        "total": 3,
        "available": 2,
        "unavailable": 1,
        "coverage_ratio": 2 / 3,
    }
    assert summary["questions"]["total"] == 2
    assert summary["questions"]["full_coverage"] == 1
    assert summary["questions"]["partial_coverage"] == 1

    evaluation_sets = summary["evaluation_sets"]
    assert evaluation_sets["section_level_clean_question_ids"] == ["CM_01"]
    assert evaluation_sets["section_level_end_to_end_question_ids"] == [
        "CM_01",
        "SYN_02",
    ]
    assert evaluation_sets["document_level_clean_question_ids"] == [
        "CM_01",
        "SYN_02",
    ]


def test_partial_question_contains_issue_and_correct_eligibility() -> None:
    coverage = build_coverage_metadata(make_artifact())
    question = next(
        item
        for item in coverage["question_coverage"]
        if item["question_id"] == "SYN_02"
    )

    assert question["coverage_status"] == "partial"
    assert question["total_gold_sections"] == 2
    assert question["available_gold_sections"] == 1
    assert question["unavailable_gold_sections"] == 1
    assert question["section_coverage_ratio"] == 0.5
    assert question["evaluation_eligibility"] == {
        "document_level_clean": True,
        "document_level_end_to_end": True,
        "section_level_clean": False,
        "section_level_end_to_end": True,
    }
    assert question["unavailable_gold_details"][0]["status"] == "unresolved"
    assert question["clean_evaluation_exclusion_reasons"] == [
        "incomplete_gold_section_coverage_in_kg"
    ]


def test_missing_document_excludes_only_clean_document_evaluation() -> None:
    artifact = make_artifact()
    artifact["summary"]["graph_document_ids_found"] = [
        "Cardiomyopathies_2023"
    ]

    coverage = build_coverage_metadata(artifact)
    question = next(
        item
        for item in coverage["question_coverage"]
        if item["question_id"] == "SYN_02"
    )

    assert question["missing_document_ids"] == ["Syncope_2018"]
    assert question["evaluation_eligibility"]["document_level_clean"] is False
    assert question["evaluation_eligibility"]["document_level_end_to_end"] is True


def test_enrich_artifact_preserves_original_data_and_updates_schema() -> None:
    artifact = make_artifact()
    enriched = enrich_artifact(artifact)

    assert enriched["schema_version"] == "1.1"
    assert enriched["resolutions"] == artifact["resolutions"]
    assert "coverage_policy" in enriched
    assert "coverage_summary" in enriched
    assert "question_coverage" in enriched
    assert artifact["schema_version"] == "1.0"
    assert "coverage_summary" not in artifact
