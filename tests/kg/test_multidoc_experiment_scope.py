from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.kg.experiment_scope import (
    load_run_document_scope,
    normalize_document_scope,
    validate_result_document_scope,
    validate_selected_gold_document_scope,
)


class Result:
    def __init__(self, document_id: str, section_uid: str = "s") -> None:
        self.document_id = document_id
        self.section_uid = section_uid


def question(qid: str, *docs: str):
    return {
        "id": qid,
        "question": "q",
        "sources": [
            {"document_id": doc, "sections": ["1"]}
            for doc in docs
        ],
    }


def test_normalize_document_scope_deduplicates_preserving_order():
    assert normalize_document_scope(["Cardiomyopathies_2023", "Cardio-oncology_2022", "Cardiomyopathies_2023"]) == [
        "Cardiomyopathies_2023",
        "Cardio-oncology_2022",
    ]


def test_gold_scope_accepts_cross_document_question_inside_scope():
    indexed = {
        "Q1": question("Q1", "Cardiomyopathies_2023", "Cardio-oncology_2022")
    }
    validate_selected_gold_document_scope(
        indexed,
        ["Q1"],
        ["Cardiomyopathies_2023", "Cardio-oncology_2022"],
    )


def test_gold_scope_rejects_dataset_leak():
    indexed = {
        "Q1": question("Q1", "Cardiomyopathies_2023", "Chronic_Coronary_Syndromes_2024")
    }
    with pytest.raises(ValueError, match="outside the configured KG document scope"):
        validate_selected_gold_document_scope(
            indexed,
            ["Q1"],
            ["Cardiomyopathies_2023", "Cardio-oncology_2022"],
        )


def test_result_scope_rejects_candidate_from_third_document():
    with pytest.raises(RuntimeError, match="document-scope leak"):
        validate_result_document_scope(
            [Result("Chronic_Coronary_Syndromes_2024")],
            ["Cardiomyopathies_2023", "Cardio-oncology_2022"],
            stage="raw",
        )


def test_run_manifest_document_scope_round_trip(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "document_filtering": [
                        "Cardiomyopathies_2023",
                        "Cardio-oncology_2022",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_run_document_scope(run) == [
        "Cardiomyopathies_2023",
        "Cardio-oncology_2022",
    ]
