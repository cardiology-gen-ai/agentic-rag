from __future__ import annotations

import json

import pytest

from agentic_rag.evaluation.kg_batch import (
    aggregate_metric_rows,
    build_metric_rows,
    candidate_diagnostics,
    evaluate_rankings,
    gold_document_ids,
    gold_section_keys,
    kg_results_to_evidence,
    load_coverage_evaluation_sets,
)
from agentic_rag.evaluation.retrieval import SectionKey
from agentic_rag.kg.models import KGSectionResult


def key(document: str, section_id: str, title: str) -> SectionKey:
    return SectionKey(
        document_id=document.casefold(),
        printed_section_id=section_id,
        title=title.casefold(),
    )


def kg_result(
    *,
    uid: str,
    document: str,
    section_id: str,
    title: str,
    represented: list[str] | None = None,
    retrieval_unit_id: str | None = None,
    score: float = 1.0,
) -> KGSectionResult:
    return KGSectionResult(
        section_uid=uid,
        document_id=document,
        section_id=section_id,
        printed_section_id=section_id,
        title=title,
        text=f"Text for {title}",
        retrieval_unit_id=retrieval_unit_id,
        represented_section_ids=represented or [],
        score=score,
        score_type="concept_match",
    )


def test_gold_extraction_uses_section_and_document_keys() -> None:
    question = {
        "sources": [
            {
                "document_id": "Cardiomyopathies_2023",
                "sections": [
                    "7.1.1.1. Diagnostic criteria",
                    "7.1.1.1. Diagnostic criteria",
                ],
            }
        ]
    }

    sections = gold_section_keys(question)
    documents = gold_document_ids(question)

    assert sections == [
        key("Cardiomyopathies_2023", "7.1.1.1", "Diagnostic criteria")
    ]
    assert documents == ["cardiomyopathies_2023"]


def test_cm01_rank_eight_metrics_and_diagnostics() -> None:
    gold = [key("Cardiomyopathies_2023", "7.1.1.1", "Diagnostic criteria")]
    ranking = [
        key("Syncope_2018", "5.6.3", "Hypertrophic cardiomyopathy"),
        key("Cardiomyopathies_2023", "12.3", "Hypertrophic cardiomyopathy"),
        key("Cardiomyopathies_2023", "3.2.1", "Hypertrophic cardiomyopathy"),
        key("Cardiomyopathies_2023", "7.1", "Hypertrophic cardiomyopathy"),
        key("Cardiomyopathies_2023", "7.1.2", "Genetic testing and family screening"),
        key("Cardiomyopathies_2023", "7.1.3", "Assessment of symptoms"),
        key("Cardiomyopathies_2023", "7.1.4", "Management of symptoms and complications"),
        gold[0],
    ]

    evaluated = evaluate_rankings(
        gold_sections=gold,
        section_ranking=ranking,
    )
    diagnostics = candidate_diagnostics(gold, ranking)

    assert evaluated["section"]["hit@5"] == 0.0
    assert evaluated["section"]["hit@10"] == 1.0
    assert evaluated["section"]["recall@10"] == 1.0
    assert evaluated["section"]["reciprocal_rank@10"] == 0.125
    assert diagnostics["best_gold_rank"] == 8
    assert diagnostics["all_gold_found"] is True


def test_cm06_perfect_top_three_section_metrics() -> None:
    gold = [
        key("Cardiomyopathies_2023", "6.10.3.1", "Anticoagulation"),
        key("Cardiomyopathies_2023", "6.10.3.2", "Rate control"),
        key("Cardiomyopathies_2023", "6.10.3.3", "Rhythm control"),
    ]

    evaluated = evaluate_rankings(
        gold_sections=gold,
        section_ranking=gold,
    )

    assert evaluated["section"]["precision@3"] == 1.0
    assert evaluated["section"]["recall@3"] == 1.0
    assert evaluated["section"]["complete_recall@3"] == 1.0
    assert evaluated["section"]["reciprocal_rank@10"] == 1.0
    assert evaluated["document"]["hit@1"] == 1.0


def test_aggregated_kg_unit_covers_multiple_gold_sections_at_one_rank() -> None:
    gold = [
        key("Cardiomyopathies_2023", "6.10.3.1", "Anticoagulation"),
        key("Cardiomyopathies_2023", "6.10.3.2", "Rate control"),
        key("Cardiomyopathies_2023", "6.10.3.3", "Rhythm control"),
    ]
    ranking = [
        kg_result(
            uid="Cardiomyopathies_2023::6.10.3",
            document="Cardiomyopathies_2023",
            section_id="6.10.3",
            title="Atrial fibrillation",
            retrieval_unit_id="Cardiomyopathies_2023::retrieval::6.10.3",
            represented=["6.10.3", "6.10.3.1", "6.10.3.2", "6.10.3.3"],
        )
    ]

    evidence = kg_results_to_evidence(ranking)
    evaluated = evaluate_rankings(
        gold_sections=gold,
        section_ranking=ranking,
        cutoffs=(1, 3),
    )
    diagnostics = candidate_diagnostics(gold, ranking)

    assert len(evidence) == 1
    assert evidence[0].covered_section_ids == frozenset(
        {"6.10.3", "6.10.3.1", "6.10.3.2", "6.10.3.3"}
    )
    assert evaluated["section"]["precision@1"] == 1.0
    assert evaluated["section"]["recall@1"] == 1.0
    assert evaluated["section"]["complete_recall@1"] == 1.0
    assert diagnostics["found_count"] == 3
    assert diagnostics["best_gold_rank"] == 1
    assert diagnostics["worst_gold_rank"] == 1


def test_kg_evidence_falls_back_to_printed_section_id() -> None:
    result = kg_result(
        uid="Cardiomyopathies_2023::7.1.1.1",
        document="Cardiomyopathies_2023",
        section_id="7.1.1.1",
        title="Diagnostic criteria",
    )

    evidence = kg_results_to_evidence([result])

    assert evidence[0].covered_section_ids == frozenset({"7.1.1.1"})


def test_metric_rows_respect_clean_and_end_to_end_sets() -> None:
    metrics = {
        "hit@1": 1.0,
        "precision@1": 1.0,
        "recall@1": 1.0,
        "complete_recall@1": 1.0,
        "reciprocal_rank@1": 1.0,
        "hit@3": 1.0,
        "precision@3": 1 / 3,
        "recall@3": 1.0,
        "complete_recall@3": 1.0,
        "reciprocal_rank@3": 1.0,
    }
    records = [
        {
            "question_id": "CM_01",
            "mode": "mentions_only",
            "status": "success",
            "latency_ms": 10.0,
            "returned_count": 1,
            "metrics": {"section": metrics, "document": metrics},
        },
        {
            "question_id": "SYN_02",
            "mode": "mentions_only",
            "status": "success",
            "latency_ms": 20.0,
            "returned_count": 1,
            "metrics": {"section": metrics, "document": metrics},
        },
    ]
    sets = {
        "section_clean": {"CM_01"},
        "section_end_to_end": {"CM_01", "SYN_02"},
        "document_clean": {"CM_01", "SYN_02"},
        "document_end_to_end": {"CM_01", "SYN_02"},
    }

    rows = build_metric_rows(records, sets)
    aggregates = aggregate_metric_rows(rows)

    section_clean = next(
        item
        for item in aggregates
        if item["level"] == "section" and item["view"] == "clean"
    )
    section_e2e = next(
        item
        for item in aggregates
        if item["level"] == "section" and item["view"] == "end_to_end"
    )
    assert section_clean["query_count"] == 1
    assert section_e2e["query_count"] == 2
    assert section_e2e["means"]["mrr@3"] == 1.0
    assert section_e2e["statistics"]["hit@1"]["population_std"] == 0.0


def test_load_coverage_sets_reads_current_coverage_artifact(tmp_path) -> None:
    artifact = {
        "coverage_summary": {
            "evaluation_sets": {
                "document_level_clean_question_ids": ["Q1", "Q2"],
                "document_level_end_to_end_question_ids": ["Q1", "Q2"],
                "section_level_clean_question_ids": ["Q1"],
                "section_level_end_to_end_question_ids": ["Q1", "Q2"],
            }
        }
    }
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    sets = load_coverage_evaluation_sets(path)

    assert sets["section_clean"] == {"Q1"}
    assert sets["section_end_to_end"] == {"Q1", "Q2"}
    assert sets["document_clean"] == {"Q1", "Q2"}


def test_mixed_ranking_types_are_rejected() -> None:
    gold = [key("Cardiomyopathies_2023", "7.1.1.1", "Diagnostic criteria")]
    result = kg_result(
        uid="Cardiomyopathies_2023::7.1.1.1",
        document="Cardiomyopathies_2023",
        section_id="7.1.1.1",
        title="Diagnostic criteria",
    )

    with pytest.raises(TypeError, match="section_ranking"):
        evaluate_rankings(
            gold_sections=gold,
            section_ranking=[gold[0], result],
        )
