from __future__ import annotations

from agentic_rag.evaluation.kg_batch import (
    aggregate_metric_rows,
    build_metric_rows,
    candidate_diagnostics,
    evaluate_rankings,
    gold_document_ids,
    gold_section_keys,
)
from agentic_rag.evaluation.retrieval import SectionKey


def key(document: str, section_id: str, title: str) -> SectionKey:
    return SectionKey(
        document_id=document.casefold(),
        printed_section_id=section_id,
        title=title.casefold(),
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
    assert evaluated["section"]["ndcg@10"] == 1.0
    assert evaluated["document"]["hit@1"] == 1.0


def test_metric_rows_respect_clean_and_end_to_end_sets() -> None:
    metrics = {
        "hit@1": 1.0,
        "precision@1": 1.0,
        "recall@1": 1.0,
        "complete_recall@1": 1.0,
        "hit@3": 1.0,
        "precision@3": 1 / 3,
        "recall@3": 1.0,
        "complete_recall@3": 1.0,
        "hit@5": 1.0,
        "precision@5": 0.2,
        "recall@5": 1.0,
        "complete_recall@5": 1.0,
        "hit@10": 1.0,
        "precision@10": 0.1,
        "recall@10": 1.0,
        "complete_recall@10": 1.0,
        "reciprocal_rank@10": 1.0,
        "ndcg@10": 1.0,
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
