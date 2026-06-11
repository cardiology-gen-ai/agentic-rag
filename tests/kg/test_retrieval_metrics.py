from __future__ import annotations

import math

import pytest

from agentic_rag.evaluation.metrics import (
    aggregate_query_metrics,
    complete_recall_at_k,
    compute_query_metrics,
    deduplicate_ranking,
    filter_metric_rows_by_question_ids,
    get_evaluation_question_ids,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from agentic_rag.evaluation.retrieval import SectionKey


def test_known_binary_relevance_metrics() -> None:
    gold = {"A", "D"}
    ranking = ["A", "B", "C", "D"]

    assert hit_at_k(gold, ranking, 3) == 1.0
    assert precision_at_k(gold, ranking, 3) == pytest.approx(1 / 3)
    assert recall_at_k(gold, ranking, 3) == 0.5
    assert reciprocal_rank_at_k(gold, ranking, 10) == 1.0
    assert complete_recall_at_k(gold, ranking, 3) == 0.0
    assert complete_recall_at_k(gold, ranking, 4) == 1.0


def test_reciprocal_rank_uses_first_relevant_position() -> None:
    assert reciprocal_rank_at_k({"C"}, ["A", "B", "C", "D"], 10) == pytest.approx(
        1 / 3
    )
    assert reciprocal_rank_at_k({"D"}, ["A", "B", "C", "D"], 3) == 0.0


def test_ndcg_matches_binary_relevance_definition() -> None:
    gold = {"A", "D"}
    ranking = ["A", "B", "C", "D"]

    dcg = 1.0 + 1.0 / math.log2(5)
    idcg = 1.0 + 1.0 / math.log2(3)

    assert ndcg_at_k(gold, ranking, 4) == pytest.approx(dcg / idcg)


def test_no_relevant_results_returns_zero_metrics() -> None:
    gold = {"X"}
    ranking = ["A", "B", "C"]

    assert hit_at_k(gold, ranking, 3) == 0.0
    assert precision_at_k(gold, ranking, 3) == 0.0
    assert recall_at_k(gold, ranking, 3) == 0.0
    assert reciprocal_rank_at_k(gold, ranking, 3) == 0.0
    assert ndcg_at_k(gold, ranking, 3) == 0.0
    assert complete_recall_at_k(gold, ranking, 3) == 0.0


def test_ranking_duplicates_are_collapsed_before_scoring() -> None:
    gold = {"A", "D"}
    ranking = ["A", "A", "B", "D"]

    assert deduplicate_ranking(ranking) == ["A", "B", "D"]
    assert precision_at_k(gold, ranking, 3) == pytest.approx(2 / 3)
    assert recall_at_k(gold, ranking, 3) == 1.0


def test_precision_denominator_is_requested_k() -> None:
    assert precision_at_k({"A"}, ["A"], 3) == pytest.approx(1 / 3)


def test_metrics_support_section_keys() -> None:
    a = SectionKey("doc", "1", "first")
    b = SectionKey("doc", "2", "second")
    c = SectionKey("doc", "3", "third")

    assert recall_at_k({a, c}, [a, b, c], 2) == 0.5
    assert recall_at_k({a, c}, [a, b, c], 3) == 1.0


def test_compute_query_metrics_returns_expected_flat_keys() -> None:
    metrics = compute_query_metrics(
        {"A", "D"},
        ["A", "B", "C", "D"],
        cutoffs=(1, 3),
        rank_cutoff=4,
    )

    assert set(metrics) == {
        "hit@1",
        "precision@1",
        "recall@1",
        "complete_recall@1",
        "hit@3",
        "precision@3",
        "recall@3",
        "complete_recall@3",
        "reciprocal_rank@4",
        "ndcg@4",
    }
    assert metrics["recall@3"] == 0.5


def test_aggregate_query_metrics_exposes_mrr_alias() -> None:
    aggregate = aggregate_query_metrics(
        [
            {"hit@1": 1.0, "reciprocal_rank@10": 1.0},
            {"hit@1": 0.0, "reciprocal_rank@10": 0.5},
        ]
    )

    assert aggregate["query_count"] == 2
    assert aggregate["means"]["hit@1"] == 0.5
    assert aggregate["means"]["reciprocal_rank@10"] == 0.75
    assert aggregate["means"]["mrr@10"] == 0.75
    assert aggregate["statistics"]["hit@1"]["population_std"] == 0.5


def test_get_evaluation_question_ids_reads_clean_and_end_to_end_sets() -> None:
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

    assert get_evaluation_question_ids(
        artifact,
        level="section",
        view="clean",
    ) == ["Q1"]
    assert get_evaluation_question_ids(
        artifact,
        level="section",
        view="end_to_end",
    ) == ["Q1", "Q2"]


def test_filter_metric_rows_preserves_evaluation_set_order() -> None:
    rows = [
        {"question_id": "Q2", "metrics": {"hit@1": 0.0}},
        {"question_id": "Q1", "metrics": {"hit@1": 1.0}},
    ]

    filtered = filter_metric_rows_by_question_ids(rows, ["Q1", "Q2"])
    assert [row["question_id"] for row in filtered] == ["Q1", "Q2"]


def test_empty_gold_and_invalid_cutoffs_raise() -> None:
    with pytest.raises(ValueError, match="gold"):
        hit_at_k(set(), ["A"], 1)

    with pytest.raises(ValueError, match="positive integer"):
        hit_at_k({"A"}, ["A"], 0)

    with pytest.raises(ValueError, match="At least one cutoff"):
        compute_query_metrics({"A"}, ["A"], cutoffs=())
