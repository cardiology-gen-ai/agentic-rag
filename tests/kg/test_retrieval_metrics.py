from __future__ import annotations

import pytest

from agentic_rag.evaluation.evidence import EvidenceSection, RetrievedEvidence
from agentic_rag.evaluation.metrics import (
    compute_coverage_metrics,
    coverage_at_cutoff,
)


def section(section_id: str, document_id: str = "doc") -> EvidenceSection:
    return EvidenceSection(document_id=document_id, section_id=section_id)


def evidence(
    retrieval_unit_id: str,
    covered: set[str],
    *,
    rank: int,
    document_id: str = "doc",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        document_id=document_id,
        retrieval_unit_id=retrieval_unit_id,
        covered_sections=frozenset(
            section(section_id, document_id) for section_id in covered
        ),
        raw_rank=rank,
        source_record_ids=(retrieval_unit_id,),
        source_type="test",
    )


def test_known_coverage_metrics() -> None:
    gold = {section("A"), section("D")}
    ranking = [
        evidence("u1", {"A"}, rank=1),
        evidence("u2", {"B"}, rank=2),
        evidence("u3", {"C"}, rank=3),
        evidence("u4", {"D"}, rank=4),
    ]

    metrics = compute_coverage_metrics(ranking, gold, cutoffs=(1, 3, 4))

    assert metrics.at(3).hit == 1.0
    assert metrics.at(3).precision == pytest.approx(1 / 3)
    assert metrics.at(3).recall == 0.5
    assert metrics.at(3).complete_recall == 0.0
    assert metrics.at(4).complete_recall == 1.0
    assert metrics.at(4).reciprocal_rank == 1.0


def test_reciprocal_rank_uses_first_relevant_unit() -> None:
    gold = {section("C")}
    ranking = [
        evidence("u1", {"A"}, rank=1),
        evidence("u2", {"B"}, rank=2),
        evidence("u3", {"C"}, rank=3),
    ]

    metrics = compute_coverage_metrics(ranking, gold, cutoffs=(2, 3))

    assert metrics.at(2).reciprocal_rank == 0.0
    assert metrics.at(3).reciprocal_rank == pytest.approx(1 / 3)
    assert metrics.first_relevant_rank == 3


def test_one_hierarchical_unit_can_cover_multiple_gold_sections() -> None:
    gold = {section("1.1"), section("1.2"), section("1.3")}
    ranking = [
        evidence("parent", {"1", "1.1", "1.2", "1.3"}, rank=1)
    ]

    metrics = compute_coverage_metrics(ranking, gold, cutoffs=(1, 3))

    assert metrics.at(1).precision == 1.0
    assert metrics.at(1).recall == 1.0
    assert metrics.at(1).complete_recall == 1.0
    assert metrics.at(1).relevant_unit_count == 1
    assert metrics.at(1).covered_gold_count == 3


def test_precision_counts_retrieval_units_not_gold_sections() -> None:
    gold = {section("1.1"), section("1.2"), section("2")}
    ranking = [
        evidence("parent", {"1.1", "1.2"}, rank=1),
        evidence("irrelevant", {"9"}, rank=2),
    ]

    metrics = compute_coverage_metrics(ranking, gold, cutoffs=(2,))

    assert metrics.at(2).relevant_unit_count == 1
    assert metrics.at(2).precision == 0.5
    assert metrics.at(2).recall == pytest.approx(2 / 3)


def test_no_relevant_results_returns_zero_metrics() -> None:
    gold = {section("X")}
    ranking = [evidence("u1", {"A"}, rank=1)]

    metrics = compute_coverage_metrics(ranking, gold, cutoffs=(1, 3))

    assert metrics.at(3).hit == 0.0
    assert metrics.at(3).precision == 0.0
    assert metrics.at(3).recall == 0.0
    assert metrics.at(3).reciprocal_rank == 0.0
    assert metrics.at(3).complete_recall == 0.0


def test_precision_denominator_is_requested_k() -> None:
    metrics = compute_coverage_metrics(
        [evidence("u1", {"A"}, rank=1)],
        {section("A")},
        cutoffs=(3,),
    )

    assert metrics.at(3).precision == pytest.approx(1 / 3)


def test_coverage_at_cutoff_returns_found_and_missing_gold() -> None:
    gold = {section("A"), section("B"), section("C")}
    ranking = [
        evidence("u1", {"A", "B"}, rank=1),
        evidence("u2", {"D"}, rank=2),
    ]

    found, missing = coverage_at_cutoff(ranking, gold, k=2)

    assert found == frozenset({section("A"), section("B")})
    assert missing == frozenset({section("C")})


def test_empty_gold_and_invalid_cutoffs_raise() -> None:
    ranking = [evidence("u1", {"A"}, rank=1)]

    with pytest.raises(ValueError, match="gold_sections"):
        compute_coverage_metrics(ranking, set(), cutoffs=(1,))

    with pytest.raises(ValueError, match="cutoffs"):
        compute_coverage_metrics(ranking, {section("A")}, cutoffs=())

    with pytest.raises(ValueError, match=">= 1"):
        compute_coverage_metrics(ranking, {section("A")}, cutoffs=(0,))

def test_same_section_id_in_wrong_document_is_not_relevant() -> None:
    """Section identity must be scoped by document_id in multi-document retrieval."""
    gold = {section("7.2", document_id="doc-A")}
    ranking = [
        evidence(
            "wrong-doc-unit",
            {"7.2"},
            rank=1,
            document_id="doc-B",
        )
    ]

    metrics = compute_coverage_metrics(ranking, gold, cutoffs=(1,))

    assert metrics.at(1).hit == 0.0
    assert metrics.at(1).precision == 0.0
    assert metrics.at(1).recall == 0.0
    assert metrics.at(1).complete_recall == 0.0
    assert metrics.at(1).reciprocal_rank == 0.0
    assert metrics.first_relevant_rank is None


def test_cross_document_gold_requires_evidence_from_both_documents() -> None:
    """Complete recall must require every document-scoped gold section."""
    gold = {
        section("7.2", document_id="doc-A"),
        section("4.1", document_id="doc-B"),
    }
    ranking = [
        evidence(
            "doc-a-unit",
            {"7.2"},
            rank=1,
            document_id="doc-A",
        ),
        evidence(
            "doc-b-unit",
            {"4.1"},
            rank=2,
            document_id="doc-B",
        ),
    ]

    metrics = compute_coverage_metrics(ranking, gold, cutoffs=(1, 2))

    assert metrics.at(1).hit == 1.0
    assert metrics.at(1).recall == 0.5
    assert metrics.at(1).complete_recall == 0.0

    assert metrics.at(2).hit == 1.0
    assert metrics.at(2).recall == 1.0
    assert metrics.at(2).complete_recall == 1.0
