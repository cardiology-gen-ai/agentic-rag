from __future__ import annotations

from agentic_rag.kg.candidate_generators import _evidence_rows_to_results
from agentic_rag.kg.isa_artifact import (
    _isa_seed_query,
    _isa_seed_row_allowed,
)
from agentic_rag.kg.pipeline import _validate_mode


def _row(
    *,
    uid: str,
    query_term: str,
    evidence_source: str = "direct",
) -> dict:
    return {
        "section_uid": uid,
        "document_id": "Cardiomyopathies_2023",
        "section_id": uid,
        "printed_section_id": uid,
        "title": f"Section {uid}",
        "text": "non-empty section text",
        "query_term": query_term,
        "concept_name": query_term,
        "matched_value": query_term,
        "match_type": "exact_name",
        "lexical_weight": 3.0,
        "evidence_source": evidence_source,
        "relation_type": (
            "MENTIONS" if evidence_source == "direct" else "UMLS_ISA_ARTIFACT"
        ),
        "traversal_policy": (
            None if evidence_source == "direct" else "hierarchy_artifact_forward"
        ),
        "review_needed": evidence_source != "direct",
        "evidence_weight": 1.0 if evidence_source == "direct" else 0.5,
        "seed_concept_name": query_term,
        "seed_cui": "CSEED",
        "target_cui": "CTARGET" if evidence_source != "direct" else "CSEED",
    }


def test_isa_exact_seed_query_adds_exact_filter_only_in_strict_mode():
    permissive = _isa_seed_query("permissive")
    strict = _isa_seed_query("exact_name_only")
    needle = "AND match_type = 'exact_name'"
    assert needle not in permissive
    assert needle in strict


def test_isa_exact_seed_policy_rejects_prefix_and_partial_rows():
    assert _isa_seed_row_allowed(
        {"match_type": "exact_name"},
        seed_match_policy="exact_name_only",
    )
    assert not _isa_seed_row_allowed(
        {"match_type": "prefix"},
        seed_match_policy="exact_name_only",
    )
    assert not _isa_seed_row_allowed(
        {"match_type": "partial"},
        seed_match_policy="exact_name_only",
    )


def test_isa_direct_first_prevents_parent_evidence_from_overriding_more_direct_support():
    rows = [
        _row(uid="A", query_term="term one"),
        _row(uid="A", query_term="term two"),
        _row(uid="B", query_term="term one"),
        _row(uid="B", query_term="term two", evidence_source="umls_neighbor"),
        _row(uid="B", query_term="term three", evidence_source="umls_neighbor"),
    ]
    results = _evidence_rows_to_results(
        rows,
        terms=["term one", "term two", "term three"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
        direct_first_graph_second=True,
    )
    assert [x.section_uid for x in results] == ["A", "B"]
    assert results[0].scores.direct_concept_match == 2.0
    assert results[1].scores.direct_concept_match == 1.0
    assert results[1].scores.graph_only_concept_match == 2.0


def test_isa_graph_support_can_break_tie_when_direct_coverage_is_equal():
    rows = [
        _row(uid="A", query_term="term one"),
        _row(uid="B", query_term="term one"),
        _row(uid="B", query_term="term two", evidence_source="umls_neighbor"),
    ]
    results = _evidence_rows_to_results(
        rows,
        terms=["term one", "term two"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
        direct_first_graph_second=True,
    )
    assert [x.section_uid for x in results] == ["B", "A"]


def test_isa_strict_direct_first_mode_is_registered():
    assert _validate_mode(
        "mentions_isa_forward_artifact_strict_direct_first"
    ) == "mentions_isa_forward_artifact_strict_direct_first"
