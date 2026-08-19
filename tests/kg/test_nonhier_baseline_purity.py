from __future__ import annotations

from agentic_rag.kg.candidate_generators import (
    _candidate_source_from_diagnostics,
    _evidence_rows_to_results,
    _source_graph_distance,
)
from agentic_rag.kg.pipeline import _validate_mode


def _row(
    *,
    uid: str,
    query_term: str,
    lexical_weight: float,
    evidence_source: str = "direct",
    relation_type: str = "MENTIONS",
    target_cui: str | None = None,
    edge_id: str | None = None,
    expansion_mode: str | None = None,
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
        "match_type": "exact_name" if lexical_weight >= 3.0 else "partial",
        "lexical_weight": lexical_weight,
        "evidence_source": evidence_source,
        "relation_type": relation_type,
        "traversal_policy": (
            "nonhier_artifact_safe_forward"
            if evidence_source == "nonhier_artifact"
            else None
        ),
        "review_needed": False,
        "evidence_weight": 0.5 if evidence_source == "nonhier_artifact" else 1.0,
        "seed_concept_name": query_term,
        "seed_cui": "CSEED",
        "target_cui": target_cui,
        "artifact_edge_id": edge_id,
        "semantic_status": "valid" if edge_id else None,
        "expansion_mode": expansion_mode,
    }


def test_no_nonhier_evidence_preserves_mentions_baseline_tie_order():
    rows = [
        _row(uid="B", query_term="term", lexical_weight=3.0),
        _row(uid="A", query_term="term", lexical_weight=1.0),
    ]

    results = _evidence_rows_to_results(
        rows,
        terms=["term"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
    )

    assert [x.section_uid for x in results] == ["A", "B"]
    assert [x.score for x in results] == [1.0, 1.0]


def test_nonhier_can_promote_only_by_supporting_an_extra_query_term():
    rows = [
        _row(uid="A", query_term="term one", lexical_weight=3.0),
        _row(uid="B", query_term="term one", lexical_weight=3.0),
        _row(
            uid="B",
            query_term="term two",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_associated_morphology",
            target_cui="CTARGET",
            edge_id="NH52-002",
            expansion_mode="expand",
        ),
    ]

    results = _evidence_rows_to_results(
        rows,
        terms=["term one", "term two"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
    )

    assert [x.section_uid for x in results] == ["B", "A"]
    assert results[0].scores is not None
    assert results[0].scores.concept_match == 2.0
    assert results[1].scores is not None
    assert results[1].scores.concept_match == 1.0
    diagnostics = results[0].match_diagnostics
    graph_rows = [
        x for x in diagnostics
        if x.evidence_source == "nonhier_artifact"
    ]
    assert graph_rows[0].artifact_edge_id == "NH52-002"
    assert graph_rows[0].expansion_mode == "expand"


def test_v1_support_only_evidence_may_reinforce_existing_candidate():
    rows = [
        _row(uid="A", query_term="term one", lexical_weight=3.0),
        _row(
            uid="A",
            query_term="term two",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_focus",
            target_cui="CTARGET",
            edge_id="NH52-047",
            expansion_mode="support_only",
        ),
    ]

    results = _evidence_rows_to_results(
        rows,
        terms=["term one", "term two"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
    )
    assert len(results) == 1
    assert results[0].scores is not None
    assert results[0].scores.concept_match == 2.0


def test_strict_support_only_is_ranking_neutral_but_keeps_diagnostic():
    rows = [
        _row(uid="A", query_term="term one", lexical_weight=3.0),
        _row(
            uid="A",
            query_term="term two",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_focus",
            target_cui="CTARGET",
            edge_id="NH52-047",
            expansion_mode="support_only",
        ),
    ]

    results = _evidence_rows_to_results(
        rows,
        terms=["term one", "term two"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
        ranking_neutral_expansion_modes=frozenset({"support_only"}),
    )
    assert len(results) == 1
    assert results[0].scores is not None
    assert results[0].scores.concept_match == 1.0
    assert results[0].matched_terms == ["term one"]
    support_diags = [
        d for d in results[0].match_diagnostics
        if d.expansion_mode == "support_only"
    ]
    assert len(support_diags) == 1
    assert support_diags[0].artifact_edge_id == "NH52-047"


def test_strict_support_only_cannot_break_baseline_tie_order():
    rows = [
        _row(uid="A", query_term="term one", lexical_weight=3.0),
        _row(uid="B", query_term="term one", lexical_weight=3.0),
        _row(
            uid="B",
            query_term="term two",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_focus",
            target_cui="CTARGET",
            edge_id="NH52-047",
            expansion_mode="support_only",
        ),
    ]
    results = _evidence_rows_to_results(
        rows,
        terms=["term one", "term two"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
        ranking_neutral_expansion_modes=frozenset({"support_only"}),
    )
    assert [r.section_uid for r in results] == ["A", "B"]
    assert [r.score for r in results] == [1.0, 1.0]



def test_graph_only_nonhier_candidate_has_nonhier_source_and_distance():
    rows = [
        _row(
            uid="G",
            query_term="term",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_associated_morphology",
            target_cui="CTARGET",
            edge_id="NH52-048",
            expansion_mode="expand",
        )
    ]
    result = _evidence_rows_to_results(
        rows,
        terms=["term"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
    )[0]
    assert _candidate_source_from_diagnostics(result) == "nonhier_artifact"
    assert _source_graph_distance("nonhier_artifact") == 2



def test_strict_nonhier_modes_are_registered():
    assert _validate_mode(
        "mentions_nonhier_artifact_raw_strict"
    ) == "mentions_nonhier_artifact_raw_strict"
    assert _validate_mode(
        "mentions_nonhier_artifact_safe_strict"
    ) == "mentions_nonhier_artifact_safe_strict"



def test_direct_first_prefers_more_direct_facets_over_more_graph_facets():
    rows = [
        _row(uid="A", query_term="term one", lexical_weight=3.0),
        _row(uid="A", query_term="term two", lexical_weight=3.0),
        _row(uid="B", query_term="term one", lexical_weight=3.0),
        _row(
            uid="B",
            query_term="term two",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_associated_morphology",
            target_cui="C2",
            edge_id="NH52-005",
            expansion_mode="expand",
        ),
        _row(
            uid="B",
            query_term="term three",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_associated_morphology",
            target_cui="C2",
            edge_id="NH52-023",
            expansion_mode="expand",
        ),
    ]

    legacy = _evidence_rows_to_results(
        rows,
        terms=["term one", "term two", "term three"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
    )
    direct_first = _evidence_rows_to_results(
        rows,
        terms=["term one", "term two", "term three"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
        direct_first_graph_second=True,
    )

    assert [x.section_uid for x in legacy] == ["B", "A"]
    assert [x.section_uid for x in direct_first] == ["A", "B"]
    assert direct_first[0].scores is not None
    assert direct_first[0].scores.direct_concept_match == 2.0
    assert direct_first[0].scores.graph_only_concept_match == 0.0
    assert direct_first[1].scores is not None
    assert direct_first[1].scores.direct_concept_match == 1.0
    assert direct_first[1].scores.graph_only_concept_match == 2.0


def test_direct_first_uses_graph_facets_only_after_direct_tie():
    rows = [
        _row(uid="A", query_term="term one", lexical_weight=3.0),
        _row(uid="B", query_term="term one", lexical_weight=3.0),
        _row(
            uid="B",
            query_term="term two",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_associated_morphology",
            target_cui="C2",
            edge_id="NH52-048",
            expansion_mode="expand",
        ),
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
    assert results[0].scores.direct_concept_match == 1.0
    assert results[0].scores.graph_only_concept_match == 1.0


def test_direct_first_graph_only_candidate_stays_below_direct_candidate():
    rows = [
        _row(uid="D", query_term="term one", lexical_weight=3.0),
        _row(
            uid="G",
            query_term="term one",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_associated_morphology",
            target_cui="C2",
            edge_id="NH52-048",
            expansion_mode="expand",
        ),
        _row(
            uid="G",
            query_term="term two",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_associated_morphology",
            target_cui="C2",
            edge_id="NH52-023",
            expansion_mode="expand",
        ),
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
    assert [x.section_uid for x in results] == ["D", "G"]


def test_direct_first_safe_support_only_remains_fully_ranking_neutral():
    rows = [
        _row(uid="A", query_term="term one", lexical_weight=3.0),
        _row(uid="B", query_term="term one", lexical_weight=3.0),
        _row(
            uid="B",
            query_term="term two",
            lexical_weight=3.0,
            evidence_source="nonhier_artifact",
            relation_type="has_focus",
            target_cui="C2",
            edge_id="NH52-047",
            expansion_mode="support_only",
        ),
    ]
    results = _evidence_rows_to_results(
        rows,
        terms=["term one", "term two"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
        ranking_neutral_expansion_modes=frozenset({"support_only"}),
        direct_first_graph_second=True,
    )
    assert [x.section_uid for x in results] == ["A", "B"]
    assert results[1].scores.graph_only_concept_match == 0.0


def test_v3_nonhier_direct_first_modes_are_registered():
    assert _validate_mode(
        "mentions_nonhier_artifact_raw_strict_direct_first"
    ) == "mentions_nonhier_artifact_raw_strict_direct_first"
    assert _validate_mode(
        "mentions_nonhier_artifact_safe_strict_direct_first"
    ) == "mentions_nonhier_artifact_safe_strict_direct_first"
