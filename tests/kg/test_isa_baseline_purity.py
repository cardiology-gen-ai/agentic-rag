from __future__ import annotations

from agentic_rag.kg.candidate_generators import _evidence_rows_to_results


def _row(
    *,
    uid: str,
    query_term: str,
    lexical_weight: float,
    evidence_source: str = "direct",
    relation_type: str = "MENTIONS",
    target_cui: str | None = None,
) -> dict:
    return {
        "section_uid": uid,
        "document_id": "Cardiomyopathies_2023",
        "section_id": uid,
        "printed_section_id": uid,
        "title": f"Section {uid}",
        "text": "non-empty section text",
        "matched_concepts": [],
        "score": 0.0,
        "query_term": query_term,
        "concept_name": query_term,
        "matched_value": query_term,
        "match_type": "exact_name" if lexical_weight >= 3.0 else "partial",
        "lexical_weight": lexical_weight,
        "evidence_source": evidence_source,
        "relation_type": relation_type,
        "traversal_policy": (
            "hierarchy_artifact_forward"
            if relation_type == "UMLS_ISA_ARTIFACT"
            else None
        ),
        "review_needed": relation_type == "UMLS_ISA_ARTIFACT",
        "evidence_weight": 0.5 if relation_type == "UMLS_ISA_ARTIFACT" else 1.0,
        "seed_concept_name": query_term,
        "seed_cui": "CSEED",
        "target_cui": target_cui,
    }


def test_baseline_order_ignores_lexical_weight_for_concept_match_ties():
    # Pure mentions_only orders equal concept-match scores by Section uid.
    # The concept-graph path historically used weighted_match as a hidden
    # tie-break, which makes an ablation differ even without graph evidence.
    rows = [
        _row(uid="B", query_term="term", lexical_weight=3.0),
        _row(uid="A", query_term="term", lexical_weight=1.0),
    ]

    weighted_tiebreak = _evidence_rows_to_results(
        rows,
        terms=["term"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
    )
    baseline_order = _evidence_rows_to_results(
        rows,
        terms=["term"],
        require_all=False,
        top_k=10,
        ranking_mode="concept_match",
        preserve_concept_match_baseline_order=True,
    )

    assert [x.section_uid for x in weighted_tiebreak] == ["B", "A"]
    assert [x.section_uid for x in baseline_order] == ["A", "B"]
    assert [x.score for x in baseline_order] == [1.0, 1.0]


def test_isa_can_still_promote_candidate_by_supporting_an_extra_query_term():
    # Baseline-compatible tie ordering must not turn ISA into a no-op.
    # Section B gets support for a second distinct query term through ISA,
    # so its concept_match rises from 1 to 2 and it legitimately outranks A.
    rows = [
        _row(uid="A", query_term="term one", lexical_weight=3.0),
        _row(uid="B", query_term="term one", lexical_weight=3.0),
        _row(
            uid="B",
            query_term="term two",
            lexical_weight=3.0,
            evidence_source="umls_neighbor",
            relation_type="UMLS_ISA_ARTIFACT",
            target_cui="CTARGET",
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
