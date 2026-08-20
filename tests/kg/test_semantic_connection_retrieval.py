from __future__ import annotations

from agentic_rag.kg.candidate_generators import KGCandidate
from agentic_rag.kg.connection_artifact_retrieval import (
    _merge_semantic_connection_candidates,
    _seed_rows_from_concept_seeds,
)
from agentic_rag.kg.concept_seeders import ConceptSeed
from agentic_rag.kg.models import (
    KGMatchDiagnostic,
    KGRetrievalScores,
    KGSectionResult,
    KGSeededMatchDiagnostic,
)


def _candidate(uid: str, diagnostics, *, rank: int = 1, source="mentions"):
    section = KGSectionResult(
        section_uid=uid,
        document_id="D",
        printed_section_id=uid,
        text="text",
        matched_concepts=[],
        matched_terms=[],
        score=1.0,
        score_type="weighted_match",
        scores=KGRetrievalScores(),
        match_diagnostics=list(diagnostics),
        rank=rank,
    )
    return KGCandidate(
        section=section,
        source=source,
        source_rank=rank,
        direct=(source == "mentions"),
        seed_uid=uid if source == "mentions" else None,
        seed_rank=rank if source == "mentions" else None,
        graph_distance=0 if source == "mentions" else 2,
    )


def test_seed_rows_resolve_cui_after_semantic_selection():
    seeds = [
        ConceptSeed(
            query_term="naxos disease",
            concept_name="naxos disease",
            umls_cui=None,
            method="embedding",
            match_type="embedding",
            seed_rank=1,
            similarity=0.91,
        ),
        ConceptSeed(
            query_term="syncope",
            concept_name="syncope",
            umls_cui=None,
            method="embedding",
            match_type="embedding",
            seed_rank=1,
            similarity=0.88,
        ),
    ]
    rows = _seed_rows_from_concept_seeds(
        seeds,
        cui_by_concept_name={
            "naxos disease": "C123",
            "syncope": None,
        },
    )
    assert rows == [
        {
            "query_term": "naxos disease",
            "seed_cui": "C123",
            "seed_concept_name": "naxos disease",
            "match_type": "embedding",
            "matched_value": "naxos disease",
            "lexical_weight": 0.91,
        }
    ]


def test_graph_path_does_not_stack_multiple_votes_for_same_term():
    local = _candidate(
        "S",
        [
            KGSeededMatchDiagnostic(
                query_term="a",
                concept_name="local-a",
                match_type="embedding",
                weight=1.0,
                seed_rank=1,
                seeding_method="embedding",
                similarity=0.8,
            )
        ],
    )
    graph = _candidate(
        "S",
        [
            KGMatchDiagnostic(
                query_term="a",
                concept_name="neighbor-a",
                match_type="embedding",
                weight=1.0,
                evidence_source="direct_local_artifact",
                lexical_weight=0.7,
                seed_concept_name="local-a",
                seed_cui="C1",
                target_cui="C2",
            ),
            KGMatchDiagnostic(
                query_term="a",
                concept_name="neighbor-a2",
                match_type="embedding",
                weight=1.0,
                evidence_source="ontology_bridge_artifact",
                lexical_weight=0.75,
                seed_concept_name="local-a",
                seed_cui="C1",
                target_cui="C3",
            ),
        ],
        source="direct_local_artifact",
    )
    out = _merge_semantic_connection_candidates(
        [local], [graph], top_k=10, metadata={}
    )
    assert len(out) == 1
    # Strongest evidence for term a is the local 0.8, not 0.8+0.7+0.75.
    assert abs(out[0].section.score - 0.8) < 1e-9


def test_graph_only_candidate_can_enter_ranking_by_seed_similarity():
    local = _candidate(
        "L",
        [
            KGSeededMatchDiagnostic(
                query_term="a",
                concept_name="local-a",
                match_type="embedding",
                weight=1.0,
                seed_rank=1,
                seeding_method="embedding",
                similarity=0.6,
            )
        ],
        rank=1,
    )
    graph = _candidate(
        "G",
        [
            KGMatchDiagnostic(
                query_term="a",
                concept_name="neighbor",
                match_type="embedding",
                weight=1.0,
                evidence_source="ontology_bridge_artifact",
                lexical_weight=0.9,
                seed_concept_name="seed",
                seed_cui="C1",
                target_cui="C2",
            )
        ],
        source="ontology_bridge_artifact",
    )
    out = _merge_semantic_connection_candidates(
        [local], [graph], top_k=10, metadata={}
    )
    assert [c.section_uid for c in out] == ["G", "L"]
    assert out[0].direct is False
    assert out[1].direct is True
