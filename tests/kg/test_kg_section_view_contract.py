from __future__ import annotations

from agentic_rag.kg import candidate_generators as cg
from agentic_rag.kg import expanders
from agentic_rag.kg import tools
from agentic_rag.kg.candidate_generators import KGCandidate
from agentic_rag.kg.models import KGSectionResult
from agentic_rag.agent.output import KGRetrievalPlan
from agentic_rag.kg.rerankers import SeedRoundRobinReranker
from agentic_rag.kg.retriever import KGParameterizedRetriever


def _result(uid: str, *, rank: int = 1) -> KGSectionResult:
    doc_id, section_id = uid.split("::", 1)
    return KGSectionResult(
        section_uid=uid,
        document_id=doc_id,
        section_id=section_id,
        printed_section_id=section_id,
        title=f"Section {section_id}",
        text="Evidence text",
        retrieval_unit_id=f"{doc_id}::retrieval::{section_id}",
        section_view_schema_version="1",
        section_view_role="retrieval",
        retrieval_strategy="section_view",
        aggregation_mode="none",
        is_aggregated=False,
        content_owner_section_id=section_id,
        source_section_ids=[section_id],
        source_chunk_ids=[f"{doc_id}::{section_id}"],
        represented_section_ids=[section_id],
        matched_concepts=["concept"],
        matched_terms=["concept"],
        score=1.0,
        score_type="concept_match",
        scores={"concept_match": 1.0, "weighted_match": 1.0},
        rank=rank,
    )


def test_all_section_queries_enforce_retrieval_view_contract() -> None:
    queries = [
        tools._SEARCH_SECTIONS_BY_CONCEPTS,
        tools._SEARCH_SECTIONS_BY_TITLE,
        tools._SEARCH_SECTIONS_BY_CONCEPT_SEEDS,
        expanders._FIND_DESCENDANTS,
        cg._DIRECT_CONCEPT_GRAPH_EVIDENCE,
        cg._SAME_AS_CONCEPT_GRAPH_EVIDENCE,
        cg._UMLS_SEED_CONCEPT_GRAPH_EVIDENCE,
        cg._UMLS_SAME_AS_CONCEPT_GRAPH_EVIDENCE,
    ]

    for query in queries:
        assert "section_view_role = 'retrieval'" in query
        assert "coalesce(s.embed, false) = true" in query
        assert "coalesce(s.excluded, false) = false" in query
        assert "trim(coalesce(s.text, '')) <> ''" in query


def test_concept_catalogue_uses_only_eligible_retrieval_sections() -> None:
    query = tools._LIST_CONCEPT_CATALOGUE
    assert "s.section_view_role = 'retrieval'" in query
    assert "coalesce(s.embed, false) = true" in query
    assert "coalesce(s.excluded, false) = false" in query


def test_umls_queries_fail_closed_to_safe_materialized_edges() -> None:
    for query in (
        cg._UMLS_SEED_CONCEPT_GRAPH_EVIDENCE,
        cg._UMLS_SAME_AS_CONCEPT_GRAPH_EVIDENCE,
    ):
        assert "r.provenance" in query
        assert "'umls_connections'" in query
        assert "r.materialization_mode" in query
        assert "'safe_only'" in query
        assert "r.materialization_decision" in query
        assert "r.compatibility_status" in query
        assert "'compatible'" in query
        assert "r.local_type_compatible" in query
        assert "coalesce(r.review_needed, true) = false" in query
        assert "r.traversal_policy" in query


def test_same_as_queries_require_auto_same_cui_edges() -> None:
    for query in (
        cg._SAME_AS_CONCEPT_GRAPH_EVIDENCE,
        cg._UMLS_SAME_AS_CONCEPT_GRAPH_EVIDENCE,
    ):
        assert "same_as_rel.method" in query
        assert "'umls_cui'" in query
        assert "same_as_rel.status" in query
        assert "'auto'" in query
        assert "same_as_rel.score" in query


def test_concept_match_score_is_only_distinct_term_coverage() -> None:
    diagnostics = [
        {
            "query_term": "heart failure",
            "lexical_weight": 3.0,
            "weight": 1.0,
        },
        {
            "query_term": "heart failure",
            "lexical_weight": 1.0,
            "weight": 0.5,
        },
        {
            "query_term": "atrial fibrillation",
            "lexical_weight": 2.0,
            "weight": 0.9,
        },
    ]
    scores = cg._score_concept_graph_diagnostics(diagnostics)
    assert scores["concept_match"] == 2.0
    assert scores["weighted_match"] > 0.0


def test_section_result_from_record_preserves_retrieval_provenance() -> None:
    row = {
        "section_uid": "Doc::1",
        "document_id": "Doc",
        "section_id": "1",
        "printed_section_id": "1",
        "title": "Section 1",
        "level": 2,
        "text": "Evidence text",
        "page_start": 1,
        "page_end": 1,
        "part_index": 0,
        "part_count": 1,
        "retrieval_unit_id": "Doc::retrieval::1",
        "section_view_schema_version": "1",
        "section_view_role": "retrieval",
        "retrieval_strategy": "section_view",
        "aggregation_mode": "none",
        "is_aggregated": False,
        "content_owner_section_id": "1",
        "source_section_ids": ["1"],
        "source_chunk_ids": ["Doc::1"],
        "represented_section_ids": ["1"],
        "structural_context_section_ids": [],
        "absorbed_section_ids": [],
        "absorbed_source_section_ids": [],
        "matched_concepts": [],
        "matched_terms": [],
        "score": 1.0,
        "score_type": "concept_match",
        "scores": {"concept_match": 1.0, "weighted_match": 1.0},
        "match_diagnostics": [],
    }
    result = KGSectionResult.from_record(row)

    assert result.retrieval_unit_id == "Doc::retrieval::1"
    assert result.section_view_schema_version == "1"
    assert result.section_view_role == "retrieval"
    assert result.source_section_ids == ["1"]
    assert result.represented_section_ids == ["1"]


def test_review_needed_umls_expansion_is_rejected() -> None:
    class FakeClient:
        def run_read(self, query, parameters=None):
            return []

    try:
        cg.ConceptGraphExpansionCandidateGenerator(
            FakeClient(),
            include_same_as=True,
            umls_policies=["safe"],
            include_review_needed=True,
        )
    except ValueError as exc:
        assert "Review-needed UMLS relations are not allowed" in str(exc)
    else:
        raise AssertionError("Expected fail-closed UMLS policy validation")


def test_advanced_retriever_propagates_document_scope() -> None:
    class FakeRouter:
        def route(self, question: str, *, config=None):
            return KGRetrievalPlan.model_validate(
                {
                    "intent": "diagnosis",
                    "expected_scope": "single_section",
                    "combination_mode": "direct",
                    "calls": [
                        {
                            "tool": "search_sections_by_concepts",
                            "role": "anchor",
                            "terms": ["heart failure"],
                            "require_all": False,
                        }
                    ],
                }
            )

    class FakeTools:
        def __init__(self):
            self.calls = []

        def search_sections_by_concepts(self, concepts, **kwargs):
            self.calls.append(kwargs)
            return [_result("Doc::1")]

        def search_sections_by_title(self, title_terms, **kwargs):
            raise AssertionError("Title retrieval is not expected")

        def find_hierarchical_context_matches(
            self,
            anchor_uids,
            context_uids,
            *,
            max_depth=6,
        ):
            return []

    tools_instance = FakeTools()
    run = KGParameterizedRetriever(
        FakeRouter(),
        tools_instance,
        document_ids=[" Doc ", "doc", ""],
    ).retrieve("Question")

    assert run.status == "success"
    assert run.document_filtering == ["Doc"]
    assert tools_instance.calls[0]["document_ids"] == ["Doc"]


def test_seed_round_robin_never_returns_more_than_top_k() -> None:
    seed_1 = KGCandidate(
        section=_result("Doc::1", rank=1),
        source="mentions",
        source_rank=1,
        direct=True,
        seed_uid="Doc::1",
        seed_rank=1,
        graph_distance=0,
    )
    descendant_1 = KGCandidate(
        section=_result("Doc::1.1", rank=2),
        source="descendant",
        source_rank=1,
        direct=False,
        seed_uid="Doc::1",
        seed_rank=1,
        graph_distance=1,
    )
    seed_2 = KGCandidate(
        section=_result("Doc::2", rank=3),
        source="mentions",
        source_rank=2,
        direct=True,
        seed_uid="Doc::2",
        seed_rank=2,
        graph_distance=0,
    )
    descendant_2 = KGCandidate(
        section=_result("Doc::2.1", rank=4),
        source="descendant",
        source_rank=2,
        direct=False,
        seed_uid="Doc::2",
        seed_rank=2,
        graph_distance=1,
    )

    results = SeedRoundRobinReranker(
        descendants_per_seed=1,
    ).rerank(
        [seed_1, descendant_1, seed_2, descendant_2],
        top_k=3,
    )

    assert len(results) == 3
    assert [item.final_rank for item in results] == [1, 2, 3]
