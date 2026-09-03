from __future__ import annotations

from agentic_rag.kg.candidate_generators import KGCandidate
from agentic_rag.kg.connection_artifact_retrieval import _merge_semantic_connection_candidates, _rank_rrf_pool
from agentic_rag.kg.pipeline import _generate_candidates_with_seeds
from agentic_rag.kg.models import KGMatchDiagnostic, KGRetrievalScores, KGSectionResult, KGSeededMatchDiagnostic

def _candidate(uid, diagnostics, *, rank, source="mentions"):
    section=KGSectionResult(section_uid=uid,document_id="D",printed_section_id=uid,text="text",matched_concepts=[],matched_terms=[],score=1.0,score_type="weighted_match",scores=KGRetrievalScores(),match_diagnostics=list(diagnostics),rank=rank)
    return KGCandidate(section=section,source=source,source_rank=rank,direct=(source=="mentions"),seed_uid=uid if source=="mentions" else None,seed_rank=rank if source=="mentions" else None,graph_distance=0 if source=="mentions" else 2)

def _local(uid, rank, sim):
    return _candidate(uid,[KGSeededMatchDiagnostic(query_term="q",concept_name=f"local-{uid}",match_type="embedding",weight=1.0,seed_rank=rank,seeding_method="embedding",similarity=sim)],rank=rank)

def _graph(uid, rank, sim):
    return _candidate(uid,[KGMatchDiagnostic(query_term="q",concept_name=f"graph-{uid}",match_type="embedding",weight=1.0,evidence_source="ontology_bridge_artifact",lexical_weight=sim,seed_concept_name="seed",seed_cui="C1",target_cui=f"C-{uid}")],rank=rank,source="ontology_bridge_artifact")

def test_untruncated_semantic_union_preserves_both_channels():
    out=_merge_semantic_connection_candidates([_local("L1",1,.9),_local("L2",2,.8)],[_graph("G1",1,.95),_graph("G2",2,.7)],top_k=None,metadata={})
    assert {c.section_uid for c in out}=={"L1","L2","G1","G2"}

def test_rrf_rewards_cross_channel_overlap_without_dropping_unique_candidates():
    out=_rank_rrf_pool([_local("X",2,.8),_local("L",1,.9)],[_graph("X",2,.8),_graph("G",1,.9)],rrf_k=60,metadata={})
    assert {c.section_uid for c in out}=={"X","L","G"}
    assert out[0].section_uid=="X"
    assert out[0].metadata["pool_fusion_policy"]=="rrf"

def test_rrf_single_channel_order_is_deterministic():
    out=_rank_rrf_pool([_local("A",1,.9),_local("B",2,.8)],[],rrf_k=60,metadata={})
    assert [c.section_uid for c in out]==["A","B"]


def test_pipeline_passes_question_context_only_to_opt_in_generator():
    class ContextAwareGenerator:
        def __init__(self):
            self.question_context = None

        def generate_with_question_context(
            self,
            terms,
            *,
            top_k,
            question_context,
            require_all,
            document_ids,
        ):
            self.question_context = question_context
            return [], []

    generator = ContextAwareGenerator()
    candidates, seeds = _generate_candidates_with_seeds(
        generator,
        ["friedreich ataxia"],
        top_k=50,
        require_all=False,
        document_ids=[],
        question_context="A patient with gait ataxia and cardiomyopathy.",
    )
    assert candidates == []
    assert seeds == []
    assert generator.question_context == (
        "A patient with gait ataxia and cardiomyopathy."
    )
