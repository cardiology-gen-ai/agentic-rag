from __future__ import annotations

from agentic_rag.agent.output import KGMentionsPlan
from agentic_rag.kg.candidate_generators import (
    KGCandidate,
    SeededMentionsCandidateGenerator,
)
from agentic_rag.kg.concept_seeders import ConceptSeed
from agentic_rag.kg.models import KGSectionResult, KGRetrievalScores
from agentic_rag.kg.pipeline import ModularKGRetrievalPipeline


class FakeRouter:
    def __init__(self, terms: list[str] | None = None) -> None:
        self.terms = terms or ["hypertrophic cardiomyopathy"]

    def route(self, question: str, *, config=None) -> KGMentionsPlan:
        return KGMentionsPlan(
            terms=self.terms,
            require_all=False,
        )


def make_result(
    uid: str = "Doc::1",
    *,
    rank: int = 1,
    title: str = "Hypertrophic cardiomyopathy",
) -> KGSectionResult:
    document_id, section_id = uid.split("::", 1)
    return KGSectionResult(
        section_uid=uid,
        document_id=document_id,
        section_id=section_id,
        printed_section_id=section_id,
        title=title,
        level=3,
        text=f"Text for {title}",
        matched_concepts=[title.casefold()],
        matched_terms=[title.casefold()],
        score=1.0,
        score_type="concept_match",
        scores=KGRetrievalScores(
            concept_match=1.0,
            weighted_match=1.0,
        ),
        rank=rank,
    )


def make_candidate(uid: str = "Doc::1") -> KGCandidate:
    result = make_result(uid)
    return KGCandidate(
        section=result,
        source="mentions",
        source_rank=1,
        direct=True,
        seed_uid=result.section_uid,
        seed_rank=1,
        graph_distance=0,
    )


class FakeSeedTools:
    def __init__(self, result: KGSectionResult) -> None:
        self.result = result

    def search_sections_by_concept_seeds(self, seeds, **kwargs):
        return [self.result]


class FakeSeeder:
    name = "fake_seeder"
    concepts_per_term = 1

    def seed_concepts(self, terms, *, document_ids=None):
        return {
            term: [
                ConceptSeed(
                    query_term=term,
                    concept_name="hypertrophic cardiomyopathy",
                    canonical_type="disease",
                    umls_cui="C0000001",
                    method="lexical",
                    match_type="exact_name",
                    seed_rank=1,
                    matched_value="hypertrophic cardiomyopathy",
                )
            ]
            for term in terms
        }


class StaticGenerator:
    name = "static"
    ranking_mode = "concept_match"

    def __init__(self, candidates: list[KGCandidate]) -> None:
        self.candidates = list(candidates)

    def generate(
        self,
        terms,
        *,
        top_k,
        require_all=False,
        document_ids=None,
    ):
        return list(self.candidates)


class FailingGenerator:
    name = "failing_generator"
    ranking_mode = "concept_match"

    def generate(
        self,
        terms,
        *,
        top_k,
        require_all=False,
        document_ids=None,
    ):
        raise RuntimeError("candidate generation failed")


class NoOpExpander:
    name = "none"

    def expand(self, candidates):
        return list(candidates)


class FailingExpander:
    name = "failing_expander"

    def expand(self, candidates):
        raise RuntimeError("expansion failed")


class NoOpReranker:
    name = "none"

    def rerank(self, candidates, *, top_k):
        return list(candidates)[:top_k]


class FailingReranker:
    name = "failing_reranker"

    def rerank(self, candidates, *, top_k):
        raise RuntimeError("reranking failed")


def test_seeded_generator_has_no_request_state() -> None:
    generator = SeededMentionsCandidateGenerator(
        FakeSeedTools(make_result()),
        FakeSeeder(),
    )

    candidates, seeds = generator.generate_with_seeds(
        ["HCM"],
        top_k=10,
    )

    assert len(candidates) == 1
    assert len(seeds) == 1
    assert seeds[0].query_term == "HCM"
    assert not hasattr(generator, "last_concept_seeds")

    public_candidates = generator.generate(["HCM"], top_k=10)
    assert [item.section_uid for item in public_candidates] == ["Doc::1"]


def test_candidate_generation_failure_is_stage_specific() -> None:
    pipeline = ModularKGRetrievalPipeline(
        router=FakeRouter(),
        candidate_generator=FailingGenerator(),
        expander=NoOpExpander(),
        reranker=NoOpReranker(),
        mode="mentions_only",
    )

    run = pipeline.retrieve("Question")

    assert run.status == "execution_error"
    assert run.failed_stage == "candidate_generation"
    assert run.raw_candidates == []
    assert run.expanded_candidates == []
    assert run.results == []
    assert "candidate generation failed" in (run.error or "")


def test_expansion_failure_preserves_raw_candidates() -> None:
    candidate = make_candidate()
    pipeline = ModularKGRetrievalPipeline(
        router=FakeRouter(),
        candidate_generator=StaticGenerator([candidate]),
        expander=FailingExpander(),
        reranker=NoOpReranker(),
        mode="mentions_descendants",
    )

    run = pipeline.retrieve("Question")

    assert run.status == "execution_error"
    assert run.failed_stage == "expansion"
    assert [item.section_uid for item in run.raw_candidates] == ["Doc::1"]
    assert run.expanded_candidates == []
    assert run.results == []
    assert "expansion failed" in (run.error or "")


def test_reranking_failure_preserves_raw_and_expanded_candidates() -> None:
    candidate = make_candidate()
    pipeline = ModularKGRetrievalPipeline(
        router=FakeRouter(),
        candidate_generator=StaticGenerator([candidate]),
        expander=NoOpExpander(),
        reranker=FailingReranker(),
        mode="mentions_descendants",
    )

    run = pipeline.retrieve("Question")

    assert run.status == "execution_error"
    assert run.failed_stage == "reranking"
    assert [item.section_uid for item in run.raw_candidates] == ["Doc::1"]
    assert [item.section_uid for item in run.expanded_candidates] == ["Doc::1"]
    assert run.results == []
    assert "reranking failed" in (run.error or "")
