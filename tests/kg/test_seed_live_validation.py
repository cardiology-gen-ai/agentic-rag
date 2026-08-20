from __future__ import annotations

from types import SimpleNamespace

from agentic_rag.kg.candidate_generators import (
    HybridBestChannelCandidateGenerator,
    KGCandidate,
    SimilarityWeightedSeededMentionsCandidateGenerator,
)
from agentic_rag.kg.models import (
    KGMatchDiagnostic,
    KGRetrievalScores,
    KGSectionResult,
    KGSeededMatchDiagnostic,
)


def _section(uid, diagnostics, rank=1):
    return KGSectionResult(
        section_uid=uid,
        document_id="D",
        printed_section_id=uid,
        text="text",
        matched_concepts=[],
        matched_terms=[],
        score=1.0,
        score_type="concept_match",
        scores=KGRetrievalScores(concept_match=1.0, weighted_match=1.0),
        match_diagnostics=diagnostics,
        rank=rank,
    )


def _candidate(uid, diagnostics, rank=1):
    return KGCandidate(
        section=_section(uid, diagnostics, rank=rank),
        source="mentions",
        source_rank=rank,
        direct=True,
        seed_uid=uid,
        seed_rank=rank,
        graph_distance=0,
    )


class _SemanticBase:
    def __init__(self, candidates):
        self._candidates = candidates

    def generate_with_seeds(self, *args, **kwargs):
        return list(self._candidates), []


class _LexicalBase:
    ranking_mode = "concept_match"

    def __init__(self, candidates):
        self._candidates = candidates

    def generate(self, *args, **kwargs):
        return list(self._candidates)


def test_similarity_weighted_prefers_high_cosine_sum():
    low = _candidate(
        "low",
        [
            KGSeededMatchDiagnostic(
                query_term="a",
                concept_name="x",
                match_type="seeded_concept",
                weight=1.0,
                seed_rank=1,
                seeding_method="embedding",
                similarity=0.51,
            )
        ],
        rank=1,
    )
    high = _candidate(
        "high",
        [
            KGSeededMatchDiagnostic(
                query_term="a",
                concept_name="y",
                match_type="seeded_concept",
                weight=1.0,
                seed_rank=2,
                seeding_method="embedding",
                similarity=0.91,
            )
        ],
        rank=2,
    )
    gen = SimilarityWeightedSeededMentionsCandidateGenerator(
        _SemanticBase([low, high])
    )
    result = gen.generate(["a"], top_k=2)
    assert [c.section_uid for c in result] == ["high", "low"]
    assert result[0].section.score == 0.91


def test_hybrid_uses_best_channel_per_term():
    lexical = _candidate(
        "S",
        [
            KGMatchDiagnostic(
                query_term="a",
                concept_name="lex",
                match_type="exact_name",
                weight=1.0,
            )
        ],
    )
    semantic = _candidate(
        "S",
        [
            KGSeededMatchDiagnostic(
                query_term="a",
                concept_name="sem-a",
                match_type="seeded_concept",
                weight=1.0,
                seed_rank=1,
                seeding_method="embedding",
                similarity=0.8,
            ),
            KGSeededMatchDiagnostic(
                query_term="b",
                concept_name="sem-b",
                match_type="seeded_concept",
                weight=1.0,
                seed_rank=1,
                seeding_method="embedding",
                similarity=0.6,
            ),
        ],
    )
    gen = HybridBestChannelCandidateGenerator(
        _LexicalBase([lexical]),
        _SemanticBase([semantic]),
    )
    result = gen.generate(["a", "b"], top_k=5)
    assert len(result) == 1
    assert abs(result[0].section.score - 1.6) < 1e-9
    assert result[0].metadata["seed_channel_policy"] == "hybrid_best_channel"


def test_hybrid_union_can_add_semantic_only_section():
    lexical = _candidate(
        "L",
        [KGMatchDiagnostic(
            query_term="a",
            concept_name="lex",
            match_type="partial",
            weight=1.0,
        )],
    )
    semantic = _candidate(
        "S",
        [KGSeededMatchDiagnostic(
            query_term="b",
            concept_name="sem",
            match_type="seeded_concept",
            weight=1.0,
            seed_rank=1,
            seeding_method="embedding",
            similarity=0.9,
        )],
    )
    gen = HybridBestChannelCandidateGenerator(
        _LexicalBase([lexical]),
        _SemanticBase([semantic]),
    )
    result = gen.generate(["a", "b"], top_k=5)
    assert {c.section_uid for c in result} == {"L", "S"}
