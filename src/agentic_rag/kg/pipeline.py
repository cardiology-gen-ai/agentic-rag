"""Composable knowledge-graph retrieval pipelines.

The advanced role-aware retriever remains unchanged. This module provides
simple, independently measurable configurations for ablation studies.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_rag.agent.output import KGMentionsPlan
from agentic_rag.kg.candidate_generators import (
    CandidateGeneratorProtocol,
    ConceptGraphExpansionCandidateGenerator,
    KGCandidate,
    KGSectionSearchProtocol,
    MentionsCandidateGenerator,
    RescueConceptGraphExpansionCandidateGenerator,
    SeededMentionsCandidateGenerator,
    RankFusionSeededMentionsCandidateGenerator,
    SimilarityWeightedSeededMentionsCandidateGenerator,
    HybridBestChannelCandidateGenerator,
)
from agentic_rag.kg.connection_artifact_retrieval import (
    ConnectionArtifactCandidateGenerator,
    SemanticWeightedConnectionCandidateGenerator,
    PoolPreservingSemanticConnectionCandidateGenerator,
)
from agentic_rag.kg.concept_seeders import (
    ConceptSeed,
    ConceptSeederProtocol,
    EmbeddingConceptSeeder,
    ExactSafeAugmentedConceptSeeder,
    SameCUIAugmentedConceptSeeder,
    LexicalConceptSeeder,
)
from agentic_rag.kg.expanders import (
    CandidateExpanderProtocol,
    DescendantExpander,
    GraphReadProtocol,
    NoOpExpander,
)
from agentic_rag.kg.models import KGRankingMode
from agentic_rag.kg.isa_artifact import (
    ISAArtifactCandidateGenerator,
    ISASafeRerankCandidateGenerator,
)
from agentic_rag.kg.nonhier_artifact import NonHierArtifactCandidateGenerator
from agentic_rag.kg.query_normalization import (
    RouterTermNormalizationMode,
    normalize_mentions_plan,
)
from agentic_rag.kg.rerankers import (
    CandidateRerankerProtocol,
    NoOpReranker,
    SeedRoundRobinReranker,
)


ModularKGMode = Literal[
    "mentions_only",
    "mentions_lexical_seeded",
    "mentions_embedding_seeded",
    "mentions_embedding_seeded_similarity_weighted",
    "mentions_embedding_seeded_semantic_preselected_similarity_weighted",
    "mentions_embedding_seeded_preselection_rrf_similarity_weighted",
    "mentions_embedding_exact_safe_seeded_similarity_weighted",
    "mentions_embedding_exact_safe_same_cui_seeded_similarity_weighted",
    "mentions_lexical_semantic_best_channel",
    "mentions_descendants",
    "mentions_same_as",
    "mentions_umls_safe",
    "mentions_isa_forward_artifact",
    "mentions_isa_forward_artifact_strict_direct_first",
    "mentions_isa_semantic_safe_rerank",
    "mentions_nonhier_artifact_raw",
    "mentions_nonhier_artifact_safe",
    "mentions_nonhier_artifact_raw_strict",
    "mentions_nonhier_artifact_safe_strict",
    "mentions_nonhier_artifact_raw_strict_direct_first",
    "mentions_nonhier_artifact_safe_strict_direct_first",
    "mentions_nonhier_artifact_safe_strict_direct_first_frozen",
    "mentions_nonhier_artifact_safe_strict_direct_first_rescue",
    "mentions_same_as_rescue",
    "mentions_umls_safe_rescue",
    "mentions_direct_balanced",
    "mentions_bridge_sa_top5",
    "mentions_direct_bridge_sa_top3",
    "mentions_direct_bridge_sa_top5",
    "mentions_direct_bridge_sa_top10",
    "semantic_weighted_direct_balanced",
    "semantic_weighted_bridge_sa_top5",
    "semantic_weighted_direct_bridge_sa_top3",
    "semantic_weighted_direct_bridge_sa_top5",
    "semantic_weighted_direct_bridge_sa_top10",
    "semantic_weighted_pool_union_direct_bridge_sa_top3",
    "semantic_weighted_pool_rrf_direct_bridge_sa_top3",
]
ModularKGStatus = Literal[
    "success",
    "no_results",
    "router_error",
    "execution_error",
]
ModularKGFailureStage = Literal[
    "candidate_generation",
    "expansion",
    "reranking",
]


class KGMentionsRouterProtocol(Protocol):
    def route(
        self,
        question: str,
        *,
        config: Any | None = None,
    ) -> KGMentionsPlan: ...


class ModularKGRetrievalRun(BaseModel):
    """Complete trace for one modular KG retrieval request."""

    model_config = ConfigDict(extra="forbid")

    question: str
    mode: ModularKGMode
    status: ModularKGStatus
    plan: KGMentionsPlan | None = None

    raw_candidates: list[KGCandidate] = Field(default_factory=list)
    expanded_candidates: list[KGCandidate] = Field(default_factory=list)
    results: list[KGCandidate] = Field(default_factory=list)
    concept_seeds: list[ConceptSeed] = Field(default_factory=list)

    candidate_k: int = Field(ge=1)
    final_k: int = Field(ge=1)
    ranking_mode: KGRankingMode
    expander_name: str
    reranker_name: str
    retrieval_unit: Literal["section_node"] = "section_node"
    unit_scope: Literal["all_levels"] = "all_levels"
    document_filtering: list[str] | None = None
    router_term_normalization: RouterTermNormalizationMode = "none"

    latency_ms: float = Field(ge=0)
    error: str | None = None
    failed_stage: ModularKGFailureStage | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("question must be a non-empty string")
        return normalized

    @property
    def returned_count(self) -> int:
        return len(self.results)


class ModularKGRetrievalPipeline:
    """Route, generate, optionally expand, and rerank Section candidates."""

    def __init__(
        self,
        router: KGMentionsRouterProtocol,
        candidate_generator: CandidateGeneratorProtocol,
        expander: CandidateExpanderProtocol,
        reranker: CandidateRerankerProtocol,
        *,
        mode: ModularKGMode,
        candidate_k: int = 15,
        final_k: int = 10,
        document_ids: Sequence[str] | str | None = None,
        router_term_normalization: RouterTermNormalizationMode = "none",
    ) -> None:
        self.router = router
        self.candidate_generator = candidate_generator
        self.expander = expander
        self.reranker = reranker
        self.mode = _validate_mode(mode)
        self.candidate_k = _validate_top_k(candidate_k, "candidate_k")
        self.final_k = _validate_top_k(final_k, "final_k")
        self.document_ids = _normalize_optional_values(document_ids)
        if router_term_normalization not in {"none", "safe_v1"}:
            raise ValueError(
                "router_term_normalization must be one of: none, safe_v1"
            )
        self.router_term_normalization = router_term_normalization

    @property
    def ranking_mode(self) -> KGRankingMode:
        value = getattr(self.candidate_generator, "ranking_mode", None)
        if value not in {"concept_match", "weighted_match"}:
            raise RuntimeError(
                "Candidate generator does not expose a valid ranking_mode"
            )
        return value

    def retrieve(
        self,
        question: str,
        *,
        router_config: Any | None = None,
    ) -> ModularKGRetrievalRun:
        normalized_question = _validate_question(question)
        started = time.perf_counter()

        try:
            raw_plan = self.router.route(
                normalized_question,
                config=router_config,
            )
            plan = normalize_mentions_plan(
                raw_plan,
                mode=self.router_term_normalization,
            ).plan
        except Exception as exc:
            return self._run(
                question=normalized_question,
                status="router_error",
                plan=None,
                raw_candidates=[],
                expanded_candidates=[],
                results=[],
                concept_seeds=[],
                failed_stage=None,
                started=started,
                error=_format_exception(exc),
            )

        try:
            raw_candidates, concept_seeds = _generate_candidates_with_seeds(
                self.candidate_generator,
                plan.terms,
                top_k=self.candidate_k,
                require_all=plan.require_all,
                document_ids=self.document_ids,
            )
        except Exception as exc:
            return self._run(
                question=normalized_question,
                status="execution_error",
                plan=plan,
                raw_candidates=[],
                expanded_candidates=[],
                results=[],
                concept_seeds=[],
                failed_stage="candidate_generation",
                started=started,
                error=_format_exception(exc),
            )

        try:
            expanded_candidates = self.expander.expand(raw_candidates)
        except Exception as exc:
            return self._run(
                question=normalized_question,
                status="execution_error",
                plan=plan,
                raw_candidates=raw_candidates,
                expanded_candidates=[],
                results=[],
                concept_seeds=concept_seeds,
                failed_stage="expansion",
                started=started,
                error=_format_exception(exc),
            )

        try:
            results = self.reranker.rerank(
                expanded_candidates,
                top_k=self.final_k,
            )
        except Exception as exc:
            return self._run(
                question=normalized_question,
                status="execution_error",
                plan=plan,
                raw_candidates=raw_candidates,
                expanded_candidates=expanded_candidates,
                results=[],
                concept_seeds=concept_seeds,
                failed_stage="reranking",
                started=started,
                error=_format_exception(exc),
            )

        status: ModularKGStatus = "success" if results else "no_results"
        return self._run(
            question=normalized_question,
            status=status,
            plan=plan,
            raw_candidates=raw_candidates,
            expanded_candidates=expanded_candidates,
            results=results,
            concept_seeds=concept_seeds,
            failed_stage=None,
            started=started,
            error=None,
        )

    def retrieve_dict(
        self,
        question: str,
        *,
        router_config: Any | None = None,
    ) -> dict[str, Any]:
        return self.retrieve(
            question,
            router_config=router_config,
        ).model_dump(mode="json")

    def _run(
        self,
        *,
        question: str,
        status: ModularKGStatus,
        plan: KGMentionsPlan | None,
        raw_candidates: list[KGCandidate],
        expanded_candidates: list[KGCandidate],
        results: list[KGCandidate],
        concept_seeds: Sequence[ConceptSeed],
        failed_stage: ModularKGFailureStage | None,
        started: float,
        error: str | None,
    ) -> ModularKGRetrievalRun:
        return ModularKGRetrievalRun(
            question=question,
            mode=self.mode,
            status=status,
            plan=plan,
            raw_candidates=raw_candidates,
            expanded_candidates=expanded_candidates,
            results=results,
            concept_seeds=list(concept_seeds),
            candidate_k=self.candidate_k,
            final_k=self.final_k,
            ranking_mode=self.ranking_mode,
            expander_name=self.expander.name,
            reranker_name=self.reranker.name,
            document_filtering=(self.document_ids or None),
            router_term_normalization=self.router_term_normalization,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=error,
            failed_stage=failed_stage,
        )


def build_modular_kg_pipeline(
    mode: ModularKGMode,
    *,
    router: KGMentionsRouterProtocol,
    tools: KGSectionSearchProtocol,
    client: GraphReadProtocol | None = None,
    candidate_k: int = 15,
    final_k: int = 10,
    document_ids: Sequence[str] | str | None = None,
    exclude_summary_sections: bool = True,
    hierarchy_max_depth: int = 3,
    descendants_per_seed: int = 3,
    max_expanded_rows: int = 1000,
    concepts_per_term: int = 3,
    concept_embedding_model: str | None = None,
    concept_embedding_cache: str | None = None,
    query_term_embedding_cache: str | None = None,
    query_term_embedding_cache_read_only: bool = False,
    concept_embedding_min_similarity: float | None = None,
    concept_embedding_rank1_fallback: bool = False,
    concept_seeder: ConceptSeederProtocol | None = None,
    isa_connections_path: str | None = None,
    isa_max_depth: int = 1,
    nonhier_artifact_path: str | None = None,
    connection_config_path: str | None = None,
    graph_candidate_k: int | None = None,
    rrf_k: int = 60,
    router_term_normalization: RouterTermNormalizationMode = "none",
) -> ModularKGRetrievalPipeline:
    """Build one named ablation configuration.

    ``mentions_only`` is the pure local MENTIONS baseline. It matches only
    ``Concept.name`` before traversing MENTIONS. ``mentions_descendants``
    expands the pure MENTIONS seeds over
    HAS_CHILD and applies a deterministic seed-by-seed ordering.
    ``mentions_same_as`` and ``mentions_umls_safe`` keep the same MENTIONS plan
    terms but expand at the Concept/CUI level inside candidate generation.
    Their rescue variants preserve direct MENTIONS order and append only
    SAME_AS/UMLS-supported candidates not already present.
    ``mentions_lexical_seeded`` and ``mentions_embedding_seeded`` both first
    select explicit local Concepts, then use the same exact
    ``Concept.name -> MENTIONS -> Section`` candidate generator.
    """

    normalized_mode = _validate_mode(mode)

    if normalized_mode == "mentions_only":
        generator = MentionsCandidateGenerator(
            tools,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        expander: CandidateExpanderProtocol = NoOpExpander()
        reranker: CandidateRerankerProtocol = NoOpReranker()

    elif normalized_mode == "mentions_lexical_seeded":
        seeder = concept_seeder or LexicalConceptSeeder(
            tools,
            concepts_per_term=concepts_per_term,
        )
        generator = SeededMentionsCandidateGenerator(
            tools,
            seeder,
            exclude_summary_sections=exclude_summary_sections,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_embedding_seeded":
        seeder = concept_seeder
        if seeder is None:
            if concept_embedding_model is None:
                raise ValueError(
                    "concept_embedding_model is required for "
                    "mode='mentions_embedding_seeded'"
                )
            seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=concept_embedding_model,
                concepts_per_term=concepts_per_term,
                min_similarity=concept_embedding_min_similarity,
                keep_best_below_min_similarity=concept_embedding_rank1_fallback,
                cache_path=concept_embedding_cache,
                query_cache_path=query_term_embedding_cache,
                query_cache_read_only=query_term_embedding_cache_read_only,
            )
        generator = SeededMentionsCandidateGenerator(
            tools,
            seeder,
            exclude_summary_sections=exclude_summary_sections,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()


    elif normalized_mode == "mentions_embedding_seeded_similarity_weighted":
        seeder = concept_seeder
        if seeder is None:
            if concept_embedding_model is None:
                raise ValueError(
                    "concept_embedding_model is required for "
                    "mode='mentions_embedding_seeded_similarity_weighted'"
                )
            seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=concept_embedding_model,
                concepts_per_term=concepts_per_term,
                min_similarity=concept_embedding_min_similarity,
                keep_best_below_min_similarity=concept_embedding_rank1_fallback,
                cache_path=concept_embedding_cache,
                query_cache_path=query_term_embedding_cache,
                query_cache_read_only=query_term_embedding_cache_read_only,
            )
        semantic_base = SeededMentionsCandidateGenerator(
            tools,
            seeder,
            exclude_summary_sections=exclude_summary_sections,
        )
        generator = SimilarityWeightedSeededMentionsCandidateGenerator(
            semantic_base
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_embedding_seeded_preselection_rrf_similarity_weighted":
        seeder = concept_seeder
        if seeder is None:
            if concept_embedding_model is None:
                raise ValueError(
                    "concept_embedding_model is required for "
                    "mode='mentions_embedding_seeded_preselection_rrf_similarity_weighted'"
                )
            seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=concept_embedding_model,
                concepts_per_term=concepts_per_term,
                min_similarity=concept_embedding_min_similarity,
                keep_best_below_min_similarity=concept_embedding_rank1_fallback,
                cache_path=concept_embedding_cache,
                query_cache_path=query_term_embedding_cache,
                query_cache_read_only=query_term_embedding_cache_read_only,
            )
        fused_base = RankFusionSeededMentionsCandidateGenerator(
            tools,
            seeder,
            exclude_summary_sections=exclude_summary_sections,
            channel_k=100,
            rrf_constant=60,
        )
        generator = SimilarityWeightedSeededMentionsCandidateGenerator(fused_base)
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_embedding_seeded_semantic_preselected_similarity_weighted":
        seeder = concept_seeder
        if seeder is None:
            if concept_embedding_model is None:
                raise ValueError(
                    "concept_embedding_model is required for "
                    "mode='mentions_embedding_seeded_semantic_preselected_similarity_weighted'"
                )
            seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=concept_embedding_model,
                concepts_per_term=concepts_per_term,
                min_similarity=concept_embedding_min_similarity,
                keep_best_below_min_similarity=concept_embedding_rank1_fallback,
                cache_path=concept_embedding_cache,
                query_cache_path=query_term_embedding_cache,
                query_cache_read_only=query_term_embedding_cache_read_only,
            )
        semantic_base = SeededMentionsCandidateGenerator(
            tools,
            seeder,
            exclude_summary_sections=exclude_summary_sections,
            preselection_policy="semantic_before_cutoff",
        )
        generator = SimilarityWeightedSeededMentionsCandidateGenerator(semantic_base)
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_embedding_exact_safe_seeded_similarity_weighted":
        semantic_seeder = concept_seeder
        if semantic_seeder is None:
            if concept_embedding_model is None:
                raise ValueError(
                    "concept_embedding_model is required for "
                    "mode='mentions_embedding_exact_safe_seeded_similarity_weighted'"
                )
            semantic_seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=concept_embedding_model,
                concepts_per_term=concepts_per_term,
                min_similarity=concept_embedding_min_similarity,
                keep_best_below_min_similarity=concept_embedding_rank1_fallback,
                cache_path=concept_embedding_cache,
                query_cache_path=query_term_embedding_cache,
                query_cache_read_only=query_term_embedding_cache_read_only,
            )
        if not isinstance(semantic_seeder, EmbeddingConceptSeeder):
            raise TypeError(
                "exact-safe augmentation requires an EmbeddingConceptSeeder "
                "as the semantic base"
            )
        augmented_seeder = ExactSafeAugmentedConceptSeeder(semantic_seeder)
        semantic_base = SeededMentionsCandidateGenerator(
            tools,
            augmented_seeder,
            exclude_summary_sections=exclude_summary_sections,
        )
        generator = SimilarityWeightedSeededMentionsCandidateGenerator(
            semantic_base
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_embedding_exact_safe_same_cui_seeded_similarity_weighted":
        semantic_seeder = concept_seeder
        if semantic_seeder is None:
            if concept_embedding_model is None:
                raise ValueError(
                    "concept_embedding_model is required for "
                    "mode='mentions_embedding_exact_safe_same_cui_seeded_similarity_weighted'"
                )
            semantic_seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=concept_embedding_model,
                concepts_per_term=concepts_per_term,
                min_similarity=concept_embedding_min_similarity,
                keep_best_below_min_similarity=concept_embedding_rank1_fallback,
                cache_path=concept_embedding_cache,
                query_cache_path=query_term_embedding_cache,
                query_cache_read_only=query_term_embedding_cache_read_only,
            )
        if not isinstance(semantic_seeder, EmbeddingConceptSeeder):
            raise TypeError(
                "same-CUI S2 augmentation requires an EmbeddingConceptSeeder "
                "as the semantic base"
            )
        exact_safe = ExactSafeAugmentedConceptSeeder(semantic_seeder)
        same_cui = SameCUIAugmentedConceptSeeder(exact_safe)
        semantic_base = SeededMentionsCandidateGenerator(
            tools,
            same_cui,
            exclude_summary_sections=exclude_summary_sections,
        )
        generator = SimilarityWeightedSeededMentionsCandidateGenerator(
            semantic_base
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_lexical_semantic_best_channel":
        seeder = concept_seeder
        if seeder is None:
            if concept_embedding_model is None:
                raise ValueError(
                    "concept_embedding_model is required for "
                    "mode='mentions_lexical_semantic_best_channel'"
                )
            seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=concept_embedding_model,
                concepts_per_term=concepts_per_term,
                min_similarity=concept_embedding_min_similarity,
                keep_best_below_min_similarity=concept_embedding_rank1_fallback,
                cache_path=concept_embedding_cache,
                query_cache_path=query_term_embedding_cache,
                query_cache_read_only=query_term_embedding_cache_read_only,
            )
        lexical_generator = MentionsCandidateGenerator(
            tools,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        semantic_generator = SeededMentionsCandidateGenerator(
            tools,
            seeder,
            exclude_summary_sections=exclude_summary_sections,
        )
        generator = HybridBestChannelCandidateGenerator(
            lexical_generator,
            semantic_generator,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode in {
        "mentions_isa_forward_artifact",
        "mentions_isa_forward_artifact_strict_direct_first",
    }:
        if client is None:
            raise ValueError(
                f"client is required for mode={normalized_mode!r}"
            )
        if not isa_connections_path:
            raise ValueError(
                "isa_connections_path is required for "
                f"mode={normalized_mode!r}"
            )
        is_strict_direct_first = (
            normalized_mode
            == "mentions_isa_forward_artifact_strict_direct_first"
        )
        generator = ISAArtifactCandidateGenerator(
            client,
            connections_path=isa_connections_path,
            max_depth=isa_max_depth,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
            seed_match_policy=(
                "exact_name_only"
                if is_strict_direct_first
                else "permissive"
            ),
            direct_first_graph_second=is_strict_direct_first,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_isa_semantic_safe_rerank":
        if client is None:
            raise ValueError(
                "client is required for mode='mentions_isa_semantic_safe_rerank'"
            )
        if not isa_connections_path:
            raise ValueError(
                "isa_connections_path is required for "
                "mode='mentions_isa_semantic_safe_rerank'"
            )
        generator = ISASafeRerankCandidateGenerator(
            client,
            artifact_path=isa_connections_path,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode in {
        "mentions_nonhier_artifact_raw",
        "mentions_nonhier_artifact_safe",
        "mentions_nonhier_artifact_raw_strict",
        "mentions_nonhier_artifact_safe_strict",
        "mentions_nonhier_artifact_raw_strict_direct_first",
        "mentions_nonhier_artifact_safe_strict_direct_first",
        "mentions_nonhier_artifact_safe_strict_direct_first_frozen",
        "mentions_nonhier_artifact_safe_strict_direct_first_rescue",
    }:
        if client is None:
            raise ValueError(
                f"client is required for mode={normalized_mode!r}"
            )
        if not nonhier_artifact_path:
            raise ValueError(
                "nonhier_artifact_path is required for "
                f"mode={normalized_mode!r}"
            )
        is_raw = normalized_mode in {
            "mentions_nonhier_artifact_raw",
            "mentions_nonhier_artifact_raw_strict",
            "mentions_nonhier_artifact_raw_strict_direct_first",
        }
        is_strict = normalized_mode in {
            "mentions_nonhier_artifact_raw_strict",
            "mentions_nonhier_artifact_safe_strict",
            "mentions_nonhier_artifact_raw_strict_direct_first",
            "mentions_nonhier_artifact_safe_strict_direct_first",
            "mentions_nonhier_artifact_safe_strict_direct_first_frozen",
            "mentions_nonhier_artifact_safe_strict_direct_first_rescue",
        }
        expected_artifact_name = (
            "nonhier_semantic_raw_v1"
            if is_raw
            else "nonhier_semantic_safe_v1"
        )
        generator = NonHierArtifactCandidateGenerator(
            client,
            artifact_path=nonhier_artifact_path,
            expected_artifact_name=expected_artifact_name,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
            seed_match_policy=(
                "exact_name_only" if is_strict else "permissive"
            ),
            support_only_ranking_active=(
                False
                if normalized_mode in {
                    "mentions_nonhier_artifact_safe_strict",
                    "mentions_nonhier_artifact_safe_strict_direct_first",
                }
                else True
            ),
            direct_first_graph_second=normalized_mode in {
                "mentions_nonhier_artifact_raw_strict_direct_first",
                "mentions_nonhier_artifact_safe_strict_direct_first",
                "mentions_nonhier_artifact_safe_strict_direct_first_frozen",
                "mentions_nonhier_artifact_safe_strict_direct_first_rescue",
            },
            freeze_direct_candidate_pool=(
                normalized_mode
                == "mentions_nonhier_artifact_safe_strict_direct_first_frozen"
            ),
            baseline_tools=(
                tools
                if normalized_mode
                == "mentions_nonhier_artifact_safe_strict_direct_first_frozen"
                else None
            ),
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode in {
        "mentions_direct_balanced",
        "mentions_bridge_sa_top5",
        "mentions_direct_bridge_sa_top3",
        "mentions_direct_bridge_sa_top5",
        "mentions_direct_bridge_sa_top10",
    }:
        if client is None:
            raise ValueError(
                f"client is required for mode={normalized_mode!r}"
            )
        if not connection_config_path:
            raise ValueError(
                "connection_config_path is required for frozen "
                f"connection mode={normalized_mode!r}"
            )
        generator = ConnectionArtifactCandidateGenerator(
            client,
            config_path=connection_config_path,
            mode_name=normalized_mode,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()


    elif normalized_mode in {
        "semantic_weighted_pool_union_direct_bridge_sa_top3",
        "semantic_weighted_pool_rrf_direct_bridge_sa_top3",
    }:
        if client is None:
            raise ValueError(f"client is required for mode={normalized_mode!r}")
        if not connection_config_path:
            raise ValueError("connection_config_path is required for pool-preserving fusion")
        seeder = concept_seeder
        if seeder is None:
            if concept_embedding_model is None:
                raise ValueError("concept_embedding_model is required for pool-preserving fusion")
            seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=concept_embedding_model,
                concepts_per_term=concepts_per_term,
                min_similarity=concept_embedding_min_similarity,
                keep_best_below_min_similarity=concept_embedding_rank1_fallback,
                cache_path=concept_embedding_cache,
                query_cache_path=query_term_embedding_cache,
                query_cache_read_only=query_term_embedding_cache_read_only,
            )
        fusion_policy = (
            "rrf" if normalized_mode == "semantic_weighted_pool_rrf_direct_bridge_sa_top3"
            else "semantic_score"
        )
        generator = PoolPreservingSemanticConnectionCandidateGenerator(
            tools, client, seeder, config_path=connection_config_path, mode_name=normalized_mode,
            graph_candidate_k=(graph_candidate_k or candidate_k), fusion_policy=fusion_policy,
            rrf_k=rrf_k, exclude_summary_sections=exclude_summary_sections,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode in {
        "semantic_weighted_direct_balanced",
        "semantic_weighted_bridge_sa_top5",
        "semantic_weighted_direct_bridge_sa_top3",
        "semantic_weighted_direct_bridge_sa_top5",
        "semantic_weighted_direct_bridge_sa_top10",
    }:
        if client is None:
            raise ValueError(
                f"client is required for mode={normalized_mode!r}"
            )
        if not connection_config_path:
            raise ValueError(
                "connection_config_path is required for semantic-weighted "
                f"connection mode={normalized_mode!r}"
            )
        seeder = concept_seeder
        if seeder is None:
            if concept_embedding_model is None:
                raise ValueError(
                    "concept_embedding_model is required for "
                    f"mode={normalized_mode!r}"
                )
            seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=concept_embedding_model,
                concepts_per_term=concepts_per_term,
                min_similarity=concept_embedding_min_similarity,
                keep_best_below_min_similarity=concept_embedding_rank1_fallback,
                cache_path=concept_embedding_cache,
                query_cache_path=query_term_embedding_cache,
                query_cache_read_only=query_term_embedding_cache_read_only,
            )
        generator = SemanticWeightedConnectionCandidateGenerator(
            tools,
            client,
            seeder,
            config_path=connection_config_path,
            mode_name=normalized_mode,
            exclude_summary_sections=exclude_summary_sections,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_same_as":
        if client is None:
            raise ValueError(
                "client is required for mode='mentions_same_as'"
            )
        generator = ConceptGraphExpansionCandidateGenerator(
            client,
            include_same_as=True,
            umls_policies=[],
            include_review_needed=False,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_umls_safe":
        if client is None:
            raise ValueError(
                "client is required for mode='mentions_umls_safe'"
            )
        generator = ConceptGraphExpansionCandidateGenerator(
            client,
            include_same_as=True,
            umls_policies=["safe"],
            include_review_needed=False,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_same_as_rescue":
        if client is None:
            raise ValueError(
                "client is required for mode='mentions_same_as_rescue'"
            )
        direct_generator = MentionsCandidateGenerator(
            tools,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        expansion_generator = ConceptGraphExpansionCandidateGenerator(
            client,
            include_same_as=True,
            umls_policies=[],
            include_review_needed=False,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        generator = RescueConceptGraphExpansionCandidateGenerator(
            direct_generator,
            expansion_generator,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_umls_safe_rescue":
        if client is None:
            raise ValueError(
                "client is required for mode='mentions_umls_safe_rescue'"
            )
        direct_generator = MentionsCandidateGenerator(
            tools,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        expansion_generator = ConceptGraphExpansionCandidateGenerator(
            client,
            include_same_as=True,
            umls_policies=["safe"],
            include_review_needed=False,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        generator = RescueConceptGraphExpansionCandidateGenerator(
            direct_generator,
            expansion_generator,
        )
        expander = NoOpExpander()
        reranker = NoOpReranker()

    elif normalized_mode == "mentions_descendants":
        if client is None:
            raise ValueError(
                "client is required for mode='mentions_descendants'"
            )
        generator = MentionsCandidateGenerator(
            tools,
            ranking_mode="concept_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        expander = DescendantExpander(
            client,
            max_depth=hierarchy_max_depth,
            max_rows=max_expanded_rows,
            exclude_summary_sections=exclude_summary_sections,
        )
        reranker = SeedRoundRobinReranker(
            descendants_per_seed=descendants_per_seed
        )

    else:
        raise RuntimeError(f"Unhandled modular KG mode: {normalized_mode}")

    return ModularKGRetrievalPipeline(
        router=router,
        candidate_generator=generator,
        expander=expander,
        reranker=reranker,
        mode=normalized_mode,
        candidate_k=candidate_k,
        final_k=final_k,
        document_ids=document_ids,
        router_term_normalization=router_term_normalization,
    )


def _validate_mode(value: str) -> ModularKGMode:
    normalized = str(value).strip().lower()
    allowed = {
        "mentions_only",
        "mentions_lexical_seeded",
        "mentions_embedding_seeded",
        "mentions_embedding_seeded_similarity_weighted",
        "mentions_embedding_seeded_semantic_preselected_similarity_weighted",
        "mentions_embedding_seeded_preselection_rrf_similarity_weighted",
        "mentions_embedding_exact_safe_seeded_similarity_weighted",
        "mentions_embedding_exact_safe_same_cui_seeded_similarity_weighted",
        "mentions_lexical_semantic_best_channel",
        "mentions_descendants",
        "mentions_same_as",
        "mentions_umls_safe",
        "mentions_isa_forward_artifact",
        "mentions_isa_forward_artifact_strict_direct_first",
    "mentions_isa_semantic_safe_rerank",
        "mentions_nonhier_artifact_raw",
        "mentions_nonhier_artifact_safe",
        "mentions_nonhier_artifact_raw_strict",
        "mentions_nonhier_artifact_safe_strict",
        "mentions_nonhier_artifact_raw_strict_direct_first",
        "mentions_nonhier_artifact_safe_strict_direct_first",
        "mentions_nonhier_artifact_safe_strict_direct_first_frozen",
        "mentions_nonhier_artifact_safe_strict_direct_first_rescue",
        "mentions_same_as_rescue",
        "mentions_umls_safe_rescue",
        "mentions_direct_balanced",
        "mentions_bridge_sa_top5",
        "mentions_direct_bridge_sa_top3",
        "mentions_direct_bridge_sa_top5",
        "mentions_direct_bridge_sa_top10",
        "semantic_weighted_direct_balanced",
        "semantic_weighted_bridge_sa_top5",
        "semantic_weighted_direct_bridge_sa_top3",
        "semantic_weighted_direct_bridge_sa_top5",
        "semantic_weighted_direct_bridge_sa_top10",
        "semantic_weighted_pool_union_direct_bridge_sa_top3",
        "semantic_weighted_pool_rrf_direct_bridge_sa_top3",
    }
    if normalized not in allowed:
        raise ValueError(f"Unsupported modular KG mode: {value!r}")
    return normalized  # type: ignore[return-value]


def _validate_top_k(value: int, field_name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if normalized < 1 or normalized > 100:
        raise ValueError(f"{field_name} must be between 1 and 100")
    return normalized


def _normalize_optional_values(
    values: Sequence[str] | str | None,
) -> list[str]:
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else list(values)
    output: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        normalized = str(value).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def _validate_question(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("question must be a non-empty string")
    return normalized


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"

def _generate_candidates_with_seeds(
    generator: CandidateGeneratorProtocol,
    terms: Sequence[str] | str,
    *,
    top_k: int,
    require_all: bool,
    document_ids: Sequence[str],
) -> tuple[list[KGCandidate], list[ConceptSeed]]:
    generate_with_seeds = getattr(generator, "generate_with_seeds", None)
    if callable(generate_with_seeds):
        candidates, seeds = generate_with_seeds(
            terms,
            top_k=top_k,
            require_all=require_all,
            document_ids=document_ids,
        )
        return list(candidates), list(seeds)

    candidates = generator.generate(
        terms,
        top_k=top_k,
        require_all=require_all,
        document_ids=document_ids,
    )
    return list(candidates), []

