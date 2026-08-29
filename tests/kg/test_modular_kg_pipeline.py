from __future__ import annotations

from agentic_rag.agent.output import KGMentionsPlan
from agentic_rag.kg.candidate_generators import (
    ConceptGraphExpansionCandidateGenerator,
    MentionsCandidateGenerator,
    RescueConceptGraphExpansionCandidateGenerator,
)
from agentic_rag.kg.concept_seeders import ConceptSeed, LexicalConceptSeeder
from agentic_rag.kg.models import KGSectionResult, KGRetrievalScores
from agentic_rag.kg.pipeline import build_modular_kg_pipeline, _validate_mode


class FakeRouter:
    def __init__(self, plan: KGMentionsPlan):
        self.plan = plan

    def route(self, question: str, *, config=None) -> KGMentionsPlan:
        return self.plan


class FakeTools:
    def __init__(self, results, *, catalogue=None, seeded_results=None):
        self.results = results
        self.catalogue = catalogue or []
        self.seeded_results = seeded_results or []
        self.calls = []

    def search_sections_by_concepts(self, concepts, **kwargs):
        self.calls.append(("concepts", list(concepts), kwargs))
        return self.results

    def search_sections_by_title(self, title_terms, **kwargs):
        raise AssertionError("Title search must not be used by these baselines")

    def list_concept_catalogue(self, **kwargs):
        self.calls.append(("catalogue", [], kwargs))
        return self.catalogue

    def search_sections_by_concept_seeds(self, seeds, **kwargs):
        self.calls.append(
            (
                "seeded_concepts",
                [seed.model_dump(mode="json") for seed in seeds],
                kwargs,
            )
        )
        return self.seeded_results


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run_read(self, query, parameters=None):
        self.calls.append((query, parameters))
        return self.rows


class FakeSeeder:
    name = "fake_seeder"
    concepts_per_term = 3

    def __init__(self, seeds):
        self.seeds = seeds
        self.calls = []

    def seed_concepts(self, terms, *, document_ids=None):
        self.calls.append((list(terms), document_ids))
        return {term: list(self.seeds) for term in terms}


def make_result(
    uid: str,
    rank: int,
    title: str,
    *,
    level: int = 3,
) -> KGSectionResult:
    document_id, section_id = uid.split("::", 1)
    return KGSectionResult(
        section_uid=uid,
        document_id=document_id,
        section_id=section_id,
        printed_section_id=section_id,
        title=title,
        level=level,
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


def descendant_row(
    uid: str,
    *,
    seed_uid: str,
    seed_rank: int,
    distance: int,
    title: str,
    level: int,
):
    document_id, section_id = uid.split("::", 1)
    return {
        "seed_uid": seed_uid,
        "seed_rank": seed_rank,
        "hierarchy_distance": distance,
        "section_uid": uid,
        "document_id": document_id,
        "section_id": section_id,
        "printed_section_id": section_id,
        "title": title,
        "level": level,
        "text": f"Text for {title}",
        "page_start": 1,
        "page_end": 1,
        "part_index": 0,
        "part_count": 1,
        "matched_concepts": [],
        "matched_terms": [],
        "score": None,
        "score_type": None,
        "scores": None,
        "match_diagnostics": [],
    }


def concept_graph_row(
    uid: str,
    *,
    evidence_source: str,
    title: str,
    relation_type: str = "SAME_AS",
    traversal_policy: str | None = None,
):
    document_id, section_id = uid.split("::", 1)
    return {
        "section_uid": uid,
        "document_id": document_id,
        "section_id": section_id,
        "printed_section_id": section_id,
        "title": title,
        "level": 4,
        "text": f"Text for {title}",
        "page_start": 1,
        "page_end": 1,
        "part_index": 0,
        "part_count": 1,
        "query_term": "hypertrophic cardiomyopathy",
        "concept_name": title.casefold(),
        "matched_value": "hypertrophic cardiomyopathy",
        "match_type": "exact_name",
        "evidence_weight": 0.9 if evidence_source == "same_as" else 0.5,
        "evidence_source": evidence_source,
        "relation_type": relation_type,
        "traversal_policy": traversal_policy,
        "review_needed": False,
        "lexical_weight": 3.0,
        "seed_concept_name": "hypertrophic cardiomyopathy",
        "seed_cui": "C0000001",
        "target_cui": "C0000002",
    }


def test_mentions_plan_normalizes_terms():
    plan = KGMentionsPlan.model_validate(
        {
            "terms": [" Atrial fibrillation ", "atrial fibrillation", "HCM"],
            "require_all": False,
        }
    )

    assert plan.terms == ["Atrial fibrillation", "HCM"]


def test_validate_mode_accepts_concept_graph_ablation_modes():
    assert _validate_mode("mentions_lexical_seeded") == "mentions_lexical_seeded"
    assert (
        _validate_mode("mentions_embedding_seeded")
        == "mentions_embedding_seeded"
    )
    assert _validate_mode("mentions_same_as") == "mentions_same_as"
    assert _validate_mode("mentions_umls_safe") == "mentions_umls_safe"
    assert (
        _validate_mode("mentions_same_as_rescue")
        == "mentions_same_as_rescue"
    )
    assert (
        _validate_mode("mentions_umls_safe_rescue")
        == "mentions_umls_safe_rescue"
    )


def test_lexical_concept_seeder_uses_categorical_ordering():
    tools = FakeTools(
        [],
        catalogue=[
            {
                "concept_name": "partial concept",
                "name": "advanced heart failure therapy",
            },
            {
                "concept_name": "canonical concept",
                "umls_canonical_name": "heart failure",
            },
            {
                "concept_name": "normalized concept",
                "normalized_name": "heart failure",
            },
            {
                "concept_name": "prefix concept",
                "name": "heart failure with preserved ejection fraction",
            },
            {
                "concept_name": "local concept",
                "name": "heart failure",
            },
        ],
    )
    seeder = LexicalConceptSeeder(tools, concepts_per_term=5)

    seed_groups = seeder.seed_concepts(["heart failure"])
    seeds = seed_groups["heart failure"]

    assert [seed.concept_name for seed in seeds] == [
        "local concept",
        "prefix concept",
        "partial concept",
    ]
    assert [seed.seed_rank for seed in seeds] == [1, 2, 3]


def test_mentions_only_uses_pure_concept_match_and_returns_subsections():
    subsection = make_result(
        "Doc::7.1.1.1",
        1,
        "Diagnostic criteria",
        level=5,
    )
    tools = FakeTools([subsection])
    pipeline = build_modular_kg_pipeline(
        "mentions_only",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["hypertrophic cardiomyopathy"],
                require_all=False,
            )
        ),
        tools=tools,
    )

    run = pipeline.retrieve("Question")

    assert run.status == "success"
    assert run.retrieval_unit == "section_node"
    assert run.unit_scope == "all_levels"
    assert run.results[0].section.level == 5
    assert run.results[0].section_uid == subsection.section_uid
    assert tools.calls[0][2]["ranking_mode"] == "concept_match"
    assert tools.calls[0][2]["document_ids"] == []


def test_mentions_lexical_seeded_uses_explicit_seed_tool():
    result = make_result("Doc::1", 1, "Hypertrophic cardiomyopathy")
    tools = FakeTools(
        [],
        catalogue=[
            {
                "concept_name": "hypertrophic cardiomyopathy",
                "name": "hypertrophic cardiomyopathy",
                "umls_aliases": ["HCM"],
                "canonical_type": "disease",
                "umls_cui": "C0000001",
            }
        ],
        seeded_results=[result],
    )
    pipeline = build_modular_kg_pipeline(
        "mentions_lexical_seeded",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["hypertrophic cardiomyopathy"],
                require_all=True,
            )
        ),
        tools=tools,
    )

    run = pipeline.retrieve("Question")

    assert run.status == "success"
    assert run.ranking_mode == "concept_match"
    assert [call[0] for call in tools.calls] == [
        "catalogue",
        "seeded_concepts",
    ]
    assert tools.calls[1][1][0]["concept_name"] == (
        "hypertrophic cardiomyopathy"
    )
    assert tools.calls[1][1][0]["method"] == "lexical"
    assert tools.calls[1][2]["query_terms"] == [
        "hypertrophic cardiomyopathy"
    ]
    assert tools.calls[1][2]["require_all"] is True
    assert run.concept_seeds[0].method == "lexical"
    assert run.results[0].source == "mentions"


def test_mentions_embedding_seeded_accepts_prebuilt_seeder():
    result = make_result("Doc::1", 1, "Dilated cardiomyopathy")
    seed = ConceptSeed(
        query_term="cardiomyopathy",
        concept_name="dilated cardiomyopathy",
        canonical_type="disease",
        umls_cui=None,
        method="embedding",
        match_type="embedding",
        seed_rank=1,
        similarity=0.9,
    )
    seeder = FakeSeeder([seed])
    tools = FakeTools([], seeded_results=[result])
    pipeline = build_modular_kg_pipeline(
        "mentions_embedding_seeded",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["cardiomyopathy"],
                require_all=False,
            )
        ),
        tools=tools,
        concept_seeder=seeder,
    )

    run = pipeline.retrieve("Question")

    assert run.status == "success"
    assert seeder.calls == [(["cardiomyopathy"], [])]
    assert tools.calls[0][0] == "seeded_concepts"
    assert tools.calls[0][1][0]["method"] == "embedding"
    assert run.concept_seeds == [seed]


def test_mentions_embedding_seeded_requires_model_without_prebuilt_seeder():
    tools = FakeTools([])

    try:
        build_modular_kg_pipeline(
            "mentions_embedding_seeded",
            router=FakeRouter(
                KGMentionsPlan(
                    terms=["cardiomyopathy"],
                    require_all=False,
                )
            ),
            tools=tools,
        )
    except ValueError as exc:
        assert "concept_embedding_model is required" in str(exc)
    else:
        raise AssertionError("Expected missing embedding model to fail")


def test_mentions_weighted_mode_is_rejected():
    tools = FakeTools([])

    try:
        build_modular_kg_pipeline(
            "mentions_weighted",
            router=FakeRouter(
                KGMentionsPlan(
                    terms=["atrial fibrillation"],
                    require_all=False,
                )
            ),
            tools=tools,
        )
    except ValueError as exc:
        assert "Unsupported modular KG mode" in str(exc)
    else:
        raise AssertionError("Removed mentions_weighted mode must be rejected")


def test_mentions_descendants_expands_has_child_and_interleaves_by_seed():
    seed_1 = make_result("Doc::7.1", 1, "Hypertrophic cardiomyopathy")
    seed_2 = make_result("Doc::12.3", 2, "Hypertrophic cardiomyopathy")
    tools = FakeTools([seed_1, seed_2])
    client = FakeClient(
        [
            descendant_row(
                "Doc::7.1.1",
                seed_uid=seed_1.section_uid,
                seed_rank=1,
                distance=1,
                title="Diagnosis",
                level=4,
            ),
            descendant_row(
                "Doc::7.1.1.1",
                seed_uid=seed_1.section_uid,
                seed_rank=1,
                distance=2,
                title="Diagnostic criteria",
                level=5,
            ),
            descendant_row(
                "Doc::12.3.1",
                seed_uid=seed_2.section_uid,
                seed_rank=2,
                distance=1,
                title="Risk factors",
                level=4,
            ),
        ]
    )
    pipeline = build_modular_kg_pipeline(
        "mentions_descendants",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["hypertrophic cardiomyopathy"],
                require_all=False,
            )
        ),
        tools=tools,
        client=client,
        final_k=5,
        descendants_per_seed=2,
    )

    run = pipeline.retrieve("Question")

    assert run.status == "success"
    assert [item.section_uid for item in run.results] == [
        "Doc::7.1",
        "Doc::7.1.1",
        "Doc::7.1.1.1",
        "Doc::12.3",
        "Doc::12.3.1",
    ]
    assert run.results[2].direct is False
    assert run.results[2].graph_distance == 2
    assert run.results[2].seed_uid == seed_1.section_uid
    assert client.calls[0][1]["max_depth"] == 3


def test_mentions_same_as_builds_and_uses_graph_expansion_generator():
    tools = FakeTools([])
    client = FakeClient(
        [
            concept_graph_row(
                "Doc::7.1.1",
                evidence_source="same_as",
                title="Diagnosis",
            )
        ]
    )
    pipeline = build_modular_kg_pipeline(
        "mentions_same_as",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["hypertrophic cardiomyopathy"],
                require_all=False,
            )
        ),
        tools=tools,
        client=client,
    )

    run = pipeline.retrieve("Question")

    assert run.status == "success"
    assert run.expander_name == "none"
    assert run.reranker_name == "none"
    assert run.ranking_mode == "concept_match"
    assert tools.calls == []
    assert len(client.calls) == 2
    assert client.calls[0][1]["umls_policies"] == []
    assert run.results[0].source == "same_as"
    assert run.results[0].direct is False
    assert run.results[0].graph_distance == 1


def test_mentions_umls_safe_builds_with_safe_policy_only():
    tools = FakeTools([])
    client = FakeClient(
        [
            concept_graph_row(
                "Doc::7.1.2",
                evidence_source="umls_neighbor",
                title="Family screening",
                relation_type="UMLS_NARROWER_THAN",
                traversal_policy="safe",
            )
        ]
    )
    pipeline = build_modular_kg_pipeline(
        "mentions_umls_safe",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["hypertrophic cardiomyopathy"],
                require_all=False,
            )
        ),
        tools=tools,
        client=client,
    )

    run = pipeline.retrieve("Question")

    assert run.status == "success"
    assert tools.calls == []
    assert len(client.calls) == 4
    assert client.calls[0][1]["umls_policies"] == ["safe"]
    assert client.calls[0][1]["include_review_needed"] is False
    assert run.results[0].source == "umls_neighbor"
    assert run.results[0].graph_distance == 2


def test_mentions_same_as_rescue_builds_and_appends_after_direct():
    direct = make_result("Doc::7.1", 1, "Hypertrophic cardiomyopathy")
    tools = FakeTools([direct])
    client = FakeClient(
        [
            concept_graph_row(
                "Doc::7.1.1",
                evidence_source="same_as",
                title="Diagnosis",
            )
        ]
    )
    pipeline = build_modular_kg_pipeline(
        "mentions_same_as_rescue",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["hypertrophic cardiomyopathy"],
                require_all=False,
            )
        ),
        tools=tools,
        client=client,
        final_k=5,
    )

    run = pipeline.retrieve("Question")

    assert run.status == "success"
    assert run.expander_name == "none"
    assert run.reranker_name == "none"
    assert run.ranking_mode == "concept_match"
    assert tools.calls[0][2]["ranking_mode"] == "concept_match"
    assert len(client.calls) == 2
    assert [item.section_uid for item in run.results] == [
        "Doc::7.1",
        "Doc::7.1.1",
    ]
    assert run.results[0].source == "mentions"
    assert run.results[1].source == "same_as"


def test_mentions_umls_safe_rescue_builds_with_safe_policy_only():
    direct = make_result("Doc::7.1", 1, "Hypertrophic cardiomyopathy")
    tools = FakeTools([direct])
    client = FakeClient(
        [
            concept_graph_row(
                "Doc::7.1.2",
                evidence_source="umls_neighbor",
                title="Family screening",
                relation_type="UMLS_NARROWER_THAN",
                traversal_policy="safe",
            )
        ]
    )
    pipeline = build_modular_kg_pipeline(
        "mentions_umls_safe_rescue",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["hypertrophic cardiomyopathy"],
                require_all=False,
            )
        ),
        tools=tools,
        client=client,
        final_k=5,
    )

    run = pipeline.retrieve("Question")

    assert run.status == "success"
    assert len(client.calls) == 4
    assert client.calls[0][1]["umls_policies"] == ["safe"]
    assert client.calls[0][1]["include_review_needed"] is False
    assert [item.section_uid for item in run.results] == [
        "Doc::7.1",
        "Doc::7.1.2",
    ]
    assert run.results[1].source == "umls_neighbor"
    assert run.results[1].metadata["expansion_source"] == "umls_neighbor"


def test_rescue_generator_preserves_direct_order_and_enriches_duplicates():
    direct_1 = make_result("Doc::2", 1, "Direct second section")
    direct_2 = make_result("Doc::1", 2, "Direct first section")
    tools = FakeTools([direct_1, direct_2])
    client = FakeClient(
        [
            concept_graph_row(
                "Doc::1",
                evidence_source="same_as",
                title="Same-as support for direct section",
            ),
            concept_graph_row(
                "Doc::3",
                evidence_source="same_as",
                title="Same-as rescued section",
            ),
        ]
    )
    direct_generator = MentionsCandidateGenerator(
        tools,
        ranking_mode="concept_match",
    )
    expansion_generator = ConceptGraphExpansionCandidateGenerator(
        client,
        include_same_as=True,
        umls_policies=[],
        include_review_needed=False,
        ranking_mode="concept_match",
    )
    generator = RescueConceptGraphExpansionCandidateGenerator(
        direct_generator,
        expansion_generator,
    )

    candidates = generator.generate(
        ["hypertrophic cardiomyopathy"],
        top_k=5,
    )

    assert [item.section_uid for item in candidates] == [
        "Doc::2",
        "Doc::1",
        "Doc::3",
    ]
    assert candidates[0].source == "mentions"
    assert candidates[1].source == "mentions"
    assert candidates[1].source_rank == 2
    assert candidates[1].direct is True
    assert candidates[1].metadata["has_expansion_support"] is True
    assert candidates[1].metadata["expansion_evidence_sources"] == [
        "same_as"
    ]
    assert candidates[1].metadata["expansion_source"] == "same_as"
    assert candidates[2].source == "same_as"
    assert candidates[2].direct is False
    assert candidates[2].metadata["has_expansion_support"] is True


def test_rescue_generator_does_not_displace_direct_top_k():
    direct_1 = make_result("Doc::1", 1, "Direct one")
    direct_2 = make_result("Doc::2", 2, "Direct two")
    tools = FakeTools([direct_1, direct_2])
    client = FakeClient(
        [
            concept_graph_row(
                "Doc::3",
                evidence_source="umls_neighbor",
                title="Potential rescue",
                relation_type="UMLS_RELATED_TO",
                traversal_policy="safe",
            )
        ]
    )
    generator = RescueConceptGraphExpansionCandidateGenerator(
        MentionsCandidateGenerator(tools, ranking_mode="concept_match"),
        ConceptGraphExpansionCandidateGenerator(
            client,
            include_same_as=True,
            umls_policies=["safe"],
            include_review_needed=False,
            ranking_mode="concept_match",
        ),
    )

    candidates = generator.generate(
        ["hypertrophic cardiomyopathy"],
        top_k=2,
    )

    assert [item.section_uid for item in candidates] == ["Doc::1", "Doc::2"]
    assert all(item.source == "mentions" for item in candidates)


def test_mentions_generator_can_be_used_independently():
    result = make_result("Doc::6.10.3.2", 1, "Rate control")
    tools = FakeTools([result])
    generator = MentionsCandidateGenerator(
        tools,
        ranking_mode="concept_match",
    )

    candidates = generator.generate(
        ["atrial fibrillation"],
        top_k=10,
        require_all=False,
    )

    assert candidates[0].section_uid == result.section_uid
    assert candidates[0].source == "mentions"
    assert candidates[0].direct is True


def test_pipeline_applies_safe_router_term_normalization_before_generation():
    tools = FakeTools([])
    pipeline = build_modular_kg_pipeline(
        "mentions_only",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["pulmonary valve disease」「RASopathy"],
                require_all=False,
            )
        ),
        tools=tools,
        router_term_normalization="safe_v1",
    )

    run = pipeline.retrieve("Question")

    assert run.status == "no_results"
    assert run.plan is not None
    assert run.plan.terms == ["pulmonary valve disease", "RASopathy"]
    assert run.router_term_normalization == "safe_v1"
    assert tools.calls[0][1] == ["pulmonary valve disease", "RASopathy"]
