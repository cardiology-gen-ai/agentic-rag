from __future__ import annotations

from agentic_rag.agent.output import KGMentionsPlan
from agentic_rag.kg.candidate_generators import MentionsCandidateGenerator
from agentic_rag.kg.models import KGSectionResult, KGRetrievalScores
from agentic_rag.kg.pipeline import build_modular_kg_pipeline


class FakeRouter:
    def __init__(self, plan: KGMentionsPlan):
        self.plan = plan

    def route(self, question: str, *, config=None) -> KGMentionsPlan:
        return self.plan


class FakeTools:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search_sections_by_concepts(self, concepts, **kwargs):
        self.calls.append(("concepts", list(concepts), kwargs))
        return self.results

    def search_sections_by_title(self, title_terms, **kwargs):
        raise AssertionError("Title search must not be used by these baselines")


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run_read(self, query, parameters=None):
        self.calls.append((query, parameters))
        return self.rows


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


def test_mentions_plan_normalizes_terms():
    plan = KGMentionsPlan.model_validate(
        {
            "terms": [" Atrial fibrillation ", "atrial fibrillation", "HCM"],
            "require_all": False,
        }
    )

    assert plan.terms == ["Atrial fibrillation", "HCM"]


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


def test_mentions_weighted_changes_only_the_local_ranking_mode():
    result = make_result("Doc::1", 1, "Atrial fibrillation")
    tools = FakeTools([result])
    pipeline = build_modular_kg_pipeline(
        "mentions_weighted",
        router=FakeRouter(
            KGMentionsPlan(
                terms=["atrial fibrillation"],
                require_all=False,
            )
        ),
        tools=tools,
    )

    run = pipeline.retrieve("Question")

    assert run.status == "success"
    assert run.expander_name == "none"
    assert run.reranker_name == "none"
    assert tools.calls[0][2]["ranking_mode"] == "weighted_match"


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
