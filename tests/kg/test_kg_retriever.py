from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_rag.agent.output import KGRetrievalPlan
from agentic_rag.kg.models import KGSectionResult, KGRetrievalScores
from agentic_rag.kg.retriever import KGParameterizedRetriever


class FakeRouter:
    def __init__(self, plan: KGRetrievalPlan):
        self.plan = plan

    def route(self, question: str, *, config=None) -> KGRetrievalPlan:
        return self.plan


class FakeTools:
    def __init__(
        self,
        *,
        concept_results=None,
        title_results=None,
        hierarchy_rows=None,
    ):
        self.concept_results = concept_results or []
        self.title_results = title_results or []
        self.hierarchy_rows = hierarchy_rows or []
        self.calls = []
        self.hierarchy_calls = []

    def search_sections_by_concepts(self, concepts, **kwargs):
        self.calls.append(("concepts", list(concepts), kwargs))
        return self.concept_results

    def search_sections_by_title(self, title_terms, **kwargs):
        self.calls.append(("title", list(title_terms), kwargs))
        return self.title_results

    def find_hierarchical_context_matches(
        self,
        anchor_uids,
        context_uids,
        *,
        max_depth=6,
    ):
        self.hierarchy_calls.append(
            (list(anchor_uids), list(context_uids), max_depth)
        )
        return self.hierarchy_rows


def make_result(
    uid: str,
    rank: int,
    title: str,
    *,
    matched_terms=None,
    text: str | None = None,
    score: float = 1.0,
) -> KGSectionResult:
    document_id, section_id = uid.split("::", 1)
    return KGSectionResult(
        section_uid=uid,
        document_id=document_id,
        section_id=section_id,
        printed_section_id=section_id,
        title=title,
        text=text or f"Text for {title}",
        matched_concepts=[],
        matched_terms=matched_terms or [],
        score=score,
        score_type="weighted_match",
        scores=KGRetrievalScores(
            concept_match=score,
            weighted_match=score,
        ),
        rank=rank,
    )


def make_plan(payload) -> KGRetrievalPlan:
    return KGRetrievalPlan.model_validate(payload)


def test_plan_rejects_same_section_without_anchor_and_context():
    with pytest.raises(ValidationError):
        make_plan(
            {
                "intent": "diagnosis",
                "expected_scope": "single_section",
                "combination_mode": "same_section",
                "calls": [
                    {
                        "tool": "search_sections_by_concepts",
                        "role": "context",
                        "terms": ["hypertrophic cardiomyopathy"],
                        "require_all": False,
                    },
                    {
                        "tool": "search_sections_by_title",
                        "role": "context",
                        "terms": ["diagnostic criteria"],
                        "require_all": False,
                    },
                ],
            }
        )


def test_same_section_promotes_anchor_supported_by_context_ancestor():
    context = make_result(
        "Cardiomyopathies_2023::7.1",
        3,
        "Hypertrophic cardiomyopathy",
    )
    wrong_anchor = make_result(
        "Syncope_2018::4.2.4.8",
        1,
        "Diagnostic criteria",
    )
    correct_anchor = make_result(
        "Cardiomyopathies_2023::7.1.1.1",
        2,
        "Diagnostic criteria",
    )

    plan = make_plan(
        {
            "intent": "diagnosis",
            "expected_scope": "single_section",
            "combination_mode": "same_section",
            "calls": [
                {
                    "tool": "search_sections_by_concepts",
                    "role": "context",
                    "terms": ["hypertrophic cardiomyopathy"],
                    "require_all": False,
                },
                {
                    "tool": "search_sections_by_title",
                    "role": "anchor",
                    "terms": ["diagnostic criteria"],
                    "require_all": False,
                },
            ],
        }
    )

    tools = FakeTools(
        concept_results=[context],
        title_results=[wrong_anchor, correct_anchor],
        hierarchy_rows=[
            {
                "anchor_uid": correct_anchor.section_uid,
                "context_uid": context.section_uid,
                "context_document_id": context.document_id,
                "context_section_id": context.section_id,
                "context_printed_section_id": context.printed_section_id,
                "context_title": context.title,
                "hierarchy_distance": 3,
            }
        ],
    )

    run = KGParameterizedRetriever(
        FakeRouter(plan),
        tools,
        final_k=10,
        hierarchy_max_depth=6,
    ).retrieve("Question")

    assert run.status == "success"
    assert [item.section_uid for item in run.results] == [
        correct_anchor.section_uid,
        wrong_anchor.section_uid,
    ]
    assert run.results[0].combination_method == "hierarchical_context"
    assert run.results[0].context_supported is True
    assert run.results[0].context_matches[0].context_uid == context.section_uid
    assert run.results[0].context_matches[0].hierarchy_distance == 3
    assert run.results[1].context_supported is False
    assert tools.hierarchy_calls == [
        (
            [wrong_anchor.section_uid, correct_anchor.section_uid],
            [context.section_uid],
            6,
        )
    ]
    assert all(call[2]["document_ids"] is None for call in tools.calls)


def test_same_section_anchor_rescue_can_outrank_nonempty_base_results():
    context = make_result(
        "Cardiomyopathies_2023::7.1.5",
        1,
        "Risk stratification for sudden cardiac death prevention",
        text=(
            "Risk stratification uses risk model variables, risk categories, "
            "and ICD decision thresholds."
        ),
    )
    wrong_anchor = make_result(
        "Cardiomyopathies_2023::7.6.1.3",
        1,
        "Clinical course, outcome, and risk stratification",
    )

    plan = make_plan(
        {
            "intent": "risk_stratification",
            "expected_scope": "single_section",
            "combination_mode": "same_section",
            "calls": [
                {
                    "tool": "search_sections_by_concepts",
                    "role": "context",
                    "terms": [
                        "sudden cardiac death",
                        "hypertrophic cardiomyopathy",
                    ],
                    "require_all": False,
                },
                {
                    "tool": "search_sections_by_title",
                    "role": "anchor",
                    "terms": [
                        "risk stratification",
                        "risk model",
                        "risk categories",
                        "ICD decision thresholds",
                    ],
                    "require_all": False,
                },
            ],
        }
    )

    tools = FakeTools(
        concept_results=[context],
        title_results=[wrong_anchor],
    )
    run = KGParameterizedRetriever(
        FakeRouter(plan),
        tools,
        same_section_anchor_rescue=True,
    ).retrieve("Question")

    assert run.status == "success"
    assert [item.section_uid for item in run.results] == [
        context.section_uid,
        wrong_anchor.section_uid,
    ]
    assert all(
        item.combination_method == "same_section_anchor_rescue"
        for item in run.results
    )
    assert run.results[0].combination_score > run.results[1].combination_score


def test_multiple_facets_preserves_one_result_per_title_facet():
    anticoagulation = make_result(
        "Doc::6.10.3.1",
        3,
        "Anticoagulation",
        matched_terms=["anticoagulation"],
    )
    rate = make_result(
        "Doc::6.10.3.2",
        1,
        "Rate control",
        matched_terms=["rate control"],
    )
    rhythm = make_result(
        "Doc::6.10.3.3",
        2,
        "Rhythm control",
        matched_terms=["rhythm control"],
    )

    plan = make_plan(
        {
            "intent": "management",
            "expected_scope": "multiple_sections",
            "combination_mode": "multiple_facets",
            "calls": [
                {
                    "tool": "search_sections_by_title",
                    "role": "facet",
                    "terms": [
                        "anticoagulation",
                        "rate control",
                        "rhythm control",
                    ],
                    "require_all": False,
                }
            ],
        }
    )

    tools = FakeTools(
        title_results=[rate, rhythm, anticoagulation],
    )
    run = KGParameterizedRetriever(FakeRouter(plan), tools).retrieve(
        "Question"
    )

    assert [item.section_uid for item in run.results[:3]] == [
        anticoagulation.section_uid,
        rate.section_uid,
        rhythm.section_uid,
    ]
    assert run.results[0].covered_facets == ["anticoagulation"]
    assert run.results[1].covered_facets == ["rate control"]
    assert run.results[2].covered_facets == ["rhythm control"]
    assert all(
        item.combination_method == "facet_preserving"
        for item in run.results
    )


def test_multiple_facets_context_candidate_injection_promotes_context_hit():
    context = make_result(
        "Cardiomyopathies_2023::7.1.5",
        1,
        "Sudden cardiac death prevention in hypertrophic cardiomyopathy",
        matched_terms=["sudden cardiac death", "hypertrophic cardiomyopathy"],
        score=16.0,
    )
    facet_one = make_result(
        "Cardiomyopathies_2023::7.6.1.3",
        1,
        "Clinical course, outcome, and risk stratification",
        matched_terms=["risk stratification"],
    )
    facet_two = make_result(
        "Cardiomyopathies_2023::7.6.2.3",
        2,
        "Clinical course, management, and sudden death risk stratification",
        matched_terms=["risk stratification"],
    )

    plan = make_plan(
        {
            "intent": "risk_stratification",
            "expected_scope": "multiple_sections",
            "combination_mode": "multiple_facets",
            "calls": [
                {
                    "tool": "search_sections_by_concepts",
                    "role": "context",
                    "terms": [
                        "sudden cardiac death",
                        "hypertrophic cardiomyopathy",
                    ],
                    "require_all": False,
                },
                {
                    "tool": "search_sections_by_title",
                    "role": "facet",
                    "terms": [
                        "risk stratification",
                        "risk models",
                        "risk modifiers",
                        "risk categories",
                        "ICD decision thresholds",
                    ],
                    "require_all": False,
                },
            ],
        }
    )

    tools = FakeTools(
        concept_results=[context],
        title_results=[facet_one, facet_two],
    )
    run = KGParameterizedRetriever(
        FakeRouter(plan),
        tools,
        multiple_facets_context_aware_merge=True,
        multiple_facets_context_candidate_injection=True,
    ).retrieve("Question")

    assert run.status == "success"
    assert [item.section_uid for item in run.results] == [
        context.section_uid,
        facet_one.section_uid,
        facet_two.section_uid,
    ]
    assert all(
        item.combination_method
        == "context_aware_facet_context_injection"
        for item in run.results
    )
    assert run.results[0].combination_score > run.results[1].combination_score


def test_alternative_retrieval_uses_rrf_and_rewards_overlap():
    shared_concept = make_result("Doc::B", 2, "B")
    shared_title = make_result("Doc::B", 1, "B")
    concept_only = make_result("Doc::A", 1, "A")
    title_only = make_result("Doc::C", 2, "C")

    plan = make_plan(
        {
            "intent": "other",
            "expected_scope": "single_section",
            "combination_mode": "alternative_retrieval",
            "calls": [
                {
                    "tool": "search_sections_by_concepts",
                    "role": "alternative",
                    "terms": ["topic"],
                    "require_all": False,
                },
                {
                    "tool": "search_sections_by_title",
                    "role": "alternative",
                    "terms": ["topic"],
                    "require_all": False,
                },
            ],
        }
    )

    tools = FakeTools(
        concept_results=[concept_only, shared_concept],
        title_results=[shared_title, title_only],
    )
    run = KGParameterizedRetriever(FakeRouter(plan), tools).retrieve(
        "Question"
    )

    assert run.results[0].section_uid == "Doc::B"
    assert run.results[0].combination_method == "rrf"
    assert run.results[0].combination_score is not None
    assert len(run.results[0].contributions) == 2


def test_direct_returns_single_anchor_ranking():
    result = make_result("Doc::1", 1, "Cardiac magnetic resonance")
    plan = make_plan(
        {
            "intent": "diagnosis",
            "expected_scope": "single_section",
            "combination_mode": "direct",
            "calls": [
                {
                    "tool": "search_sections_by_concepts",
                    "role": "anchor",
                    "terms": ["cardiac magnetic resonance"],
                    "require_all": False,
                }
            ],
        }
    )

    run = KGParameterizedRetriever(
        FakeRouter(plan),
        FakeTools(concept_results=[result]),
    ).retrieve("Question")

    assert run.status == "success"
    assert run.results[0].section_uid == result.section_uid
    assert run.results[0].combination_method == "direct"
