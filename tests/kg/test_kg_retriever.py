from __future__ import annotations

from agentic_rag.agent.output import KGRetrievalPlan
from agentic_rag.kg.models import KGSectionResult, KGRetrievalScores
from agentic_rag.kg.retriever import KGParameterizedRetriever


class FakeRouter:
    def __init__(self, plan: KGRetrievalPlan):
        self.plan = plan

    def route(self, question: str, *, config=None) -> KGRetrievalPlan:
        return self.plan


class FailingRouter:
    def route(self, question: str, *, config=None):
        raise RuntimeError("router failed")


class FakeTools:
    def __init__(self, concept_results, title_results):
        self.concept_results = concept_results
        self.title_results = title_results
        self.calls = []

    def search_sections_by_concepts(self, concepts, **kwargs):
        self.calls.append(("concepts", list(concepts), kwargs))
        return self.concept_results

    def search_sections_by_title(self, title_terms, **kwargs):
        self.calls.append(("title", list(title_terms), kwargs))
        return self.title_results


def make_result(uid: str, rank: int, title: str) -> KGSectionResult:
    section_id = uid.split("::", 1)[1]
    return KGSectionResult(
        section_uid=uid,
        document_id="Doc",
        section_id=section_id,
        printed_section_id=section_id,
        title=title,
        text=f"Text for {title}",
        matched_concepts=[],
        matched_terms=[],
        score=1.0,
        score_type="weighted_match",
        scores=KGRetrievalScores(
            concept_match=1.0,
            weighted_match=1.0,
        ),
        rank=rank,
    )


def make_plan() -> KGRetrievalPlan:
    return KGRetrievalPlan.model_validate(
        {
            "intent": "diagnosis",
            "expected_scope": "single_section",
            "calls": [
                {
                    "tool": "search_sections_by_concepts",
                    "terms": ["hypertrophic cardiomyopathy"],
                    "require_all": False,
                },
                {
                    "tool": "search_sections_by_title",
                    "terms": ["diagnostic criteria"],
                    "require_all": False,
                },
            ],
        }
    )


def test_retriever_executes_calls_without_document_filter_and_fuses():
    a = make_result("Doc::A", 1, "A")
    b_concept = make_result("Doc::B", 2, "B")
    b_title = make_result("Doc::B", 1, "B")
    c = make_result("Doc::C", 2, "C")

    tools = FakeTools([a, b_concept], [b_title, c])
    retriever = KGParameterizedRetriever(
        FakeRouter(make_plan()),
        tools,
        candidate_k=15,
        final_k=10,
        rrf_k=60,
    )

    run = retriever.retrieve("Question")

    assert run.status == "success"
    assert [result.section_uid for result in run.results] == [
        "Doc::B",
        "Doc::A",
        "Doc::C",
    ]
    assert len(run.results[0].contributions) == 2
    assert all(call[2]["document_ids"] is None for call in tools.calls)
    assert all(call[2]["top_k"] == 15 for call in tools.calls)


def test_retriever_returns_router_error_trace():
    tools = FakeTools([], [])
    retriever = KGParameterizedRetriever(FailingRouter(), tools)

    run = retriever.retrieve("Question")

    assert run.status == "router_error"
    assert run.plan is None
    assert run.results == []
    assert "router failed" in run.error


def test_retriever_returns_no_results_when_tools_succeed_empty():
    tools = FakeTools([], [])
    retriever = KGParameterizedRetriever(
        FakeRouter(make_plan()),
        tools,
    )

    run = retriever.retrieve("Question")

    assert run.status == "no_results"
    assert run.results == []
    assert [execution.status for execution in run.tool_executions] == [
        "no_results",
        "no_results",
    ]
