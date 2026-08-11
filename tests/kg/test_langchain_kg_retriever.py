from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from agentic_rag.kg.candidate_generators import KGCandidate
from agentic_rag.kg.langchain_retriever import (
    KGRetrievalError,
    LangChainKGRetriever,
)
from agentic_rag.kg.models import KGSectionResult, KGRetrievalScores


def make_section(
    uid: str = "Doc::1",
    *,
    text: str = "Section text",
    title: str = "Section title",
) -> KGSectionResult:
    document_id, section_id = uid.split("::", 1)

    kwargs = {
        "section_uid": uid,
        "document_id": document_id,
        "section_id": section_id,
        "printed_section_id": section_id,
        "title": title,
        "level": 3,
        "text": text,
        "page_start": 10,
        "page_end": 11,
        "matched_concepts": ["heart failure"],
        "matched_terms": ["heart failure"],
        "score": 2.0,
        "score_type": "concept_match",
        "scores": KGRetrievalScores(
            concept_match=2.0,
            weighted_match=3.0,
        ),
        "rank": 1,
    }

    # These fields are part of the current Section-view contract.  Keeping the
    # construction conditional also makes the adapter test tolerant of an
    # older local checkout while still checking provenance when available.
    optional = {
        "retrieval_unit_id": "ru-1",
        "section_view_schema_version": "1",
        "section_view_role": "retrieval",
        "is_aggregated": True,
        "source_section_ids": ["1", "1.1"],
        "represented_section_ids": ["1", "1.1"],
    }
    model_fields = KGSectionResult.model_fields
    kwargs.update(
        {
            key: value
            for key, value in optional.items()
            if key in model_fields
        }
    )

    return KGSectionResult(**kwargs)


def make_candidate(
    uid: str = "Doc::1",
    *,
    text: str = "Section text",
    title: str = "Section title",
) -> KGCandidate:
    return KGCandidate(
        section=make_section(
            uid,
            text=text,
            title=title,
        ),
        source="mentions",
        source_rank=1,
        final_rank=1,
        direct=True,
        seed_uid=uid,
        seed_rank=1,
        graph_distance=0,
        metadata={"support": "direct"},
    )


def make_run(
    *,
    status: str = "success",
    results=None,
    error: str | None = None,
    failed_stage: str | None = None,
):
    return SimpleNamespace(
        status=status,
        mode="mentions_only",
        ranking_mode="concept_match",
        expander_name="none",
        reranker_name="none",
        results=list(results or []),
        error=error,
        failed_stage=failed_stage,
    )


class FakePipeline:
    def __init__(self, runs):
        self.runs = list(runs)
        self.calls = []

    def retrieve(self, query, *, router_config=None):
        self.calls.append(
            {
                "query": query,
                "router_config": router_config,
            }
        )
        if not self.runs:
            raise AssertionError("No fake run configured")
        return self.runs.pop(0)


def test_invoke_returns_langchain_documents_in_pipeline_order() -> None:
    first = make_candidate(
        "Doc::1",
        text="First evidence",
        title="First",
    )
    second = make_candidate(
        "Doc::2",
        text="Second evidence",
        title="Second",
    )
    pipeline = FakePipeline(
        [make_run(results=[first, second])]
    )
    retriever = LangChainKGRetriever(pipeline=pipeline)

    documents = retriever.invoke("heart failure")

    assert all(isinstance(item, Document) for item in documents)
    assert [item.page_content for item in documents] == [
        "First evidence",
        "Second evidence",
    ]
    assert [item.metadata["kg_rank"] for item in documents] == [1, 2]
    assert pipeline.calls == [
        {
            "query": "heart failure",
            "router_config": None,
        }
    ]


def test_document_preserves_core_kg_metadata_and_provenance() -> None:
    candidate = make_candidate()
    retriever = LangChainKGRetriever(
        pipeline=FakePipeline(
            [make_run(results=[candidate])]
        )
    )

    document = retriever.invoke("heart failure")[0]
    metadata = document.metadata

    assert metadata["retriever"] == "kg"
    assert metadata["kg_mode"] == "mentions_only"
    assert metadata["section_uid"] == "Doc::1"
    assert metadata["document_id"] == "Doc"
    assert metadata["section_id"] == "1"
    assert metadata["matched_concepts"] == ["heart failure"]
    assert metadata["matched_terms"] == ["heart failure"]
    assert metadata["kg_score"] == 2.0
    assert metadata["kg_source"] == "mentions"
    assert metadata["kg_candidate_metadata"] == {
        "support": "direct"
    }

    if "retrieval_unit_id" in KGSectionResult.model_fields:
        assert metadata["retrieval_unit_id"] == "ru-1"
    if "source_section_ids" in KGSectionResult.model_fields:
        assert metadata["source_section_ids"] == ["1", "1.1"]


def test_no_results_is_a_valid_empty_retrieval() -> None:
    retriever = LangChainKGRetriever(
        pipeline=FakePipeline(
            [make_run(status="no_results")]
        )
    )

    assert retriever.invoke("unknown concept") == []


def test_pipeline_error_raises_instead_of_looking_like_no_results() -> None:
    retriever = LangChainKGRetriever(
        pipeline=FakePipeline(
            [
                make_run(
                    status="execution_error",
                    error="RuntimeError: reranker failed",
                    failed_stage="reranking",
                )
            ]
        )
    )

    with pytest.raises(KGRetrievalError) as exc_info:
        retriever.invoke("heart failure")

    error = exc_info.value
    assert error.status == "execution_error"
    assert error.failed_stage == "reranking"
    assert "reranker failed" in str(error)


def test_error_policy_empty_is_available_for_explicit_fallbacks() -> None:
    retriever = LangChainKGRetriever(
        pipeline=FakePipeline(
            [
                make_run(
                    status="router_error",
                    error="router unavailable",
                )
            ]
        ),
        error_policy="empty",
    )

    assert retriever.invoke("heart failure") == []


def test_router_config_is_forwarded_without_touching_runnable_config() -> None:
    pipeline = FakePipeline([make_run(status="no_results")])
    router_config = {"configurable": {"model": "test-router"}}
    retriever = LangChainKGRetriever(
        pipeline=pipeline,
        router_config=router_config,
    )

    retriever.invoke(
        "heart failure",
        config={"tags": ["kg-test"]},
    )

    assert pipeline.calls[0]["router_config"] == router_config


def test_two_invocations_do_not_share_request_results() -> None:
    pipeline = FakePipeline(
        [
            make_run(
                results=[
                    make_candidate(
                        "Doc::1",
                        text="First request",
                    )
                ]
            ),
            make_run(
                results=[
                    make_candidate(
                        "Doc::2",
                        text="Second request",
                    )
                ]
            ),
        ]
    )
    retriever = LangChainKGRetriever(pipeline=pipeline)

    first = retriever.invoke("first query")
    second = retriever.invoke("second query")

    assert first[0].page_content == "First request"
    assert second[0].page_content == "Second request"
    assert first[0].metadata["section_uid"] == "Doc::1"
    assert second[0].metadata["section_uid"] == "Doc::2"
