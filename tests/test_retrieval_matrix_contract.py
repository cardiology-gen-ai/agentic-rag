from __future__ import annotations

import json

from langchain_core.documents import Document

from agentic_rag.evaluation.dataset import load_evaluation_questions
from agentic_rag.evaluation.document_metrics import (
    compute_document_coverage_metrics,
)
from agentic_rag.evaluation.evidence import (
    EvidenceSection,
    RetrievedEvidence,
)
from agentic_rag.evaluation.fusion import (
    build_component_ranking,
    reciprocal_rank_fuse_components,
)
from agentic_rag.evaluation.retrieval_matrix import assert_same_membership
from agentic_rag.utils.bm25_rerank import rerank_dense_candidates_bm25plus


def _hier_doc(doc_id: str, unit_id: str, record_id: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": doc_id,
            "record_id": record_id,
            "retrieval_unit_id": unit_id,
            "represented_section_ids": [unit_id],
            "prebuilt_source_type": "hierarchical_section_view",
        },
    )


def test_rerank_identity_is_document_scoped() -> None:
    candidates = [
        _hier_doc("Doc_A", "1", "same", "alpha beta"),
        _hier_doc("Doc_B", "1", "same", "beta gamma"),
    ]
    reranked = rerank_dense_candidates_bm25plus(
        "beta", candidates, expected_candidate_count=2
    )
    assert_same_membership(candidates, reranked)
    assert len(reranked) == 2


def test_rrf_record_mapping_is_document_scoped() -> None:
    dense = [
        _hier_doc("Doc_A", "1", "same", "alpha"),
        _hier_doc("Doc_B", "1", "same", "beta"),
    ]
    bm25 = list(reversed(dense))
    fused = reciprocal_rank_fuse_components(
        [
            build_component_ranking("dense", dense),
            build_component_ranking("bm25", bm25),
        ],
        rrf_k=60,
        top_k=2,
    )
    assert len(fused.documents) == 2
    assert {item.metadata["doc_id"] for item in fused.documents} == {
        "Doc_A",
        "Doc_B",
    }


def test_cross_document_metrics_require_both_gold_documents() -> None:
    gold = frozenset(
        {
            EvidenceSection("Doc_A", "1"),
            EvidenceSection("Doc_B", "2"),
        }
    )
    ranking = [
        RetrievedEvidence(
            document_id="Doc_A",
            retrieval_unit_id="1",
            covered_sections=frozenset({EvidenceSection("Doc_A", "1")}),
            raw_rank=1,
            source_record_ids=("a",),
            source_type="hierarchical_section_view",
        ),
        RetrievedEvidence(
            document_id="Doc_X",
            retrieval_unit_id="9",
            covered_sections=frozenset({EvidenceSection("Doc_X", "9")}),
            raw_rank=2,
            source_record_ids=("x",),
            source_type="hierarchical_section_view",
        ),
    ]
    metrics = compute_document_coverage_metrics(ranking, gold, cutoffs=(1, 2))
    assert metrics.first_gold_document_rank == 1
    assert metrics.at(1).hit == 1.0
    assert metrics.at(1).recall == 0.5
    assert metrics.at(1).complete_recall == 0.0
    assert metrics.at(2).recall == 0.5
    assert metrics.at(2).wrong_document_fraction == 0.5


def test_same_membership_rejects_cross_document_change() -> None:
    before = [_hier_doc("Doc_A", "1", "same", "alpha")]
    after = [_hier_doc("Doc_B", "1", "same", "alpha")]
    try:
        assert_same_membership(before, after)
    except AssertionError:
        pass
    else:
        raise AssertionError("document-scoped membership change was accepted")


def test_dataset_loader_accepts_cross_document_group(tmp_path) -> None:
    dataset = {
        "metadata": {"total_questions": 1},
        "questions": [
            {
                "id": "XD_01",
                "question": "Combine evidence from both guidelines.",
                "complexity": "advanced",
                "sources": [
                    {
                        "document_id": "Doc_A",
                        "sections": ["1. Section A"],
                    },
                    {
                        "document_id": "Doc_B",
                        "sections": ["2. Section B"],
                    },
                ],
                "metadata": {
                    "evaluation_group": "cross_document",
                    "question_type": "cross_document",
                },
            }
        ],
    }
    path = tmp_path / "cross_document.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    questions = load_evaluation_questions(path)

    assert len(questions) == 1
    assert questions[0].group == "cross_document"
    assert {(s.document_id, s.section_id) for s in questions[0].gold_sections} == {
        ("Doc_A", "1"),
        ("Doc_B", "2"),
    }
