from __future__ import annotations

from agentic_rag.kg.candidate_generators import KGCandidate
from agentic_rag.kg.fixed_pool_rerankers import rerank_bm25plus
from agentic_rag.kg.models import KGSectionResult


def _candidate(uid: str, text: str, rank: int) -> KGCandidate:
    doc, sid = uid.split("::", 1)
    return KGCandidate(
        section=KGSectionResult(
            section_uid=uid,
            document_id=doc,
            section_id=sid,
            printed_section_id=sid,
            title=uid,
            text=text,
            retrieval_unit_id=uid,
            section_view_role="retrieval",
            represented_section_ids=[sid],
        ),
        source="mentions",
        source_rank=rank,
        direct=True,
        seed_uid=uid,
        seed_rank=rank,
        graph_distance=0,
    )


def test_bm25plus_preserves_fixed_pool_membership_and_reranks_by_full_question():
    candidates = [
        _candidate("Doc::1", "alpha alpha treatment", 1),
        _candidate("Doc::2", "beta beta beta therapy", 2),
        _candidate("Doc::3", "gamma management", 3),
    ]

    ranked, details = rerank_bm25plus(candidates, "beta therapy")

    assert ranked[0].section_uid == "Doc::2"
    assert {item.section_uid for item in ranked} == {
        item.section_uid for item in candidates
    }
    assert len(ranked) == len(candidates)
    assert [row["bm25plus_rank"] for row in details] == [1, 2, 3]
