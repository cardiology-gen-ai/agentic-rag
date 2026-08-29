from __future__ import annotations

import pytest

from agentic_rag.kg.candidate_generators import KGCandidate
from agentic_rag.kg.fixed_pool_rerankers import rank_reciprocal_rank_fusion
from agentic_rag.kg.models import KGSectionResult


def _candidate(uid: str, rank: int) -> KGCandidate:
    doc, sid = uid.split("::", 1)
    section = KGSectionResult(
        section_uid=uid,
        document_id=doc,
        section_id=sid,
        printed_section_id=sid,
        title=uid,
        text="clinical text",
        retrieval_unit_id=uid,
        section_view_role="retrieval",
        represented_section_ids=[sid],
    )
    return KGCandidate(
        section=section,
        source="mentions",
        source_rank=rank,
        direct=True,
        seed_uid=uid,
        seed_rank=rank,
        graph_distance=0,
    )


def test_rrf_preserves_membership_and_uses_parent_ranks():
    a = _candidate("Doc::A", 1)
    b = _candidate("Doc::B", 2)
    c = _candidate("Doc::C", 3)

    ranking, details = rank_reciprocal_rank_fusion(
        {
            "li": [a, b, c],
            "bm25plus": [b, c, a],
        },
        constant=60,
    )

    assert [item.section_uid for item in ranking] == ["Doc::B", "Doc::A", "Doc::C"]
    assert {item.section_uid for item in ranking} == {"Doc::A", "Doc::B", "Doc::C"}
    assert [item.final_rank for item in ranking] == [1, 2, 3]
    by_uid = {row["section_uid"]: row for row in details}
    assert by_uid["Doc::B"]["li_rank"] == 2
    assert by_uid["Doc::B"]["bm25plus_rank"] == 1
    assert by_uid["Doc::B"]["rrf_constant"] == 60


def test_rrf_tie_break_is_parent_symmetric_and_deterministic():
    a = _candidate("Doc::A", 1)
    c = _candidate("Doc::C", 2)

    first, _ = rank_reciprocal_rank_fusion(
        {"li": [a, c], "bm25plus": [c, a]}, constant=60
    )
    second, _ = rank_reciprocal_rank_fusion(
        {"li": [a, c], "bm25plus": [c, a]}, constant=60
    )

    # Both candidates have identical RRF score/rank-sum/best-rank; UID is the
    # final deterministic tie-break, so no parent is privileged.
    assert [item.section_uid for item in first] == ["Doc::A", "Doc::C"]
    assert [item.section_uid for item in second] == ["Doc::A", "Doc::C"]


def test_rrf_rejects_candidate_membership_mismatch():
    a = _candidate("Doc::A", 1)
    b = _candidate("Doc::B", 2)
    c = _candidate("Doc::C", 3)

    with pytest.raises(RuntimeError, match="candidate-pool membership"):
        rank_reciprocal_rank_fusion(
            {"li": [a, b], "bm25plus": [a, c]}, constant=60
        )


def test_rrf_rejects_nonpositive_constant():
    a = _candidate("Doc::A", 1)
    b = _candidate("Doc::B", 2)

    with pytest.raises(ValueError, match="positive integer"):
        rank_reciprocal_rank_fusion(
            {"li": [a, b], "bm25plus": [b, a]}, constant=0
        )
