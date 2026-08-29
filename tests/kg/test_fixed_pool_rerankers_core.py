from __future__ import annotations

import numpy as np

from agentic_rag.kg.candidate_generators import KGCandidate
from agentic_rag.kg.fixed_pool_rerankers import rank_specificity_coverage
from agentic_rag.kg.late_interaction import concept_maxsim_score
from agentic_rag.kg.models import KGSectionResult, KGSeededMatchDiagnostic


def _candidate(uid: str, diagnostics, rank: int) -> KGCandidate:
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
        match_diagnostics=list(diagnostics),
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


def _diag(term: str, concept: str, similarity: float, rank: int):
    return KGSeededMatchDiagnostic(
        query_term=term,
        concept_name=concept,
        match_type="embedding",
        weight=1.0,
        seed_rank=rank,
        seeding_method="embedding",
        similarity=similarity,
    )


def test_r2a_prefers_coverage_then_unique_specificity():
    c1 = _candidate(
        "Doc::1",
        [_diag("a", "A", 0.9, 1), _diag("b", "B", 0.8, 1)],
        1,
    )
    c2 = _candidate("Doc::2", [_diag("a", "C", 0.99, 1)], 2)
    c3 = _candidate(
        "Doc::3",
        [_diag("a", "D", 0.7, 1), _diag("b", "E", 0.7, 1)],
        3,
    )
    specificity = {"A": 1.0, "B": 1.0, "C": 10.0, "D": 2.0, "E": 2.0}

    ranked = rank_specificity_coverage([c1, c2, c3], specificity)

    assert [item.section_uid for item in ranked] == ["Doc::3", "Doc::1", "Doc::2"]


def test_late_interaction_prefers_exact_concept_identity_before_casefold_fallback():
    terms = ["disease"]
    term_map = {"disease": np.asarray([1.0, 0.0], dtype=np.float32)}
    concept_map = {
        "DMD": ("DMD", np.asarray([0.0, 1.0], dtype=np.float32)),
        "dmd": ("dmd", np.asarray([1.0, 0.0], dtype=np.float32)),
    }

    gene_score, gene_evidence = concept_maxsim_score(
        terms, ["DMD"], term_map, concept_map
    )
    disease_score, disease_evidence = concept_maxsim_score(
        terms, ["dmd"], term_map, concept_map
    )

    assert gene_score == 0.0
    assert disease_score == 1.0
    assert gene_evidence[0]["best_concept"] == "DMD"
    assert disease_evidence[0]["best_concept"] == "dmd"
