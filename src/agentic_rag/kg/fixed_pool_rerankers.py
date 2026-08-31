"""Canonical rerankers for frozen KG Section candidate pools.

The functions in this module are deliberately free of gold labels and graph
writes.  They operate on a fixed :class:`KGCandidate` universe and therefore
make the experimental contract explicit: candidate membership is determined
upstream; rerankers may change only the ordering.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentic_rag.kg.candidate_generators import KGCandidate, deduplicate_candidates


def load_specificity_csv(path: Path) -> dict[str, float]:
    """Load exact-Concept specificity values exported by the canonical exporter."""

    values: dict[str, float] = {}
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("concept_name") or "").strip()
            if not name:
                continue
            if name in values:
                raise ValueError(f"Duplicate specificity row for Concept {name!r}")
            value = float(row["idf"])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite specificity for Concept {name!r}")
            values[name] = value
    if not values:
        raise ValueError(f"Specificity CSV is empty: {resolved}")
    return values



def validate_specificity_subset(
    specificity: Mapping[str, float],
    concept_universe: Mapping[str, Any] | Sequence[str],
) -> dict[str, Any]:
    """Validate the corpus-specificity domain against a frozen Concept universe.

    Specificity is exported only for Concepts that MENTION at least one
    retrieval Section.  A frozen embedding catalogue may therefore contain
    additional Concepts that are not assigned an IDF value.  This is valid as
    long as specificity introduces no Concept outside the frozen catalogue.

    The actual reranker remains strict: :func:`specificity_features` raises if
    a best-evidence Concept used by a candidate lacks specificity.  Hence this
    universe check allows catalogue-only Concepts without silently assigning
    them an arbitrary score.
    """

    concept_names = (
        set(str(name) for name in concept_universe.keys())
        if isinstance(concept_universe, Mapping)
        else set(str(name) for name in concept_universe)
    )
    specificity_names = set(str(name) for name in specificity.keys())
    extra = sorted(specificity_names - concept_names, key=lambda x: (x.casefold(), x))
    if extra:
        raise RuntimeError(
            "Specificity contains Concepts outside the frozen embedding universe: "
            + repr(extra[:20])
        )
    excluded = sorted(concept_names - specificity_names, key=lambda x: (x.casefold(), x))
    return {
        "specificity_concept_count": len(specificity_names),
        "concept_universe_count": len(concept_names),
        "specificity_excluded_concept_count": len(excluded),
        "specificity_excluded_concept_examples": excluded[:20],
    }

def specificity_features(
    candidate: KGCandidate,
    specificity: Mapping[str, float],
) -> dict[str, float | int]:
    """Return the historical R2a-U features for one candidate.

    This is the typed/core equivalent of the feature computation originally
    implemented in ``scripts/analyze_v11_specificity_ranking.py``.
    """

    by_term: dict[str, list[Any]] = defaultdict(list)
    for diagnostic in candidate.section.match_diagnostics:
        term = str(diagnostic.query_term or "").strip().casefold()
        if term:
            by_term[term].append(diagnostic)

    best_by_term: dict[str, Any] = {}
    for term, diagnostics in by_term.items():
        scored: list[tuple[float, int, int, str, Any]] = []
        for diagnostic in diagnostics:
            similarity = _evidence_similarity(diagnostic)
            if similarity is None:
                continue
            is_graph = 1 if getattr(diagnostic, "evidence_source", None) else 0
            seed_rank = int(getattr(diagnostic, "seed_rank", None) or 10**9)
            concept_name = str(getattr(diagnostic, "concept_name", None) or "")
            scored.append(
                (similarity, -is_graph, -seed_rank, concept_name, diagnostic)
            )
        if scored:
            scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
            best_by_term[term] = scored[0][4]

    concept_names: set[str] = set()
    missing: list[str] = []
    graph_terms = 0
    semantic_quality = 0.0

    for diagnostic in best_by_term.values():
        semantic_quality += float(_evidence_similarity(diagnostic) or 0.0)
        if getattr(diagnostic, "evidence_source", None):
            graph_terms += 1
        name = str(getattr(diagnostic, "concept_name", None) or "").strip()
        if not name:
            continue
        if name not in specificity:
            missing.append(name)
        else:
            concept_names.add(name)

    if missing:
        raise KeyError(
            "Best-evidence Concepts missing from specificity CSV: "
            + repr(sorted(set(missing), key=lambda x: (x.casefold(), x))[:20])
        )

    return {
        "coverage": len(best_by_term),
        "semantic_quality": semantic_quality,
        "unique_specificity": sum(float(specificity[name]) for name in concept_names),
        "unique_concept_count": len(concept_names),
        "local_best_terms": len(best_by_term) - graph_terms,
        "graph_best_terms": graph_terms,
    }


def rank_specificity_coverage(
    candidates: Sequence[KGCandidate],
    specificity: Mapping[str, float],
) -> list[KGCandidate]:
    """Rank a fixed pool using the canonical R2a-U policy.

    Order: coverage DESC, unique specificity DESC, original candidate rank,
    Section UID.  Candidate membership is preserved exactly.
    """

    unique = deduplicate_candidates(candidates)
    indexed = [
        (index, candidate, specificity_features(candidate, specificity))
        for index, candidate in enumerate(unique, start=1)
    ]
    ordered = [
        candidate
        for _, candidate, _ in sorted(
            indexed,
            key=lambda item: (
                -int(item[2]["coverage"]),
                -float(item[2]["unique_specificity"]),
                item[0],
                item[1].section_uid,
            ),
        )
    ]
    return _assign_final_ranks(ordered)


def rerank_bm25plus(
    candidates: Sequence[KGCandidate],
    question: str,
) -> tuple[list[KGCandidate], list[dict[str, Any]]]:
    """Rerank a fixed KG Section pool with the shared BM25Plus implementation."""

    from agentic_rag.utils.bm25 import build_bm25_dict, rank_bm25_documents

    pool = deduplicate_candidates(candidates)
    if not pool:
        return [], []

    query = str(question).strip()
    if not query:
        raise ValueError("BM25Plus requires a non-empty full question")

    before_uids = [candidate.section_uid for candidate in pool]
    documents = [
        _candidate_to_document(candidate, original_rank=rank)
        for rank, candidate in enumerate(pool, start=1)
    ]
    bm25 = build_bm25_dict(documents)
    ranked_documents = rank_bm25_documents(bm25, query, k=len(documents))

    by_uid = {candidate.section_uid: candidate for candidate in pool}
    after_uids = [str(doc.metadata.get("record_id") or "").strip() for doc in ranked_documents]

    if len(after_uids) != len(before_uids) or set(after_uids) != set(before_uids):
        raise RuntimeError("BM25Plus changed fixed candidate-pool membership")
    if len(set(after_uids)) != len(after_uids):
        raise RuntimeError("BM25Plus introduced duplicate candidate identities")

    ranking = _assign_final_ranks([by_uid[uid] for uid in after_uids])
    original_rank = {uid: rank for rank, uid in enumerate(before_uids, start=1)}
    details: list[dict[str, Any]] = []

    for rank, document in enumerate(ranked_documents, start=1):
        metadata = document.metadata
        uid = str(metadata["record_id"])
        details.append(
            {
                "section_uid": uid,
                "original_candidate_rank": original_rank[uid],
                "bm25plus_rank": rank,
                "bm25plus_score": metadata.get("raw_score"),
                "bm25_query_token_count": metadata.get("bm25_query_token_count"),
                "bm25_query_token_overlap_count": metadata.get(
                    "bm25_query_token_overlap_count"
                ),
                "bm25_has_query_token_overlap": metadata.get(
                    "bm25_has_query_token_overlap"
                ),
                "bm25_matched_query_tokens": list(
                    metadata.get("bm25_matched_query_tokens") or []
                ),
            }
        )

    return ranking, details



def rank_reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[KGCandidate]],
    *,
    constant: int = 60,
) -> tuple[list[KGCandidate], list[dict[str, Any]]]:
    """Fuse equal-weight fixed-pool rankings with Reciprocal Rank Fusion.

    The function is intentionally gold-free and score-scale agnostic.  Every
    parent ranking must contain exactly the same Section candidate universe.
    RRF uses equal weights and a fixed positive rank constant; no fitted or
    benchmark-derived parameters are accepted here.

    Ranking order is: RRF score DESC, best parent rank ASC, Section UID ASC.
    This mirrors the canonical evidence-level RRF tie style while remaining
    symmetric with respect to the parent ranking names.
    """

    if constant <= 0:
        raise ValueError("RRF constant must be a positive integer")
    if len(rankings) < 2:
        raise ValueError("RRF requires at least two parent rankings")

    parent_names = [str(name).strip() for name in rankings]
    if any(not name for name in parent_names):
        raise ValueError("RRF parent names must be non-empty")
    if len(set(parent_names)) != len(parent_names):
        raise ValueError("RRF parent names must be unique")

    rank_maps: dict[str, dict[str, int]] = {}
    candidate_by_uid: dict[str, KGCandidate] = {}
    reference_uids: set[str] | None = None

    for parent, ranking in rankings.items():
        ordered = deduplicate_candidates(ranking)
        if len(ordered) != len(ranking):
            raise RuntimeError(f"RRF parent {parent!r} contains duplicate candidates")
        uids = [candidate.section_uid for candidate in ordered]
        uid_set = set(uids)
        if reference_uids is None:
            reference_uids = uid_set
            candidate_by_uid = {candidate.section_uid: candidate for candidate in ordered}
        elif uid_set != reference_uids:
            missing = sorted(reference_uids - uid_set)
            unexpected = sorted(uid_set - reference_uids)
            raise RuntimeError(
                f"RRF parent {parent!r} changed fixed candidate-pool membership; "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        rank_maps[parent] = {uid: rank for rank, uid in enumerate(uids, start=1)}

    if not reference_uids:
        return [], []

    fused_rows: list[tuple[float, int, int, str, KGCandidate, dict[str, int]]] = []
    for uid in sorted(reference_uids):
        parent_ranks = {parent: rank_maps[parent][uid] for parent in parent_names}
        score = sum(1.0 / (constant + rank) for rank in parent_ranks.values())
        rank_sum = sum(parent_ranks.values())
        best_rank = min(parent_ranks.values())
        fused_rows.append(
            (score, rank_sum, best_rank, uid, candidate_by_uid[uid], parent_ranks)
        )

    fused_rows.sort(key=lambda row: (-row[0], row[2], row[3]))
    ranking = _assign_final_ranks([row[4] for row in fused_rows])
    details: list[dict[str, Any]] = []
    for fused_rank, row in enumerate(fused_rows, start=1):
        score, rank_sum, best_rank, uid, _, parent_ranks = row
        detail: dict[str, Any] = {
            "section_uid": uid,
            "rrf_rank": fused_rank,
            "rrf_score": score,
            "rrf_constant": constant,
            "parent_rank_sum": rank_sum,
            "best_parent_rank": best_rank,
        }
        for parent in parent_names:
            detail[f"{parent}_rank"] = parent_ranks[parent]
        details.append(detail)

    return ranking, details

def _candidate_to_document(candidate: KGCandidate, *, original_rank: int):
    from langchain_core.documents import Document
    section = candidate.section
    if not section.page_content.strip():
        raise ValueError(f"{section.section_uid}: empty BM25Plus page_content")
    return Document(
        page_content=section.page_content,
        metadata={
            "record_id": section.section_uid,
            "section_uid": section.section_uid,
            "document_id": section.document_id,
            "section_id": section.section_id,
            "printed_section_id": section.printed_section_id,
            "title": section.title,
            "original_candidate_rank": original_rank,
        },
    )


def _evidence_similarity(diagnostic: Any) -> float | None:
    raw = getattr(diagnostic, "similarity", None)
    evidence_source = getattr(diagnostic, "evidence_source", None)
    if raw is None and evidence_source in {
        "direct_local_artifact",
        "ontology_bridge_artifact",
    }:
        raw = getattr(diagnostic, "lexical_weight", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None

    term = str(getattr(diagnostic, "query_term", None) or "").strip().casefold()
    concept = str(getattr(diagnostic, "concept_name", None) or "").strip().casefold()
    return 1.0 if term and concept and term == concept else value


def _assign_final_ranks(candidates: Sequence[KGCandidate]) -> list[KGCandidate]:
    return [
        candidate.model_copy(update={"final_rank": rank})
        for rank, candidate in enumerate(candidates, start=1)
    ]
