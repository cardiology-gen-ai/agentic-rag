"""Pure helpers for gold-evidence retrieval funnel diagnostics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def reconstruct_preselection(
    section_concepts: Mapping[str, set[str]],
    seeds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct canonical direct-MENTIONS preselection ordering."""
    by_term: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for seed in seeds:
        by_term[str(seed["term_key"])].append(
            (str(seed["concept_name"]), int(seed["downstream_seed_rank"]))
        )
    scored: list[dict[str, Any]] = []
    for uid, concepts in section_concepts.items():
        best_ranks: list[int] = []
        matched_terms: list[str] = []
        for term_key, candidates in by_term.items():
            ranks = [rank for concept, rank in candidates if concept in concepts]
            if ranks:
                matched_terms.append(term_key)
                best_ranks.append(min(ranks))
        if not best_ranks:
            continue
        scored.append(
            {
                "section_uid": uid,
                "matched_term_count": len(set(matched_terms)),
                "seed_rank_sum": sum(best_ranks),
                "best_seed_rank": min(best_ranks),
            }
        )
    scored.sort(
        key=lambda x: (
            -x["matched_term_count"],
            x["seed_rank_sum"],
            x["best_seed_rank"],
            x["section_uid"],
        )
    )
    for rank, row in enumerate(scored, 1):
        row["preselection_rank"] = rank
    return scored


def failure_stage(
    *,
    graph_uid: str | None,
    concept_count: int,
    topm: bool,
    selected: bool,
    pre_rank: int | None,
    budget: int,
    final_rank_value: int | None,
) -> str:
    if graph_uid is None:
        return "G0_RETRIEVAL_VIEW_COVERAGE"
    if concept_count == 0:
        return "G1_GOLD_SECTION_HAS_NO_CONCEPTS"
    if not topm:
        return "G2_ROUTER_OR_TOPM_NEIGHBOURHOOD"
    if not selected:
        return "G3_SEED_SELECTION"
    if pre_rank is None:
        return "G4_DIRECT_MENTIONS_INCONSISTENCY"
    if pre_rank > budget:
        return "G5_PRESELECTION_BUDGET"
    if final_rank_value is None:
        return "G6_FINAL_RERANKING"
    return "SUCCESS"



def reconstruct_preselection_variants(
    section_concepts: Mapping[str, set[str]],
    seeds: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Rank the complete direct-MENTIONS union under diagnostic policies.

    The policies are deliberately parameter-free and reuse only evidence
    already present in the frozen semantic seeds:

    ``current``
        Canonical preliminary graph ordering.
    ``semantic_before_cutoff``
        Apply the existing similarity-weighted channel score to the complete
        direct union *before* truncation. This is the exact ordering that the
        current wrapper would use if it were allowed to see every generated
        Section.
    ``count_removed``
        Remove matched-term coverage priority and rank by mean best seed rank
        per matched term, avoiding the raw seed-rank-sum penalty on Sections
        supported by more terms.
    ``single_best``
        Rank only by the strongest individual semantic seed similarity.

    Gold is not an input to any policy.
    """

    by_term: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for seed in seeds:
        similarity = seed.get("similarity", seed.get("q0_score", 0.0))
        by_term[str(seed["term_key"])].append(
            (
                str(seed["concept_name"]),
                int(seed["downstream_seed_rank"]),
                float(similarity if similarity is not None else 0.0),
            )
        )

    rows: list[dict[str, Any]] = []
    for uid, concepts in section_concepts.items():
        best_ranks: list[int] = []
        best_similarities: list[float] = []
        matched_terms: list[str] = []
        for term_key, candidates in by_term.items():
            matches = [item for item in candidates if item[0] in concepts]
            if not matches:
                continue
            matched_terms.append(term_key)
            best_ranks.append(min(item[1] for item in matches))
            best_similarities.append(max(item[2] for item in matches))
        if not best_ranks:
            continue
        term_count = len(set(matched_terms))
        rows.append(
            {
                "section_uid": uid,
                "matched_term_count": term_count,
                "seed_rank_sum": sum(best_ranks),
                "best_seed_rank": min(best_ranks),
                "mean_best_seed_rank": sum(best_ranks) / term_count,
                "semantic_similarity_sum": sum(best_similarities),
                "best_seed_similarity": max(best_similarities),
            }
        )

    current = sorted(
        rows,
        key=lambda x: (
            -x["matched_term_count"],
            x["seed_rank_sum"],
            x["best_seed_rank"],
            x["section_uid"],
        ),
    )
    current_rank = {row["section_uid"]: i for i, row in enumerate(current, 1)}

    variants = {
        "current": current,
        "semantic_before_cutoff": sorted(
            rows,
            key=lambda x: (
                -x["semantic_similarity_sum"],
                -x["matched_term_count"],
                current_rank[x["section_uid"]],
                x["section_uid"],
            ),
        ),
        "count_removed": sorted(
            rows,
            key=lambda x: (
                x["mean_best_seed_rank"],
                x["best_seed_rank"],
                x["section_uid"],
            ),
        ),
        "single_best": sorted(
            rows,
            key=lambda x: (
                -x["best_seed_similarity"],
                x["best_seed_rank"],
                x["section_uid"],
            ),
        ),
    }

    output: dict[str, list[dict[str, Any]]] = {}
    for name, ordered in variants.items():
        enriched: list[dict[str, Any]] = []
        for rank, row in enumerate(ordered, 1):
            enriched.append({**row, "preselection_rank": rank})
        output[name] = enriched
    return output
