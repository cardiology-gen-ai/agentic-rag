"""Coverage-aware retrieval metrics.

A retrieval unit is relevant when at least one of its covered original
guideline sections intersects the gold section set. A unit counts once for
precision even when it covers several gold sections. Recall is section-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from agentic_rag.evaluation.evidence import EvidenceSection, RetrievedEvidence


@dataclass(frozen=True)
class CutoffMetrics:
    """Metrics computed at one ranking cutoff."""

    k: int
    hit: float
    precision: float
    recall: float
    complete_recall: float
    reciprocal_rank: float
    relevant_unit_count: int
    covered_gold_count: int


@dataclass(frozen=True)
class CoverageMetrics:
    """Coverage-aware metrics for one query."""

    cutoffs: tuple[CutoffMetrics, ...]
    first_relevant_rank: int | None
    found_gold_sections: frozenset[EvidenceSection]
    missing_gold_sections: frozenset[EvidenceSection]

    def at(self, k: int) -> CutoffMetrics:
        """Return metrics for cutoff ``k``."""

        for item in self.cutoffs:
            if item.k == k:
                return item
        raise KeyError(f"Cutoff {k} was not computed")


def compute_coverage_metrics(
    ranking: Sequence[RetrievedEvidence],
    gold_sections: Iterable[EvidenceSection],
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10, 20),
) -> CoverageMetrics:
    """Compute coverage-aware metrics for one ranked result list.

    Precision@k uses ``k`` as denominator even when fewer than ``k`` normalized
    evidence units are available. This makes candidate-pool exhaustion visible
    instead of silently changing the metric denominator.

    A hierarchical retrieval unit that covers several gold sections:
    - counts as one relevant unit for precision;
    - covers every intersecting gold section for recall;
    - occupies one ranking position.
    """

    normalized_cutoffs = _validate_cutoffs(cutoffs)
    gold = frozenset(gold_sections)

    if not gold:
        raise ValueError("gold_sections must contain at least one section")

    first_relevant_rank = _first_relevant_rank(ranking, gold)
    cutoff_metrics: list[CutoffMetrics] = []

    for k in normalized_cutoffs:
        found_at_k, _ = coverage_at_cutoff(
            ranking,
            gold,
            k=k,
        )
        top_k = ranking[:k]
        relevant_unit_count = sum(
            bool(evidence.covered_sections & gold)
            for evidence in top_k
        )
        covered_count = len(found_at_k)
        recall = covered_count / len(gold)

        cutoff_metrics.append(
            CutoffMetrics(
                k=k,
                hit=float(relevant_unit_count > 0),
                precision=relevant_unit_count / k,
                recall=recall,
                complete_recall=float(covered_count == len(gold)),
                reciprocal_rank=(
                    1.0 / first_relevant_rank
                    if (
                        first_relevant_rank is not None
                        and first_relevant_rank <= k
                    )
                    else 0.0
                ),
                relevant_unit_count=relevant_unit_count,
                covered_gold_count=covered_count,
            )
        )

    found_in_pool, missing_in_pool = coverage_at_cutoff(
        ranking,
        gold,
        k=len(ranking),
    )

    return CoverageMetrics(
        cutoffs=tuple(cutoff_metrics),
        first_relevant_rank=first_relevant_rank,
        found_gold_sections=found_in_pool,
        missing_gold_sections=missing_in_pool,
    )


def coverage_at_cutoff(
    ranking: Sequence[RetrievedEvidence],
    gold_sections: Iterable[EvidenceSection],
    *,
    k: int,
) -> tuple[
    frozenset[EvidenceSection],
    frozenset[EvidenceSection],
]:
    """Return found and missing gold sections within the first ``k`` units."""

    if k < 0:
        raise ValueError("k must be >= 0")

    gold = frozenset(gold_sections)
    if not gold:
        raise ValueError("gold_sections must contain at least one section")

    found: set[EvidenceSection] = set()

    for evidence in ranking[:k]:
        found.update(evidence.covered_sections & gold)

    found_frozen = frozenset(found)
    return found_frozen, gold - found_frozen


def _first_relevant_rank(
    ranking: Sequence[RetrievedEvidence],
    gold: frozenset[EvidenceSection],
) -> int | None:
    for rank, evidence in enumerate(ranking, start=1):
        if evidence.covered_sections & gold:
            return rank
    return None


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    if not cutoffs:
        raise ValueError("cutoffs must not be empty")

    normalized = tuple(int(k) for k in cutoffs)

    if any(k < 1 for k in normalized):
        raise ValueError("every cutoff must be >= 1")

    if len(set(normalized)) != len(normalized):
        raise ValueError("cutoffs must not contain duplicates")

    if tuple(sorted(normalized)) != normalized:
        raise ValueError("cutoffs must be strictly increasing")

    return normalized
