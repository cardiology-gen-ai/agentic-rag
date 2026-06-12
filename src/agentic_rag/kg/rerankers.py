"""Optional reranking stages for modular KG retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from agentic_rag.kg.candidate_generators import (
    KGCandidate,
    deduplicate_candidates,
)


class CandidateRerankerProtocol(Protocol):
    """Common interface for deterministic candidate rerankers."""

    name: str

    def rerank(
        self,
        candidates: Sequence[KGCandidate],
        *,
        top_k: int,
    ) -> list[KGCandidate]: ...


class NoOpReranker:
    """Preserve generator order after Section-level deduplication."""

    name = "none"

    def rerank(
        self,
        candidates: Sequence[KGCandidate],
        *,
        top_k: int,
    ) -> list[KGCandidate]:
        validated_top_k = _validate_top_k(top_k)
        ordered = deduplicate_candidates(candidates)[:validated_top_k]
        return _assign_final_ranks(ordered)


class SeedRoundRobinReranker:
    """Interleave direct seeds with their hierarchical descendants.

    The method has no learned or manually tuned relevance weights. Seeds are
    visited in their original rank order. For each seed, the direct Section is
    emitted first, followed by a bounded number of descendants ordered by
    ``HAS_CHILD`` distance and Section UID. This is intended as a transparent
    hierarchy-expansion ablation, not as the final weighted graph reranker.
    """

    name = "seed_round_robin"

    def __init__(self, *, descendants_per_seed: int = 3) -> None:
        try:
            normalized = int(descendants_per_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "descendants_per_seed must be an integer"
            ) from exc
        if normalized < 0 or normalized > 100:
            raise ValueError(
                "descendants_per_seed must be between 0 and 100"
            )
        self.descendants_per_seed = normalized

    def rerank(
        self,
        candidates: Sequence[KGCandidate],
        *,
        top_k: int,
    ) -> list[KGCandidate]:
        validated_top_k = _validate_top_k(top_k)
        unique = deduplicate_candidates(candidates)

        direct = sorted(
            (item for item in unique if item.direct),
            key=lambda item: (item.source_rank, item.section_uid),
        )
        descendants_by_seed: dict[str, list[KGCandidate]] = defaultdict(list)
        for item in unique:
            if item.direct or not item.seed_uid:
                continue
            descendants_by_seed[item.seed_uid].append(item)

        for items in descendants_by_seed.values():
            items.sort(
                key=lambda item: (
                    item.graph_distance or 10**9,
                    item.section_uid,
                )
            )

        output: list[KGCandidate] = []
        seen: set[str] = set()

        for seed in direct:
            _append_unique(output, seen, seed)
            for descendant in descendants_by_seed.get(seed.section_uid, [
            ])[: self.descendants_per_seed]:
                _append_unique(output, seen, descendant)
                if len(output) >= validated_top_k:
                    return _assign_final_ranks(output)

            if len(output) >= validated_top_k:
                return _assign_final_ranks(output)

        # Preserve descendants whose seed was not present after deduplication.
        remaining = sorted(
            (
                item
                for item in unique
                if not item.direct and item.section_uid not in seen
            ),
            key=lambda item: (
                item.seed_rank or 10**9,
                item.graph_distance or 10**9,
                item.section_uid,
            ),
        )
        for item in remaining:
            _append_unique(output, seen, item)
            if len(output) >= validated_top_k:
                break

        return _assign_final_ranks(output[:validated_top_k])


def _append_unique(
    output: list[KGCandidate],
    seen: set[str],
    candidate: KGCandidate,
) -> None:
    if candidate.section_uid in seen:
        return
    seen.add(candidate.section_uid)
    output.append(candidate)


def _assign_final_ranks(
    candidates: Sequence[KGCandidate],
) -> list[KGCandidate]:
    return [
        candidate.model_copy(update={"final_rank": rank})
        for rank, candidate in enumerate(candidates, start=1)
    ]


def _validate_top_k(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc
    if normalized < 1 or normalized > 100:
        raise ValueError("top_k must be between 1 and 100")
    return normalized
