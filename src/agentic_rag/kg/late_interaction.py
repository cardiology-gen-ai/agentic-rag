"""Core Concept-aware late-interaction scoring for KG Section candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np


T = TypeVar("T")


def concept_maxsim_score(
    terms: Sequence[str],
    concept_names: Sequence[str],
    term_map: Mapping[str, np.ndarray],
    concept_map: Mapping[str, tuple[str, np.ndarray]],
) -> tuple[float, list[dict[str, Any]]]:
    """Score a Section by mean query-term MaxSim over its local Concepts.

    The function is intentionally free of gold labels, graph reads and
    candidate selection. It is the canonical implementation of the LI policy
    historically used by the v14 fixed-pool analyzers.
    """

    if not terms:
        raise ValueError("LI requires at least one router term")

    resolved: list[tuple[str, np.ndarray]] = []
    for name in concept_names:
        exact_name = str(name)
        # Exact Concept identity is authoritative.  The casefold fallback keeps
        # historical v14 artifacts readable without collapsing new exact-case
        # maps such as DMD (gene) versus dmd (disease).
        item = concept_map.get(exact_name)
        if item is None:
            item = concept_map.get(exact_name.casefold())
        if item is not None:
            resolved.append(item)

    if not resolved:
        evidence = [
            {
                "query_term": str(term),
                "best_concept": None,
                "max_similarity": -1.0,
            }
            for term in terms
        ]
        return -1.0, evidence

    names = [item[0] for item in resolved]
    matrix = np.vstack([item[1] for item in resolved]).astype(np.float32)
    per_term: list[float] = []
    evidence: list[dict[str, Any]] = []

    for term in terms:
        key = str(term).strip().casefold()
        if key not in term_map:
            raise KeyError(
                "Router term missing from LI embedding artifact: "
                f"{term!r}"
            )
        similarities = matrix @ term_map[key]
        best_idx = max(
            range(len(names)),
            key=lambda index: (float(similarities[index]), -index),
        )
        best = float(similarities[best_idx])
        per_term.append(best)
        evidence.append(
            {
                "query_term": str(term),
                "best_concept": names[best_idx],
                "max_similarity": best,
            }
        )

    return float(sum(per_term) / len(per_term)), evidence


def rank_by_concept_maxsim(
    universe: Sequence[T],
    scores: Mapping[str, float],
    uid_fn: Callable[[T], str],
) -> list[T]:
    """Rank a fixed candidate universe by LI score with stable tie-breaking."""

    indexed = list(enumerate(universe, start=1))
    return [
        candidate
        for _, candidate in sorted(
            indexed,
            key=lambda item: (
                -float(scores[uid_fn(item[1])]),
                item[0],
                uid_fn(item[1]),
            ),
        )
    ]
