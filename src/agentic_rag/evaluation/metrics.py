"""Pure, backend-independent metrics for ranked retrieval evaluation.

The same functions can evaluate document rankings and semantic-section
rankings. Inputs are hashable identifiers, for example normalized document IDs
or :class:`agentic_rag.evaluation.retrieval.SectionKey` objects.

Rankings are deduplicated while preserving first occurrence. This prevents
repeated chunks or repeated graph hits for the same semantic target from
inflating relevance counts.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Literal, TypeVar


RankingItem = TypeVar("RankingItem", bound=Hashable)
EvaluationLevel = Literal["document", "section"]
EvaluationView = Literal["clean", "end_to_end"]

DEFAULT_CUTOFFS: tuple[int, ...] = (1, 3, 5, 10)


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def _validate_cutoffs(cutoffs: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({_validate_k(k) for k in cutoffs}))
    if not normalized:
        raise ValueError("At least one cutoff is required")
    return normalized


def deduplicate_ranking(
    ranking: Iterable[RankingItem],
) -> list[RankingItem]:
    """Deduplicate a ranking while preserving the first-occurrence order."""

    unique: list[RankingItem] = []
    seen: set[RankingItem] = set()

    for item in ranking:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)

    return unique


def _prepare_inputs(
    gold: Iterable[RankingItem],
    ranking: Iterable[RankingItem],
) -> tuple[set[RankingItem], list[RankingItem]]:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("gold must contain at least one relevant item")

    return gold_set, deduplicate_ranking(ranking)


def hit_at_k(
    gold: Iterable[RankingItem],
    ranking: Iterable[RankingItem],
    k: int,
) -> float:
    """Return 1 when at least one relevant item appears in the top ``k``."""

    k = _validate_k(k)
    gold_set, ranked = _prepare_inputs(gold, ranking)
    return float(any(item in gold_set for item in ranked[:k]))


def precision_at_k(
    gold: Iterable[RankingItem],
    ranking: Iterable[RankingItem],
    k: int,
) -> float:
    """Return binary relevance precision at ``k``.

    The denominator is always the requested cutoff ``k``. Therefore a backend
    returning fewer than ``k`` unique results is penalized for the missing
    positions, which keeps comparisons across retrievers consistent.
    """

    k = _validate_k(k)
    gold_set, ranked = _prepare_inputs(gold, ranking)
    relevant = sum(item in gold_set for item in ranked[:k])
    return relevant / k


def recall_at_k(
    gold: Iterable[RankingItem],
    ranking: Iterable[RankingItem],
    k: int,
) -> float:
    """Return the fraction of all gold items found in the top ``k``."""

    k = _validate_k(k)
    gold_set, ranked = _prepare_inputs(gold, ranking)
    relevant = sum(item in gold_set for item in ranked[:k])
    return relevant / len(gold_set)


def reciprocal_rank_at_k(
    gold: Iterable[RankingItem],
    ranking: Iterable[RankingItem],
    k: int,
) -> float:
    """Return reciprocal rank of the first relevant item within top ``k``.

    This is a per-question reciprocal-rank value. Its mean over questions is
    MRR@k.
    """

    k = _validate_k(k)
    gold_set, ranked = _prepare_inputs(gold, ranking)

    for rank, item in enumerate(ranked[:k], start=1):
        if item in gold_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    gold: Iterable[RankingItem],
    ranking: Iterable[RankingItem],
    k: int,
) -> float:
    """Return nDCG@k using binary relevance for all gold items."""

    k = _validate_k(k)
    gold_set, ranked = _prepare_inputs(gold, ranking)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in gold_set
    )

    ideal_hits = min(len(gold_set), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def complete_recall_at_k(
    gold: Iterable[RankingItem],
    ranking: Iterable[RankingItem],
    k: int,
) -> float:
    """Return 1 only when every gold item appears within top ``k``."""

    return float(math.isclose(recall_at_k(gold, ranking, k), 1.0))


def compute_query_metrics(
    gold: Iterable[RankingItem],
    ranking: Iterable[RankingItem],
    *,
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
    rank_cutoff: int | None = None,
) -> dict[str, float]:
    """Compute all standard metrics for one query.

    ``reciprocal_rank@k`` is kept under its mathematically correct per-query
    name. During aggregation, its mean is additionally exposed as ``mrr@k``.
    """

    normalized_cutoffs = _validate_cutoffs(cutoffs)
    resolved_rank_cutoff = (
        _validate_k(rank_cutoff)
        if rank_cutoff is not None
        else max(normalized_cutoffs)
    )

    gold_set, ranked = _prepare_inputs(gold, ranking)
    metrics: dict[str, float] = {}

    for k in normalized_cutoffs:
        metrics[f"hit@{k}"] = hit_at_k(gold_set, ranked, k)
        metrics[f"precision@{k}"] = precision_at_k(gold_set, ranked, k)
        metrics[f"recall@{k}"] = recall_at_k(gold_set, ranked, k)
        metrics[f"complete_recall@{k}"] = complete_recall_at_k(
            gold_set,
            ranked,
            k,
        )

    metrics[f"reciprocal_rank@{resolved_rank_cutoff}"] = reciprocal_rank_at_k(
        gold_set,
        ranked,
        resolved_rank_cutoff,
    )
    metrics[f"ndcg@{resolved_rank_cutoff}"] = ndcg_at_k(
        gold_set,
        ranked,
        resolved_rank_cutoff,
    )

    return metrics


def aggregate_query_metrics(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Aggregate per-query metric dictionaries.

    All rows must contain exactly the same metric keys. The output includes
    macro statistics across questions. The mean reciprocal-rank value is also
    copied to the conventional aggregate name ``mrr@k``.
    """

    if not rows:
        raise ValueError("rows must contain at least one query metric mapping")

    expected_keys = set(rows[0].keys())
    if not expected_keys:
        raise ValueError("metric rows must not be empty")

    for index, row in enumerate(rows[1:], start=1):
        if set(row.keys()) != expected_keys:
            raise ValueError(
                "All metric rows must contain the same keys; "
                f"row 0 has {sorted(expected_keys)}, row {index} has "
                f"{sorted(row.keys())}"
            )

    per_metric: dict[str, dict[str, float | int]] = {}
    means: dict[str, float] = {}

    for name in sorted(expected_keys):
        values = [float(row[name]) for row in rows]
        mean = statistics.fmean(values)
        means[name] = mean
        per_metric[name] = {
            "count": len(values),
            "mean": mean,
            "median": statistics.median(values),
            "population_std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }

        if name.startswith("reciprocal_rank@"):
            cutoff = name.split("@", 1)[1]
            means[f"mrr@{cutoff}"] = mean

    return {
        "query_count": len(rows),
        "means": means,
        "statistics": per_metric,
    }


def get_evaluation_question_ids(
    coverage_artifact: Mapping[str, Any],
    *,
    level: EvaluationLevel,
    view: EvaluationView,
) -> list[str]:
    """Read one clean/end-to-end question set from an enriched gold artifact."""

    if level not in {"document", "section"}:
        raise ValueError(f"Unsupported evaluation level: {level!r}")
    if view not in {"clean", "end_to_end"}:
        raise ValueError(f"Unsupported evaluation view: {view!r}")

    coverage_summary = coverage_artifact.get("coverage_summary")
    if not isinstance(coverage_summary, Mapping):
        raise ValueError("Artifact must contain a 'coverage_summary' object")

    evaluation_sets = coverage_summary.get("evaluation_sets")
    if not isinstance(evaluation_sets, Mapping):
        raise ValueError(
            "Artifact coverage_summary must contain an 'evaluation_sets' object"
        )

    key = f"{level}_level_{view}_question_ids"
    question_ids = evaluation_sets.get(key)
    if not isinstance(question_ids, list) or not all(
        isinstance(item, str) and item.strip()
        for item in question_ids
    ):
        raise ValueError(
            f"Evaluation set {key!r} must be a list of non-empty strings"
        )

    return list(question_ids)


def filter_metric_rows_by_question_ids(
    rows: Sequence[Mapping[str, Any]],
    question_ids: Iterable[str],
) -> list[Mapping[str, Any]]:
    """Filter metric rows by question ID while preserving requested order.

    Each row must contain a ``question_id`` field. Duplicate rows or missing
    requested question IDs are treated as errors because either condition would
    make aggregate retrieval metrics ambiguous.
    """

    by_question: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("Every metric row must contain a non-empty question_id")
        if question_id in by_question:
            raise ValueError(f"Duplicate metric row for question {question_id!r}")
        by_question[question_id] = row

    requested = list(question_ids)
    missing = [question_id for question_id in requested if question_id not in by_question]
    if missing:
        raise ValueError(f"Missing metric rows for questions: {missing}")

    return [by_question[question_id] for question_id in requested]
