"""Document-aware diagnostics for multi-document retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from agentic_rag.evaluation.evidence import EvidenceSection, RetrievedEvidence


@dataclass(frozen=True)
class DocumentCutoffMetrics:
    k: int
    hit: float
    recall: float
    complete_recall: float
    wrong_document_fraction: float
    wrong_document_count: int
    retrieved_unit_count: int
    unique_document_count: int
    covered_gold_document_count: int


@dataclass(frozen=True)
class DocumentCoverageMetrics:
    cutoffs: tuple[DocumentCutoffMetrics, ...]
    first_gold_document_rank: int | None
    gold_documents: frozenset[str]

    def at(self, k: int) -> DocumentCutoffMetrics:
        for item in self.cutoffs:
            if item.k == k:
                return item
        raise KeyError(f"Cutoff {k} was not computed")


def compute_document_coverage_metrics(
    ranking: Sequence[RetrievedEvidence],
    gold_sections: Iterable[EvidenceSection],
    *,
    cutoffs: Sequence[int] = (5, 10, 20),
) -> DocumentCoverageMetrics:
    """Compute document reach/coverage diagnostics over evidence ranking.

    Document recall uses unique gold document identifiers.
    ``wrong_document_fraction`` divides non-gold evidence units by the number
    of actually retrieved units in the top-k prefix; it is therefore a
    diagnostic rather than an IR precision metric.
    """

    normalized_cutoffs = _validate_cutoffs(cutoffs)
    gold_documents = frozenset(section.document_id for section in gold_sections)
    if not gold_documents:
        raise ValueError("gold_sections must contain at least one document")

    first_rank = next(
        (
            rank
            for rank, evidence in enumerate(ranking, start=1)
            if evidence.document_id in gold_documents
        ),
        None,
    )

    metrics: list[DocumentCutoffMetrics] = []
    for k in normalized_cutoffs:
        top_k = ranking[:k]
        observed = [evidence.document_id for evidence in top_k]
        covered = set(observed) & gold_documents
        wrong_count = sum(doc_id not in gold_documents for doc_id in observed)
        retrieved_count = len(observed)
        recall = len(covered) / len(gold_documents)

        metrics.append(
            DocumentCutoffMetrics(
                k=k,
                hit=float(bool(covered)),
                recall=recall,
                complete_recall=float(len(covered) == len(gold_documents)),
                wrong_document_fraction=(
                    wrong_count / retrieved_count if retrieved_count else 0.0
                ),
                wrong_document_count=wrong_count,
                retrieved_unit_count=retrieved_count,
                unique_document_count=len(set(observed)),
                covered_gold_document_count=len(covered),
            )
        )

    return DocumentCoverageMetrics(
        cutoffs=tuple(metrics),
        first_gold_document_rank=first_rank,
        gold_documents=gold_documents,
    )


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(cutoffs)
    if not normalized:
        raise ValueError("cutoffs must not be empty")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("cutoffs must be unique and increasing")
    if any(k < 1 for k in normalized):
        raise ValueError("cutoffs must be >= 1")
    return normalized
