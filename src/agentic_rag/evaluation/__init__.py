"""Retrieval evaluation utilities."""

from agentic_rag.evaluation.dataset import (
    EvaluationQuestion,
    load_evaluation_questions,
    parse_section_id,
)
from agentic_rag.evaluation.evidence import (
    EvidenceNormalizationResult,
    EvidenceSection,
    RetrievedEvidence,
    normalize_retrieved_documents,
)
from agentic_rag.evaluation.metrics import (
    CoverageMetrics,
    CutoffMetrics,
    compute_coverage_metrics,
)

__all__ = [
    "CoverageMetrics",
    "CutoffMetrics",
    "EvaluationQuestion",
    "EvidenceNormalizationResult",
    "EvidenceSection",
    "RetrievedEvidence",
    "compute_coverage_metrics",
    "load_evaluation_questions",
    "normalize_retrieved_documents",
    "parse_section_id",
]
