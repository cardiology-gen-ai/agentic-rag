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
    coverage_at_cutoff,
)
from agentic_rag.evaluation.trace import (
    TraceTextMode,
    build_retrieval_trace,
)

__all__ = [
    "CoverageMetrics",
    "CutoffMetrics",
    "EvaluationQuestion",
    "EvidenceNormalizationResult",
    "EvidenceSection",
    "RetrievedEvidence",
    "TraceTextMode",
    "build_retrieval_trace",
    "compute_coverage_metrics",
    "coverage_at_cutoff",
    "load_evaluation_questions",
    "normalize_retrieved_documents",
    "parse_section_id",
]
