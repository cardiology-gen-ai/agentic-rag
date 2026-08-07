"""LangChain adapter for the modular knowledge-graph retriever.

The KG retrieval core intentionally remains framework-independent.  This module
only adapts ``ModularKGRetrievalPipeline.retrieve()`` results to LangChain
``Document`` objects.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from agentic_rag.kg.candidate_generators import KGCandidate


_KG_PROVENANCE_FIELDS = (
    "retrieval_unit_id",
    "section_view_schema_version",
    "section_view_role",
    "section_view_strategy",
    "section_view_aggregation",
    "is_aggregated",
    "content_owner_section_id",
    "source_section_ids",
    "source_chunk_ids",
    "represented_section_ids",
    "structural_context_section_ids",
    "absorbed_section_ids",
    "absorbed_source_section_ids",
)


class KGRetrievalError(RuntimeError):
    """Raised when the KG pipeline reports an execution/router failure."""

    def __init__(
        self,
        *,
        status: str,
        message: str | None = None,
        failed_stage: str | None = None,
    ) -> None:
        self.status = status
        self.failed_stage = failed_stage
        self.pipeline_error = message

        details = f"KG retrieval failed with status={status!r}"
        if failed_stage:
            details += f", stage={failed_stage!r}"
        if message:
            details += f": {message}"

        super().__init__(details)


class LangChainKGRetriever(BaseRetriever):
    """Expose a modular KG retrieval pipeline as a LangChain retriever.

    ``pipeline`` is deliberately typed as ``Any`` so this adapter does not
    impose Pydantic validation or inheritance requirements on the framework-
    independent retrieval core.

    By default pipeline failures raise ``KGRetrievalError``.  ``no_results`` is
    a valid retrieval outcome and returns an empty document list.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline: Any = Field(exclude=True)
    router_config: dict[str, Any] | None = None
    error_policy: Literal["raise", "empty"] = "raise"

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager  # BaseRetriever tracing is handled by LangChain itself.

        run = self.pipeline.retrieve(
            query,
            router_config=self.router_config,
        )
        return run_to_documents(
            run,
            error_policy=self.error_policy,
        )


def run_to_documents(
    run: Any,
    *,
    error_policy: Literal["raise", "empty"] = "raise",
) -> list[Document]:
    """Convert one modular KG run to ordered LangChain documents."""

    status = str(getattr(run, "status", "")).strip()

    if status == "no_results":
        return []

    if status != "success":
        if error_policy == "empty":
            return []
        if error_policy != "raise":
            raise ValueError(
                f"Unsupported KG adapter error_policy: {error_policy!r}"
            )

        raise KGRetrievalError(
            status=status or "unknown",
            message=_optional_text(getattr(run, "error", None)),
            failed_stage=_optional_text(
                getattr(run, "failed_stage", None)
            ),
        )

    results = list(getattr(run, "results", []) or [])
    return [
        candidate_to_document(
            candidate,
            run=run,
            position=position,
        )
        for position, candidate in enumerate(results, start=1)
    ]


def candidate_to_document(
    candidate: KGCandidate,
    *,
    run: Any,
    position: int,
) -> Document:
    """Convert one ranked KG candidate without losing retrieval provenance."""

    section = candidate.section
    section_data = section.model_dump(mode="json")

    metadata: dict[str, Any] = {
        "retriever": "kg",
        "kg_mode": getattr(run, "mode", None),
        "kg_status": getattr(run, "status", None),
        "kg_ranking_mode": getattr(run, "ranking_mode", None),
        "kg_expander": getattr(run, "expander_name", None),
        "kg_reranker": getattr(run, "reranker_name", None),
        "kg_rank": position,
        "kg_source": candidate.source,
        "kg_source_rank": candidate.source_rank,
        "kg_candidate_final_rank": candidate.final_rank,
        "kg_direct": candidate.direct,
        "kg_seed_uid": candidate.seed_uid,
        "kg_seed_rank": candidate.seed_rank,
        "kg_graph_distance": candidate.graph_distance,
        "kg_candidate_metadata": candidate.metadata,
        "section_uid": section.section_uid,
        "document_id": section.document_id,
        "section_id": section.section_id,
        "printed_section_id": section.printed_section_id,
        "title": section.title,
        "level": section.level,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "part_index": section.part_index,
        "part_count": section.part_count,
        "matched_concepts": list(section.matched_concepts),
        "matched_terms": list(section.matched_terms),
        "kg_score": section.score,
        "kg_score_type": section.score_type,
        "kg_section_rank": section.rank,
        "kg_scores": (
            section.scores.model_dump(mode="json")
            if section.scores is not None
            else None
        ),
        "kg_match_diagnostics": [
            item.model_dump(mode="json")
            for item in section.match_diagnostics
        ],
    }

    for field_name in _KG_PROVENANCE_FIELDS:
        if field_name in section_data:
            metadata[field_name] = section_data[field_name]

    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None
    }

    return Document(
        page_content=section.text,
        metadata=metadata,
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
