"""Evidence-level reciprocal-rank fusion for retrieval evaluation.

This module intentionally does not use the legacy ``FusionStrategyFactory``.
Fixed chunks are normalized to evidence units before fusion, so overlapping
chunks from the same guideline section cannot receive duplicate RRF votes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document

from agentic_rag.evaluation.evidence import (
    EvidenceNormalizationResult,
    RetrievedEvidence,
    normalize_retrieved_documents,
)


_FIXED_SOURCE_TYPES = frozenset({"fixed_chunks", "fixed"})
_HIERARCHICAL_SOURCE_TYPES = frozenset(
    {"hierarchical_section_view", "hierarchical"}
)


@dataclass(frozen=True)
class ComponentRanking:
    """One normalized component ranking used by RRF."""

    name: str
    documents: tuple[Document, ...]
    normalized: EvidenceNormalizationResult
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ComponentRanking.name must not be empty")
        if self.weight <= 0:
            raise ValueError("ComponentRanking.weight must be > 0")
        if self.normalized.raw_result_count != len(self.documents):
            raise ValueError(
                "ComponentRanking documents must match the normalized raw "
                "result count"
            )


@dataclass(frozen=True)
class FusionResult:
    """Synthetic documents and normalized evidence ranking produced by RRF."""

    documents: tuple[Document, ...]
    normalized: EvidenceNormalizationResult
    provenance: tuple[dict[str, Any], ...]


def build_component_ranking(
    name: str,
    documents: Sequence[Document],
    *,
    source_type: str | None = None,
    weight: float = 1.0,
) -> ComponentRanking:
    """Copy and normalize one backend ranking before fusion."""

    copied = tuple(_copy_document(document) for document in documents)
    normalized = normalize_retrieved_documents(
        copied,
        source_type=source_type,
    )
    return ComponentRanking(
        name=name,
        documents=copied,
        normalized=normalized,
        weight=weight,
    )


def reciprocal_rank_fuse_components(
    components: Sequence[ComponentRanking],
    *,
    rrf_k: int = 60,
    top_k: int = 50,
) -> FusionResult:
    """Fuse normalized evidence rankings using weighted RRF.

    Each evidence unit receives at most one contribution from each component.
    Scores use ``weight / (rrf_k + rank)`` with ranks starting from one.
    Ties are resolved by best component rank and retrieval-unit identifier.
    """

    if len(components) < 2:
        raise ValueError("RRF requires at least two component rankings")
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    names = [component.name for component in components]
    if len(set(names)) != len(names):
        raise ValueError("RRF component names must be unique")

    source_types = {component.normalized.source_type for component in components}
    if len(source_types) != 1:
        raise ValueError(
            "All RRF components must use the same source representation; "
            f"observed={sorted(source_types)!r}"
        )
    source_type = next(iter(source_types))

    record_maps = {
        component.name: _documents_by_record_id(component.documents)
        for component in components
    }
    accumulator: dict[tuple[str, str], dict[str, Any]] = {}

    for component_order, component in enumerate(components):
        for rank, evidence in enumerate(component.normalized.evidence, start=1):
            key = (evidence.document_id, evidence.retrieval_unit_id)
            contribution = component.weight / (rrf_k + rank)
            representative = _representative_document(
                evidence,
                record_maps[component.name],
            )

            current = accumulator.get(key)
            if current is None:
                accumulator[key] = {
                    "document_id": evidence.document_id,
                    "retrieval_unit_id": evidence.retrieval_unit_id,
                    "covered_sections": evidence.covered_sections,
                    "score": contribution,
                    "best_rank": rank,
                    "representative": representative,
                    "representative_key": (rank, component_order),
                    "component_ranks": {component.name: rank},
                    "component_weights": {component.name: component.weight},
                    "component_contributions": {
                        component.name: contribution
                    },
                    "component_raw_scores": {
                        component.name: evidence.raw_score
                    },
                    "component_source_record_ids": {
                        component.name: list(evidence.source_record_ids)
                    },
                }
                continue

            if current["covered_sections"] != evidence.covered_sections:
                raise ValueError(
                    "The same retrieval unit has inconsistent section "
                    f"coverage across components: {key!r}"
                )

            current["score"] += contribution
            current["best_rank"] = min(current["best_rank"], rank)
            current["component_ranks"][component.name] = rank
            current["component_weights"][component.name] = component.weight
            current["component_contributions"][component.name] = contribution
            current["component_raw_scores"][component.name] = evidence.raw_score
            current["component_source_record_ids"][component.name] = list(
                evidence.source_record_ids
            )

            representative_key = (rank, component_order)
            if representative_key < current["representative_key"]:
                current["representative"] = representative
                current["representative_key"] = representative_key

    ranked = sorted(
        accumulator.values(),
        key=lambda item: (
            -float(item["score"]),
            int(item["best_rank"]),
            str(item["document_id"]),
            str(item["retrieval_unit_id"]),
        ),
    )[:top_k]

    documents: list[Document] = []
    provenance: list[dict[str, Any]] = []

    for fused_rank, item in enumerate(ranked, start=1):
        document = _build_fused_document(
            item,
            source_type=source_type,
            rrf_k=rrf_k,
            fused_rank=fused_rank,
        )
        documents.append(document)
        provenance.append(
            {
                "fused_rank": fused_rank,
                "document_id": item["document_id"],
                "retrieval_unit_id": item["retrieval_unit_id"],
                "covered_sections": sorted(
                    f"{section.document_id}::{section.section_id}"
                    for section in item["covered_sections"]
                ),
                "rrf_score": float(item["score"]),
                "component_ranks": dict(item["component_ranks"]),
                "component_weights": dict(item["component_weights"]),
                "component_contributions": dict(
                    item["component_contributions"]
                ),
                "component_raw_scores": dict(item["component_raw_scores"]),
                "component_source_record_ids": dict(
                    item["component_source_record_ids"]
                ),
            }
        )

    normalized = normalize_retrieved_documents(
        documents,
        source_type=source_type,
    )

    return FusionResult(
        documents=tuple(documents),
        normalized=normalized,
        provenance=tuple(provenance),
    )


def _build_fused_document(
    item: Mapping[str, Any],
    *,
    source_type: str,
    rrf_k: int,
    fused_rank: int,
) -> Document:
    representative: Document = item["representative"]
    metadata = dict(representative.metadata)
    document_id = str(item["document_id"])
    retrieval_unit_id = str(item["retrieval_unit_id"])
    section_ids = sorted(
        section.section_id for section in item["covered_sections"]
    )
    digest = hashlib.sha256(
        f"{document_id}\0{retrieval_unit_id}".encode("utf-8")
    ).hexdigest()[:20]

    metadata.update(
        {
            "record_id": f"rrf::{digest}",
            "doc_id": document_id,
            "prebuilt_source_type": source_type,
            "retrieval_unit_id": retrieval_unit_id,
            "raw_score": float(item["score"]),
            "retrieval_backend": "hybrid",
            "retrieval_algorithm": "RRF",
            "rrf_k": rrf_k,
            "rrf_rank": fused_rank,
            "rrf_score": float(item["score"]),
            "rrf_component_ranks": dict(item["component_ranks"]),
            "rrf_component_weights": dict(item["component_weights"]),
            "rrf_component_contributions": dict(
                item["component_contributions"]
            ),
            "rrf_component_raw_scores": dict(item["component_raw_scores"]),
            "rrf_component_source_record_ids": dict(
                item["component_source_record_ids"]
            ),
        }
    )

    if source_type in _FIXED_SOURCE_TYPES:
        metadata["source_section_ids"] = section_ids
    elif source_type in _HIERARCHICAL_SOURCE_TYPES:
        metadata["represented_section_ids"] = section_ids
    else:
        raise ValueError(f"Unsupported source_type for RRF: {source_type!r}")

    if section_ids:
        metadata["section_id"] = section_ids[0]

    return Document(
        page_content=representative.page_content,
        metadata=metadata,
    )


def _documents_by_record_id(
    documents: Sequence[Document],
) -> dict[tuple[str, str], Document]:
    output: dict[tuple[str, str], Document] = {}
    for document in documents:
        metadata = document.metadata
        record_id = metadata.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Every component document must have record_id")

        document_id = metadata.get("doc_id", metadata.get("document_id"))
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                "Every component document must have doc_id/document_id"
            )

        identity = (document_id.strip(), record_id.strip())
        if identity in output:
            raise ValueError(
                "Duplicate record_id in component ranking for the same "
                f"document: {identity!r}"
            )
        output[identity] = document
    return output


def _representative_document(
    evidence: RetrievedEvidence,
    record_map: Mapping[tuple[str, str], Document],
) -> Document:
    for record_id in evidence.source_record_ids:
        identity = (evidence.document_id, record_id)
        if identity in record_map:
            return record_map[identity]
    raise ValueError(
        "No component document found for document-scoped evidence source "
        f"records: document_id={evidence.document_id!r}, "
        f"record_ids={evidence.source_record_ids!r}"
    )


def _copy_document(document: Document) -> Document:
    return Document(
        page_content=str(document.page_content or ""),
        metadata=dict(document.metadata),
    )
