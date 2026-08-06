"""Normalize backend-specific retrieval results into comparable evidence units.

A hierarchical retrieval unit remains one ranking position even when it
contains multiple original sections. Fixed chunks belonging to the same
section set are collapsed into one evidence unit while preserving the best
raw rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from langchain_core.documents import Document


_FIXED_SOURCE_TYPES = frozenset({"fixed_chunks", "fixed"})
_HIERARCHICAL_SOURCE_TYPES = frozenset(
    {"hierarchical_section_view", "hierarchical"}
)


@dataclass(frozen=True, order=True)
class EvidenceSection:
    """Identity of an original guideline section."""

    document_id: str
    section_id: str

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not self.section_id.strip():
            raise ValueError("section_id must be a non-empty string")


@dataclass(frozen=True)
class RetrievedEvidence:
    """One ranked evidence unit after backend normalization."""

    document_id: str
    retrieval_unit_id: str
    covered_sections: frozenset[EvidenceSection]
    raw_rank: int
    source_record_ids: tuple[str, ...]
    source_type: str
    raw_score: float | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not self.retrieval_unit_id.strip():
            raise ValueError("retrieval_unit_id must be a non-empty string")
        if not self.covered_sections:
            raise ValueError("covered_sections must contain at least one section")
        if self.raw_rank < 1:
            raise ValueError("raw_rank must be 1-based and >= 1")
        if not self.source_record_ids:
            raise ValueError("source_record_ids must not be empty")
        if not self.source_type.strip():
            raise ValueError("source_type must be a non-empty string")

    @property
    def covered_section_ids(self) -> frozenset[str]:
        return frozenset(section.section_id for section in self.covered_sections)


@dataclass(frozen=True)
class EvidenceNormalizationResult:
    """Normalized ranking and diagnostics for one query."""

    evidence: tuple[RetrievedEvidence, ...]
    raw_result_count: int
    normalized_evidence_count: int
    duplicate_result_count: int
    source_type: str

    def __post_init__(self) -> None:
        if self.raw_result_count < 0:
            raise ValueError("raw_result_count must be >= 0")
        if self.normalized_evidence_count != len(self.evidence):
            raise ValueError(
                "normalized_evidence_count must equal len(evidence)"
            )
        expected_duplicates = (
            self.raw_result_count - self.normalized_evidence_count
        )
        if self.duplicate_result_count != expected_duplicates:
            raise ValueError(
                "duplicate_result_count must equal "
                "raw_result_count - normalized_evidence_count"
            )

    def has_at_least(self, k: int) -> bool:
        if k < 1:
            raise ValueError("k must be >= 1")
        return self.normalized_evidence_count >= k

    def top_k(self, k: int) -> tuple[RetrievedEvidence, ...]:
        if k < 1:
            raise ValueError("k must be >= 1")
        return self.evidence[:k]


def normalize_retrieved_documents(
    documents: Sequence[Document],
    *,
    source_type: str | None = None,
) -> EvidenceNormalizationResult:
    """Convert raw LangChain documents into ranked evidence units.

    For ``fixed_chunks``, chunks with the same document and
    ``source_section_ids`` are collapsed. For
    ``hierarchical_section_view``, every raw result remains one ranking
    position and coverage is defined only by ``represented_section_ids``.
    """

    resolved_source_type = _resolve_source_type(documents, source_type)

    if resolved_source_type in _FIXED_SOURCE_TYPES:
        return _normalize_fixed(documents, resolved_source_type)

    if resolved_source_type in _HIERARCHICAL_SOURCE_TYPES:
        return _normalize_hierarchical(documents, resolved_source_type)

    raise ValueError(
        "Unsupported source_type "
        f"{resolved_source_type!r}. Expected one of "
        f"{sorted(_FIXED_SOURCE_TYPES | _HIERARCHICAL_SOURCE_TYPES)}"
    )


def _normalize_fixed(
    documents: Sequence[Document],
    source_type: str,
) -> EvidenceNormalizationResult:
    grouped: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    for raw_rank, document in enumerate(documents, start=1):
        metadata = _metadata(document)
        document_id = _required_string(
            metadata,
            ("doc_id", "document_id"),
            field_name="document identifier",
        )
        section_ids = _required_string_sequence(
            metadata,
            "source_section_ids",
        )
        section_key = tuple(sorted(set(section_ids)))
        record_id = _required_string(
            metadata,
            ("record_id",),
            field_name="record_id",
        )
        score = _optional_score(metadata)

        key = (document_id, section_key)
        group = grouped.get(key)

        if group is None:
            grouped[key] = {
                "document_id": document_id,
                "section_ids": section_key,
                "raw_rank": raw_rank,
                "record_ids": [record_id],
                "raw_score": score,
            }
            continue

        if record_id not in group["record_ids"]:
            group["record_ids"].append(record_id)

    evidence: list[RetrievedEvidence] = []

    for group in grouped.values():
        document_id = group["document_id"]
        section_ids = group["section_ids"]
        covered_sections = frozenset(
            EvidenceSection(document_id, section_id)
            for section_id in section_ids
        )
        retrieval_unit_id = _fixed_retrieval_unit_id(
            document_id,
            section_ids,
        )

        evidence.append(
            RetrievedEvidence(
                document_id=document_id,
                retrieval_unit_id=retrieval_unit_id,
                covered_sections=covered_sections,
                raw_rank=group["raw_rank"],
                source_record_ids=tuple(group["record_ids"]),
                source_type=source_type,
                raw_score=group["raw_score"],
            )
        )

    evidence.sort(key=lambda item: item.raw_rank)

    return EvidenceNormalizationResult(
        evidence=tuple(evidence),
        raw_result_count=len(documents),
        normalized_evidence_count=len(evidence),
        duplicate_result_count=len(documents) - len(evidence),
        source_type=source_type,
    )


def _normalize_hierarchical(
    documents: Sequence[Document],
    source_type: str,
) -> EvidenceNormalizationResult:
    evidence: list[RetrievedEvidence] = []
    seen_units: set[tuple[str, str]] = set()

    for raw_rank, document in enumerate(documents, start=1):
        metadata = _metadata(document)
        document_id = _required_string(
            metadata,
            ("doc_id", "document_id"),
            field_name="document identifier",
        )
        retrieval_unit_id = _required_string(
            metadata,
            ("retrieval_unit_id",),
            field_name="retrieval_unit_id",
        )
        represented_section_ids = _required_string_sequence(
            metadata,
            "represented_section_ids",
        )
        record_id = _required_string(
            metadata,
            ("record_id",),
            field_name="record_id",
        )
        score = _optional_score(metadata)

        unit_key = (document_id, retrieval_unit_id)
        if unit_key in seen_units:
            raise ValueError(
                "Duplicate hierarchical retrieval unit in raw ranking: "
                f"{unit_key!r}"
            )
        seen_units.add(unit_key)

        covered_sections = frozenset(
            EvidenceSection(document_id, section_id)
            for section_id in represented_section_ids
        )

        evidence.append(
            RetrievedEvidence(
                document_id=document_id,
                retrieval_unit_id=retrieval_unit_id,
                covered_sections=covered_sections,
                raw_rank=raw_rank,
                source_record_ids=(record_id,),
                source_type=source_type,
                raw_score=score,
            )
        )

    return EvidenceNormalizationResult(
        evidence=tuple(evidence),
        raw_result_count=len(documents),
        normalized_evidence_count=len(evidence),
        duplicate_result_count=0,
        source_type=source_type,
    )


def _resolve_source_type(
    documents: Sequence[Document],
    explicit_source_type: str | None,
) -> str:
    if explicit_source_type is not None:
        source_type = explicit_source_type.strip()
        if not source_type:
            raise ValueError("source_type must not be empty")
        return source_type

    if not documents:
        raise ValueError(
            "source_type must be supplied when documents is empty"
        )

    observed = {
        str(_metadata(document).get("prebuilt_source_type", "")).strip()
        for document in documents
    }
    observed.discard("")

    if len(observed) != 1:
        raise ValueError(
            "Could not infer one source_type from prebuilt_source_type; "
            f"observed={sorted(observed)!r}"
        )

    return next(iter(observed))


def _metadata(document: Document) -> Mapping[str, Any]:
    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise TypeError("Every raw result must expose mapping metadata")
    return metadata


def _required_string(
    metadata: Mapping[str, Any],
    candidate_keys: Iterable[str],
    *,
    field_name: str,
) -> str:
    for key in candidate_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        f"Missing non-empty {field_name}; checked keys "
        f"{tuple(candidate_keys)!r}"
    )


def _required_string_sequence(
    metadata: Mapping[str, Any],
    key: str,
) -> tuple[str, ...]:
    value = metadata.get(key)

    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(
            f"Metadata field {key!r} must be a non-empty sequence"
        )

    normalized = tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )

    if not normalized:
        raise ValueError(
            f"Metadata field {key!r} must contain non-empty strings"
        )

    return normalized


def _optional_score(metadata: Mapping[str, Any]) -> float | None:
    for key in ("raw_score", "score", "similarity_score"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _fixed_retrieval_unit_id(
    document_id: str,
    section_ids: tuple[str, ...],
) -> str:
    joined_sections = ",".join(section_ids)
    return f"{document_id}::fixed_evidence::{joined_sections}"
