"""Deterministic retrieval traces for dense and future graph retrievers.

A trace records the observable path from a query to the evaluated ranking:
raw candidates, normalization/deduplication, evidence coverage, and backend
metadata. It is provenance data, not model chain-of-thought.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Literal, Mapping, Sequence

from langchain_core.documents import Document

from agentic_rag.evaluation.evidence import (
    EvidenceNormalizationResult,
    EvidenceSection,
)
from agentic_rag.evaluation.metrics import coverage_at_cutoff


TraceTextMode = Literal["none", "preview", "full"]


def build_retrieval_trace(
    *,
    question_id: str,
    system: str,
    query: str,
    gold_sections: Iterable[EvidenceSection],
    raw_documents: Sequence[Document],
    normalized: EvidenceNormalizationResult,
    cutoffs: Sequence[int],
    text_mode: TraceTextMode = "preview",
    preview_chars: int = 1000,
    backend_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-serializable trace for a question-system pair."""

    _validate_trace_options(
        text_mode=text_mode,
        preview_chars=preview_chars,
        cutoffs=cutoffs,
    )

    gold = frozenset(gold_sections)
    if not gold:
        raise ValueError("gold_sections must not be empty")

    raw_candidates, raw_by_record_id = _serialize_raw_candidates(
        raw_documents,
        gold=gold,
        text_mode=text_mode,
        preview_chars=preview_chars,
    )

    normalized_evidence = _serialize_normalized_evidence(
        normalized,
        gold=gold,
        raw_by_record_id=raw_by_record_id,
    )

    coverage_by_cutoff: dict[str, dict[str, list[str]]] = {}
    for k in cutoffs:
        found, missing = coverage_at_cutoff(
            normalized.evidence,
            gold,
            k=k,
        )
        coverage_by_cutoff[str(k)] = {
            "found_gold_sections": _serialize_sections(found),
            "missing_gold_sections": _serialize_sections(missing),
        }

    candidate_found, candidate_missing = coverage_at_cutoff(
        normalized.evidence,
        gold,
        k=len(normalized.evidence),
    )

    return {
        "trace_schema_version": "1.1",
        "question_id": question_id,
        "system": system,
        "query": query,
        "gold_sections": _serialize_sections(gold),
        "source_type": normalized.source_type,
        "raw_result_count": normalized.raw_result_count,
        "normalized_evidence_count":
            normalized.normalized_evidence_count,
        "duplicate_result_count":
            normalized.duplicate_result_count,
        "candidate_pool_found_gold_sections":
            _serialize_sections(candidate_found),
        "candidate_pool_missing_gold_sections":
            _serialize_sections(candidate_missing),
        "coverage_by_cutoff": coverage_by_cutoff,
        "raw_candidates": raw_candidates,
        "normalized_evidence": normalized_evidence,
        "backend_trace": dict(backend_trace or {}),
        "trace_text": {
            "mode": text_mode,
            "preview_chars": (
                preview_chars
                if text_mode == "preview"
                else None
            ),
        },
    }


def _serialize_raw_candidates(
    raw_documents: Sequence[Document],
    *,
    gold: frozenset[EvidenceSection],
    text_mode: TraceTextMode,
    preview_chars: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    serialized: list[dict[str, Any]] = []
    raw_by_record_id: dict[str, dict[str, Any]] = {}

    for raw_rank, document in enumerate(raw_documents, start=1):
        metadata = _metadata(document)
        record_id = _required_string(metadata, "record_id")

        if record_id in raw_by_record_id:
            raise ValueError(
                "Duplicate record_id in raw retrieval results: "
                f"{record_id!r}"
            )

        document_id = _required_one_of(
            metadata,
            ("doc_id", "document_id"),
            label="document identifier",
        )
        source_section_ids = _string_list(
            metadata.get("source_section_ids")
        )
        represented_section_ids = _string_list(
            metadata.get("represented_section_ids")
        )
        section_ids = (
            represented_section_ids
            or source_section_ids
            or _string_list([metadata.get("section_id")])
        )

        candidate_sections = frozenset(
            EvidenceSection(document_id, section_id)
            for section_id in section_ids
        )
        covered_gold = candidate_sections & gold
        page_content = str(
            getattr(document, "page_content", "") or ""
        )

        item = {
            "raw_rank": raw_rank,
            "record_id": record_id,
            "document_id": document_id,
            "section_id": metadata.get("section_id"),
            "retrieval_unit_id":
                metadata.get("retrieval_unit_id"),
            "source_key": metadata.get("source_key"),
            "source_type":
                metadata.get("prebuilt_source_type"),
            "source_section_ids": source_section_ids,
            "represented_section_ids":
                represented_section_ids,
            "structural_context_section_ids":
                _string_list(
                    metadata.get(
                        "structural_context_section_ids"
                    )
                ),
            "candidate_sections":
                _serialize_sections(candidate_sections),
            "covered_gold_sections":
                _serialize_sections(covered_gold),
            "is_gold_relevant": bool(covered_gold),
            "raw_score": _optional_number(
                metadata,
                ("raw_score", "score", "similarity_score"),
            ),
            "retrieval_backend": metadata.get("retrieval_backend"),
            "retrieval_algorithm": metadata.get("retrieval_algorithm"),
            "bm25_score_positive": metadata.get("bm25_score_positive"),
            "bm25_query_token_count": metadata.get("bm25_query_token_count"),
            "bm25_query_token_overlap_count": metadata.get(
                "bm25_query_token_overlap_count"
            ),
            "bm25_has_query_token_overlap": metadata.get(
                "bm25_has_query_token_overlap"
            ),
            "bm25_matched_query_tokens": _string_list(
                metadata.get("bm25_matched_query_tokens")
            ),
            "bm25_corpus_position": metadata.get("bm25_corpus_position"),
            "text_chars": len(page_content),
            "text_sha256": hashlib.sha256(
                page_content.encode("utf-8")
            ).hexdigest(),
            "text": _select_text(
                page_content,
                mode=text_mode,
                preview_chars=preview_chars,
            ),
        }

        serialized.append(item)
        raw_by_record_id[record_id] = item

    return serialized, raw_by_record_id


def _serialize_normalized_evidence(
    normalized: EvidenceNormalizationResult,
    *,
    gold: frozenset[EvidenceSection],
    raw_by_record_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for normalized_rank, evidence in enumerate(
        normalized.evidence,
        start=1,
    ):
        missing_records = [
            record_id
            for record_id in evidence.source_record_ids
            if record_id not in raw_by_record_id
        ]
        if missing_records:
            raise ValueError(
                "Normalized evidence references raw records "
                f"not present in the trace: {missing_records!r}"
            )

        contributor_raw_ranks = sorted(
            int(raw_by_record_id[record_id]["raw_rank"])
            for record_id in evidence.source_record_ids
        )
        covered_gold = evidence.covered_sections & gold

        output.append(
            {
                "normalized_rank": normalized_rank,
                "best_raw_rank": evidence.raw_rank,
                "contributor_raw_ranks":
                    contributor_raw_ranks,
                "retrieval_unit_id":
                    evidence.retrieval_unit_id,
                "document_id": evidence.document_id,
                "source_type": evidence.source_type,
                "source_record_ids":
                    list(evidence.source_record_ids),
                "covered_sections":
                    _serialize_sections(
                        evidence.covered_sections
                    ),
                "covered_gold_sections":
                    _serialize_sections(covered_gold),
                "is_gold_relevant": bool(covered_gold),
                "raw_score": evidence.raw_score,
            }
        )

    return output


def _select_text(
    text: str,
    *,
    mode: TraceTextMode,
    preview_chars: int,
) -> str | None:
    if mode == "none":
        return None
    if mode == "full":
        return text
    return text[:preview_chars]


def _validate_trace_options(
    *,
    text_mode: str,
    preview_chars: int,
    cutoffs: Sequence[int],
) -> None:
    if text_mode not in {"none", "preview", "full"}:
        raise ValueError(
            "text_mode must be one of: none, preview, full"
        )
    if preview_chars < 1:
        raise ValueError("preview_chars must be >= 1")
    if not cutoffs:
        raise ValueError("cutoffs must not be empty")
    if any(int(k) < 1 for k in cutoffs):
        raise ValueError("every cutoff must be >= 1")


def _metadata(document: Document) -> Mapping[str, Any]:
    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise TypeError(
            "Every raw document must expose mapping metadata"
        )
    return metadata


def _required_string(
    mapping: Mapping[str, Any],
    key: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Missing non-empty string metadata field {key!r}"
        )
    return value.strip()


def _required_one_of(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    *,
    label: str,
) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"Missing non-empty {label}; checked {tuple(keys)!r}"
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set, frozenset)):
        value = [value]
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]



def _optional_number(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None

def _serialize_sections(
    sections: Iterable[EvidenceSection],
) -> list[str]:
    return sorted(
        f"{section.document_id}::{section.section_id}"
        for section in sections
    )
