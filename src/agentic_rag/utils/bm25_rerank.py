"""Deterministic BM25Plus reranking for a dense candidate pool.

This module deliberately does *not* perform corpus-wide retrieval. It takes the
candidate list produced by a dense retriever and reranks exactly that candidate
set with the same BM25Plus implementation used by the sparse baseline.

Main invariant:
    candidate_ids_before == candidate_ids_after

Only ranking may change.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.documents import Document

from agentic_rag.utils.bm25 import (
    BM25_ALGORITHM,
    build_bm25_dict,
    rank_bm25_documents,
)

DENSE_BM25PLUS_RERANK_ALGORITHM = "DenseToBM25PlusRerank"
DENSE_BM25PLUS_RERANK_BACKEND = "dense_bm25plus_rerank"


def rerank_dense_candidates_bm25plus(
    query: str,
    candidates: Sequence[Document],
    *,
    expected_candidate_count: int | None = None,
) -> list[Document]:
    """Rerank a dense candidate pool with BM25Plus without changing membership."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if expected_candidate_count is not None and expected_candidate_count < 1:
        raise ValueError("expected_candidate_count must be >= 1")

    if not candidates:
        if expected_candidate_count is not None:
            raise ValueError(
                "Dense candidate pool is empty; expected "
                f"{expected_candidate_count} candidates"
            )
        return []

    dense_candidates = [
        _copy_document_with_dense_provenance(document, dense_rank=rank)
        for rank, document in enumerate(candidates, start=1)
    ]

    candidate_ids = [_candidate_identity(document) for document in dense_candidates]
    duplicates = _duplicates(candidate_ids)
    if duplicates:
        raise ValueError(
            "Dense candidate pool contains duplicate evidence identities: "
            f"{duplicates!r}"
        )

    if (
        expected_candidate_count is not None
        and len(dense_candidates) != expected_candidate_count
    ):
        raise ValueError(
            "Dense candidate pool size mismatch: expected "
            f"{expected_candidate_count}, found {len(dense_candidates)}"
        )

    local_bm25 = build_bm25_dict(dense_candidates)
    reranked = rank_bm25_documents(
        local_bm25,
        query,
        k=len(dense_candidates),
    )

    finalized: list[Document] = []
    for reranked_rank, document in enumerate(reranked, start=1):
        metadata = dict(document.metadata)
        bm25plus_score = metadata.get("raw_score")
        bm25plus_raw_rank = metadata.get("bm25_raw_rank")

        metadata.update(
            {
                "reranker_used": True,
                "reranker_algorithm": BM25_ALGORITHM,
                "retrieval_backend": DENSE_BM25PLUS_RERANK_BACKEND,
                "retrieval_algorithm": DENSE_BM25PLUS_RERANK_ALGORITHM,
                "bm25plus_rerank_score": bm25plus_score,
                "bm25plus_rerank_rank": reranked_rank,
                "bm25plus_local_raw_rank": bm25plus_raw_rank,
                "dense_candidate_pool_size": len(dense_candidates),
                "rerank_candidate_pool_size": len(dense_candidates),
            }
        )
        finalized.append(_copy_document(document, metadata=metadata))

    reranked_ids = [_candidate_identity(document) for document in finalized]

    if len(reranked_ids) != len(candidate_ids):
        raise AssertionError(
            "BM25Plus reranking changed candidate-pool cardinality"
        )
    if set(reranked_ids) != set(candidate_ids):
        raise AssertionError(
            "BM25Plus reranking changed candidate-pool membership"
        )
    if len(reranked_ids) != len(set(reranked_ids)):
        raise AssertionError(
            "BM25Plus reranking introduced duplicate evidence identities"
        )

    return finalized


def _copy_document_with_dense_provenance(
    document: Document,
    *,
    dense_rank: int,
) -> Document:
    if not isinstance(document, Document):
        raise TypeError(
            "Dense candidates must be LangChain Document objects, "
            f"found {type(document)!r}"
        )

    metadata = dict(document.metadata)
    metadata.update(
        {
            "dense_original_rank": dense_rank,
            "dense_original_raw_score": metadata.get("raw_score"),
        }
    )
    return _copy_document(document, metadata=metadata)


def _candidate_identity(document: Document) -> str:
    metadata = document.metadata

    doc_id = None
    for key in ("doc_id", "document_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            doc_id = value.strip()
            break
    local_id = None
    for key in (
        "retrieval_unit_key",
        "retrieval_unit_id",
        "record_id",
        "chunk_id",
        "id",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            local_id = value.strip()
            break

    if local_id is None:
        document_id = getattr(document, "id", None)
        if document_id is not None and str(document_id).strip():
            local_id = str(document_id).strip()

    if local_id is None:
        raise ValueError(
            "Dense candidate has no stable local identity. Expected one of "
            "retrieval_unit_key, retrieval_unit_id, record_id, chunk_id, "
            "metadata id, or Document.id"
        )

    return f"{doc_id}::{local_id}" if doc_id is not None else local_id


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    return sorted(duplicates)


def _copy_document(
    document: Document,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Document:
    kwargs: dict[str, Any] = {
        "page_content": str(document.page_content or ""),
        "metadata": dict(document.metadata if metadata is None else metadata),
    }

    document_id = getattr(document, "id", None)
    if document_id is not None:
        kwargs["id"] = document_id

    return Document(**kwargs)
