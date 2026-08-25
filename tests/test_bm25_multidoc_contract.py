from __future__ import annotations

import pytest
from langchain_core.documents import Document

from agentic_rag.utils.bm25 import validate_bm25_documents


def _doc(doc_id: str, record_id: str, key: str | None = None) -> Document:
    metadata = {
        "doc_id": doc_id,
        "record_id": record_id,
        "prebuilt_source_type": "fixed_chunks",
        "source_section_ids": ["1"],
    }
    if key is not None:
        metadata["retrieval_unit_key"] = key
    return Document(page_content="non empty text", metadata=metadata)


def test_same_local_record_id_across_documents_is_valid() -> None:
    result = validate_bm25_documents(
        [
            _doc("Doc_A", "chunk-1", "Doc_A::fixed::chunk-1"),
            _doc("Doc_B", "chunk-1", "Doc_B::fixed::chunk-1"),
        ],
        expected_count=2,
        expected_source_type="fixed_chunks",
        expected_document_counts={"Doc_A": 1, "Doc_B": 1},
    )
    assert result["record_id_count"] == 1
    assert result["document_scoped_identity_count"] == 2
    assert result["retrieval_unit_key_count"] == 2


def test_duplicate_document_scoped_identity_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate document-scoped retrieval identity",
    ):
        validate_bm25_documents(
            [_doc("Doc_A", "chunk-1"), _doc("Doc_A", "chunk-1")]
        )


def test_duplicate_retrieval_unit_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate retrieval_unit_key"):
        validate_bm25_documents(
            [
                _doc("Doc_A", "chunk-1", "shared"),
                _doc("Doc_B", "chunk-1", "shared"),
            ]
        )


def test_expected_document_counts_are_enforced() -> None:
    with pytest.raises(ValueError, match="Per-document count mismatch"):
        validate_bm25_documents(
            [_doc("Doc_A", "chunk-1"), _doc("Doc_B", "chunk-1")],
            expected_document_counts={"Doc_A": 2},
        )
