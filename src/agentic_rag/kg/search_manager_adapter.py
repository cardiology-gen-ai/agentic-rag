"""Compatibility bridge between the KG LangChain retriever and the agent.

The existing agent expects a search-manager-like object exposing
``search(query) -> SearchResult``.  It also expects every returned LangChain
Document to provide the legacy metadata keys ``filename``, ``chunk_idx`` and
``headers``.

This adapter adds only that compatibility surface.  It does not change the
classic SearchManager or the KG retrieval core.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from agentic_rag.utils.search import SearchResult


class KGSearchManagerAdapter:
    """Expose a LangChain KG retriever through the agent SearchManager contract."""

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def search(self, query: str) -> SearchResult:
        documents = self.retriever.invoke(query)
        return SearchResult(
            chunks=[
                _with_agent_compatibility_metadata(document)
                for document in documents
            ]
        )


def _with_agent_compatibility_metadata(
    document: Document,
) -> Document:
    """Return a copy carrying metadata required by the current agent stack."""

    metadata: dict[str, Any] = deepcopy(document.metadata)

    document_id = _first_non_empty(
        metadata.get("document_id"),
        metadata.get("filename"),
        "unknown",
    )
    section_uid = _first_non_empty(
        metadata.get("retrieval_unit_id"),
        metadata.get("section_uid"),
        metadata.get("section_id"),
        "unknown",
    )

    metadata.setdefault("filename", str(document_id))
    metadata.setdefault("chunk_idx", str(section_uid))
    metadata.setdefault(
        "headers",
        {
            "section": [
                _section_header(metadata)
            ]
        },
    )

    return Document(
        page_content=document.page_content,
        metadata=metadata,
    )


def _section_header(metadata: dict[str, Any]) -> str:
    printed_id = _optional_text(
        metadata.get("printed_section_id")
    )
    title = _optional_text(metadata.get("title"))

    if printed_id and title:
        return f"{printed_id} {title}"
    if title:
        return title
    if printed_id:
        return printed_id

    return _first_non_empty(
        metadata.get("section_id"),
        metadata.get("section_uid"),
        "Retrieved section",
    )


def _first_non_empty(*values: Any) -> str:
    for value in values:
        normalized = _optional_text(value)
        if normalized is not None:
            return normalized
    return "unknown"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
