from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from agentic_rag.kg.search_manager_adapter import (
    KGSearchManagerAdapter,
)


class FakeKGRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager,
    ) -> list[Document]:
        return list(self.documents)


def make_kg_document(
    *,
    section_uid: str = "Cardiomyopathies_2023::6.8.2",
    retrieval_unit_id: str = "cm-6.8.2",
) -> Document:
    return Document(
        page_content="Genetic testing should be considered.",
        metadata={
            "retriever": "kg",
            "document_id": "Cardiomyopathies_2023",
            "section_uid": section_uid,
            "section_id": "6.8.2",
            "printed_section_id": "6.8.2",
            "title": "Genetic testing",
            "retrieval_unit_id": retrieval_unit_id,
            "kg_rank": 1,
        },
    )


def test_search_returns_existing_agent_search_result_contract() -> None:
    adapter = KGSearchManagerAdapter(
        FakeKGRetriever(
            documents=[make_kg_document()]
        )
    )

    result = adapter.search("genetic testing")

    assert len(result.chunks) == 1
    document = result.chunks[0]

    assert document.page_content == (
        "Genetic testing should be considered."
    )
    assert document.metadata["retriever"] == "kg"

    # Legacy agent compatibility keys.
    assert document.metadata["filename"] == "Cardiomyopathies_2023"
    assert document.metadata["chunk_idx"] == "cm-6.8.2"
    assert document.metadata["headers"] == {
        "section": ["6.8.2 Genetic testing"]
    }


def test_search_result_source_helpers_accept_kg_documents() -> None:
    adapter = KGSearchManagerAdapter(
        FakeKGRetriever(
            documents=[
                make_kg_document(),
                make_kg_document(
                    section_uid="Cardiomyopathies_2023::6.8.3",
                    retrieval_unit_id="cm-6.8.3",
                ),
            ]
        )
    )

    result = adapter.search("genetic testing")

    # These are the exact helpers used downstream by Agent / GenerationService.
    unique = result.extract_unique_chunks()
    payload = result.to_sources_payload()
    formatted = result.format_sources()

    assert len(unique) == 2
    assert len(payload) == 2
    assert payload[0]["filename"] == "Cardiomyopathies_2023"
    assert payload[0]["chunk_idx"] == "cm-6.8.2"
    assert isinstance(formatted, str)
    assert formatted


def test_bridge_does_not_mutate_original_langchain_document() -> None:
    original = make_kg_document()

    adapter = KGSearchManagerAdapter(
        FakeKGRetriever(documents=[original])
    )
    result = adapter.search("genetic testing")

    assert "filename" not in original.metadata
    assert "chunk_idx" not in original.metadata
    assert "headers" not in original.metadata

    assert "filename" in result.chunks[0].metadata


def test_existing_legacy_metadata_is_preserved() -> None:
    document = make_kg_document()
    document.metadata.update(
        {
            "filename": "custom.pdf",
            "chunk_idx": 42,
            "headers": {"legacy": ["Existing header"]},
        }
    )

    adapter = KGSearchManagerAdapter(
        FakeKGRetriever(documents=[document])
    )

    returned = adapter.search("question").chunks[0]

    assert returned.metadata["filename"] == "custom.pdf"
    assert returned.metadata["chunk_idx"] == 42
    assert returned.metadata["headers"] == {
        "legacy": ["Existing header"]
    }
