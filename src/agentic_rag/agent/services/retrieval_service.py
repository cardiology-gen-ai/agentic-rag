from abc import ABC, abstractmethod
from logging import Logger
from typing import List, Dict

from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from agentic_rag.agent import output
from agentic_rag.agent.state import GraphState
from agentic_rag.agent.services.llm_service import LLMService
from agentic_rag.managers.search_manager import SearchManager
from agentic_rag.utils.search import SearchResult


class RetrievalService:
    def __init__(self, llm_service: LLMService, search_manager: SearchManager, grader_config: RunnableConfig):
        self.llm_service = llm_service
        self.search_manager = search_manager
        self.grader_config = grader_config

    def retrieve(self, query: str) -> SearchResult:
        return self.search_manager.search(query)

    def filter_retrieved_documents(self, query: str, documents: List[Document]) -> SearchResult:
        # doc_filter = LLMChainFilter.from_llm(llm=self.llm_service.llm, prompt=None)
        # filtered_docs = doc_filter.compress_documents(documents=documents, query=query)
        input_context = "\n\n".join(f"Document {i + 1}:\n{doc}" for i, doc in enumerate(documents))
        runnable = self.llm_service.build_node(
            "retrieval_grader", structured_output=True, output_schema=output.GradeDocumentsBatch,
        )
        response = runnable.invoke(
            {"question": query, "documents": input_context}, config=self.grader_config, with_retry=True,
        )
        grades = response.grades
        filtered_docs = documents if len(grades) != len(documents) else \
            [doc for i, doc in enumerate(documents) if grades[i].binary_score == "yes"]
        return SearchResult(chunks=list(filtered_docs))


class RetrievalServiceNode(ABC):
    def __init__(self, retrieval_service: RetrievalService, logger: Logger):
        self.retrieval_service = retrieval_service
        self.logger = logger

    @abstractmethod
    def __call__(self, state: GraphState) -> Dict:
        pass


class Retrieve(RetrievalServiceNode):
    def __call__(self, state: GraphState) -> Dict:
        contextual_question = state["contextual_question"]
        self.logger.info(f"Retrieving documents for contextualized question: {contextual_question}...")
        retrieved_documents = self.retrieval_service.retrieve(query=contextual_question)
        self.logger.info(f"Retrieved {len(retrieved_documents.chunks)} documents")
        return {"documents": retrieved_documents}


class RetrievalFilter(RetrievalServiceNode):
    def __call__(self, state: GraphState) -> Dict:
        self.logger.info(f"Grading {len(state['documents'].chunks)} retrieved documents...")
        documents = state["documents"].chunks
        filtered_documents = self.retrieval_service.filter_retrieved_documents(
            query=state["contextual_question"], documents=documents,
        )
        self.logger.info(f"{len(filtered_documents.chunks)} / {len(state['documents'].chunks)} documents are relevant for the question.")
        return {"documents": filtered_documents}
