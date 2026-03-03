from abc import ABC, abstractmethod
from logging import Logger
from typing import Dict, List

from langchain_core.runnables import RunnableConfig

from agentic_rag.agent import output
from agentic_rag.agent.state import GraphState
from agentic_rag.agent.services.llm_service import LLMService
from agentic_rag.utils.search import SearchResult


class RoutingService:
    def __init__(self, llm_service: LLMService, router_config: RunnableConfig, index_description: str):
        self.llm_service = llm_service
        self.router_config = router_config
        self.index_description = index_description

    def route_rag(self, question: str) -> str:
        runnable = self.llm_service.build_node(
            "router", structured_output=True, output_schema=output.RouteQuery,
            index_description=self.index_description,
        )
        result = runnable.invoke({"question": question}, config=self.router_config, with_retry=True)
        return result.branch

    def route_document(self, question: str) -> str:
        runnable = self.llm_service.build_node(
            "document_request_detector", structured_output=True, output_schema=output.DocumentRequest,
        )
        result = runnable.invoke({"question": question}, config=self.router_config, with_retry=True)
        return result.binary_score

    @staticmethod
    def route_rag_response(documents: List, document_request: bool) -> str:
        if len(documents) > 0:
            return "document_response" if document_request is True else "rag_response"
        return "default_response"

    @staticmethod
    def route_agent(request_type: str) -> str:
        return "conversational" if request_type == "conversational" else "rag"

    # @staticmethod
    # def route_rag_response(document_request: bool) -> str:
    #    return "document_response" if document_request else "rag_response"


class RoutingServiceNode(ABC):
    def __init__(self, routing_service: RoutingService, logger: Logger):
        self.routing_service = routing_service
        self.logger = logger

    @abstractmethod
    def __call__(self, state: GraphState) -> str:
        pass


class RagRouter(RoutingServiceNode):
    def __call__(self, state: GraphState) -> Dict:
        self.logger.info("Routing question...")
        route = self.routing_service.route_rag(question=state["contextual_question"])
        self.logger.info(f"Detected route: {route}")
        return {"request_type": route}


# class DocumentRouter(RoutingServiceNode):
#     def __call__(self, state: GraphState) -> Dict:
#         self.logger.info("Checking if user question requires a document...")
#         document_request = self.routing_service.route_document(question=state["contextual_question"])
#         if document_request == "yes":
#             self.logger.info("User question implies a document request.")
#         else:
#             self.logger.info("The user question does not imply a document request.")
#         return {"document_request": document_request}


# class ResponseRouter(RoutingServiceNode):
#     def __call__(self, state: GraphState) -> str:
#         documents = state.get("documents", [])
#         route_input = documents.chunks if isinstance(documents, SearchResult) else documents
#         response_route = RoutingService.route_response(route_input)
#         return response_route


class RagResponseRouter(RoutingServiceNode):
    def __call__(self, state: GraphState) -> str:
        state_documents = state.get("documents", None)
        documents = state_documents.chunks if state_documents is not None else []
        document_request = True if state["request_type"] == "document_request" else False
        rag_response_route = RoutingService.route_rag_response(documents=documents, document_request=document_request)
        return rag_response_route


class AgentRouter(RoutingServiceNode):
    def __call__(self, state: GraphState) -> str:
        return RoutingService.route_agent(state["request_type"])
