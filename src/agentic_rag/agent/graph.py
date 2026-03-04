import json
import os
import asyncio
from logging import Logger
from pathlib import Path
import traceback

import mlflow
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cardiology_gen_ai.utils.logger import get_logger

from agentic_rag.agent.services.context_service import ContextualizeQuestion, DetectLanguage
from agentic_rag.agent.services.generation_service import GenerateRagResponse, GenerateDefaultResponse, \
    GenerateDocumentResponse, GenerateConversationalResponse
from agentic_rag.agent.services.retrieval_service import Retrieve, RetrievalFilter
from agentic_rag.agent.services.routing_service import RagRouter, RagResponseRouter, AgentRouter
from agentic_rag.agent.services.service_container import ServiceContainer
from agentic_rag.agent.state import GraphState
from agentic_rag.config.manager import NodePromptConfig
from agentic_rag.utils.search import SearchResult
from agentic_rag.config.manager import AgentConfigManager
from agentic_rag.managers.llm_manager import LLMManager
from agentic_rag.managers.search_manager import SearchManager
from agentic_rag.persistence.db import ensure_database
from agentic_rag.persistence.message import AgentMemory
from agentic_rag.utils.chat import ChatRequest, ChatResponse, format_chat_request


GENERATION_LIMIT = 1
os.environ["TOKENIZERS_PARALLELISM"] = "False"


class AgentBuilder:
    def __init__(self, services: ServiceContainer, logger: Logger):
        self.services = services
        self.logger = logger

    def build(self) -> StateGraph:
        graph = StateGraph(GraphState)
        graph.add_node("question_contextualizer", ContextualizeQuestion(self.services.context_service, logger=self.logger))
        graph.add_node("language_detector", DetectLanguage(self.services.context_service, logger=self.logger))
        # graph.add_node("document_request_detector", DocumentRouter(self.services.routing_service, logger=self.logger))
        graph.add_node("rag_router", RagRouter(self.services.routing_service, self.logger))
        graph.add_node("retriever", Retrieve(self.services.retrieval_service, logger=self.logger))
        graph.add_node("retrieval_filter", RetrievalFilter(self.services.retrieval_service, logger=self.logger))
        graph.add_node("rag_response_generator", GenerateRagResponse(self.services.generation_service, logger=self.logger), tags=["final_answer"])
        graph.add_node("default_response_generator", GenerateDefaultResponse(self.services.generation_service, logger=self.logger), tags=["final_answer"])
        graph.add_node("document_response_generator", GenerateDocumentResponse(self.services.generation_service, logger=self.logger))
        graph.add_node("conversational_response_generator", GenerateConversationalResponse(self.services.generation_service, logger=self.logger), tags=["final_answer"])

        graph.add_edge(START, "language_detector")
        graph.add_edge("language_detector", "question_contextualizer")
        graph.add_edge("question_contextualizer", "rag_router")
        graph.add_conditional_edges(
            "rag_router",
            AgentRouter(self.services.routing_service, self.logger),
            {
                "rag": "retriever",
                "conversational": "conversational_response_generator",
            }
        )
        # graph.add_edge("retriever", "retrieval_filter")
        # graph.add_conditional_edges(
        #     "retrieval_filter",
        #     ResponseRouter(self.services.retrieval_service, self.logger),
        #     {
        #         "rag_or_document_response": "document_request_detector",
        #         "default_response": "default_response_generator",
        #     }
        # )
        graph.add_conditional_edges(
            "retriever",  # "retrieval_filter",
            RagResponseRouter(self.services.routing_service, self.logger),
            {
                "document_response": "document_response_generator",
                "rag_response": "rag_response_generator",
                "default_response": "default_response_generator",
            }
        )
        graph.add_edge("conversational_response_generator", END)
        graph.add_edge("default_response_generator", END)
        graph.add_edge("document_response_generator", END)
        graph.add_edge("rag_response_generator", END)

        return graph


class GraphExecutor:
    def __init__(self, compiled_graph: CompiledStateGraph, logger: Logger):
        self.compiled_graph = compiled_graph
        self.logger = logger

    async def run(self, input_state, config, step_logger_fn=None):
        response = {}
        current_event_list = []
        for event in self.compiled_graph.stream(input=input_state, config=config, stream_mode="debug"):
            current_event_list.append(event)
            if event["type"] == "checkpoint":
                if step_logger_fn:
                    try:
                        await step_logger_fn(current_event_list)
                    except Exception as e:
                        self.logger.error(f"Step logger error: {e}")
                if not event["payload"]["metadata"].get("next"):
                    response = event["payload"].get("values", {})
                current_event_list = []
        return response

    async def stream(self, input_state, config, step_logger_fn=None):
        response = {}
        current_event_list = []
        async for event in self.compiled_graph.astream_events(
                input_state,
                config=config,
                version="v2"
        ):
            current_event_list.append(event)
            if event["event"] == "on_chain_end":
                if step_logger_fn:
                    try:
                        await step_logger_fn(current_event_list)
                    except Exception as e:
                        self.logger.error(f"Step logger error: {e}")
                current_event_list = []
                if event.get("name") == "LangGraph":
                    response = event["data"].get("output", {})
            elif event["event"] == "on_chat_model_stream":
                if "final_answer" in event.get("tags", []):
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield {"type": "token", "content": chunk.content}
        yield {"type": "final", "content": response}


class Agent:
    def __init__(self, agent_id: str, config_path: str = None):
        self.config = AgentConfigManager(app_id=agent_id, config_path=config_path).config
        self.logger = get_logger(f"Agent {self.config.name}")
        self.llm_manager = LLMManager(self.config.llm)
        self.search_manager = SearchManager(index_config=self.config.indexing, search_config=self.config.search)
        self.nodes_config: NodePromptConfig = self.config.nodes
        # self.memory = AgentMemory()

        agent_prompt = self.config.system_prompt
        index_description = [index.description for index  in self.config.indexing] \
            if isinstance(self.config.indexing, list) else self.config.indexing.description
        self.services = ServiceContainer(
            llm_manager=self.llm_manager, nodes_prompt_config=self.nodes_config, search_manager=self.search_manager,
            agent_prompt=agent_prompt, index_description=index_description,
            allowed_languages=self.config.allowed_languages,
        )

        graph = AgentBuilder(self.services, self.logger).build()
        self.compiled_graph = graph.compile() # checkpointer=self.memory.checkpointer)
        self.executor = GraphExecutor(self.compiled_graph, self.logger)

    def draw_graph(self, filename: str = None) -> None:
        if not filename:
            filename = f"{type(self).__name__}.txt"
        mermaid_syntax = self.compiled_graph.get_graph().draw_mermaid()
        with open(filename, "w") as file:
            file.write(mermaid_syntax)

    async def answer(self, request: ChatRequest, step_logger_fn=None) -> ChatResponse:
        config = {
            "configurable": {
                "user_id": request.user_id,
                "thread_id": request.conversation.id,
            }
        }
        messages = self.services.context_service.convert_conversation_to_messages(request.conversation)
        summary = ""
        # summary, history = self.services.context_service.summarize_history(
        #     history=messages, messages_to_keep=self.config.memory.length, summary="",
        # )
        input_state: GraphState = {
            "question": request.conversation.question.content,
            "messages": messages,  # history,
            "summary": summary,
            "generation_count": 0,
        }
        try:
            graph_response = await self.executor.run(
                input_state=input_state,
                config=config,
                step_logger_fn=step_logger_fn,
            )
            response, contextual_question = graph_response["response"], graph_response["contextual_question"]
            sources = graph_response.get("documents", SearchResult(chunks=[])).to_sources_payload()
            is_faulted = False
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            is_faulted, sources, contextual_question = True, [], ""
            response = self.services.context_service.handle_error(e, self.config.allowed_languages)
        return ChatResponse(
            role="assistant",
            content=response,
            metadata={
                "sources": sources,
                # "n_gen": response.get("generation_count"),
                "contextual_question": contextual_question,
            },
            is_faulted=is_faulted
        )

    async def answer_stream(self, request: ChatRequest, step_logger_fn=None):
        config = {
            "configurable": {
                "user_id": request.user_id,
                "thread_id": request.conversation.id,
            }
        }
        messages = self.services.context_service.convert_conversation_to_messages(
            request.conversation
        )
        input_state: GraphState = {
            "question": request.conversation.question.content,
            "messages": messages,
            "summary": "",
            "generation_count": 0,
        }
        try:
            async for event in self.executor.stream(
                    input_state=input_state,
                    config=config,
                    step_logger_fn=step_logger_fn,
            ):
                yield event
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            self.logger.error(traceback.format_exc())
            error_message = self.services.context_service.handle_error(e, self.config.allowed_languages)
            yield {"type": "error", "content": error_message}

async def stream_graph(request_chat):
    async for event in agent.answer_stream(chat_request):
        if event["type"] == "token":
            print(event["content"], end="", flush=True)
        elif event["type"] == "final":
            print("\n\n--- FINAL ---")
            print(event["content"])
        elif event["type"] == "error":
            print(f"\nERROR: {event['content']}")

if __name__ == "__main__":
    if os.getenv("MLFLOW_DB", None):
        ensure_database(db_name=os.getenv("MLFLOW_DB"))
        mlflow.set_tracking_uri(f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:5432/{os.getenv("MLFLOW_DB")}")
        mlflow.set_experiment(f"{os.getenv('AGENT_ID')}_tracing")
        mlflow.langchain.autolog()
    agent = Agent(agent_id="cardiology_protocols", config_path=os.getenv("CONFIG_PATH"))
    with open(Path("tests/data/synthetic_data.json"), "r", encoding="utf-8") as f:
        items = json.load(f)
    with mlflow.start_run(run_name=f"test_graph_synthetic_data_no_grade", nested=True):
        for item in items:
            chat_request = format_chat_request(item["question"])
            # metadata["chunk_idx"]
            agent_response = asyncio.run(stream_graph(chat_request))
            # print(agent_response.content)
