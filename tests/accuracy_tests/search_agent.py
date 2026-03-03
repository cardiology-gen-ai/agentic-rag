import os
import json
import pathlib
import datetime
import asyncio
import uuid
from logging import Logger
from typing import TypedDict, Dict, List, Annotated, Optional

import mlflow
from langchain.agents.middleware import SummarizationMiddleware
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_core.messages import HumanMessage, AnyMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from cardiology_gen_ai.utils.logger import get_logger
from langgraph.runtime import Runtime

from agentic_rag.utils.search import SearchResult
from src.agentic_rag.managers.nodes_manager import NodeFactory, NodeConfig
from src.agentic_rag.config.manager import AgentConfigManager, AgentConfig
from src.agentic_rag.managers.llm_manager import LLMManager
from src.agentic_rag.managers.search_manager import SearchManager
from src.agentic_rag.persistence.db import ensure_database
from src.agentic_rag.persistence.message import AgentMemory
from src.agentic_rag.agent import output
from src.agentic_rag.utils.chat import ChatRequest, ConversationRequest, MessageSchema, ChatResponse
from src.agentic_rag.utils.nodes import load_yaml

GENERATION_LIMIT = 1
os.environ["TOKENIZERS_PARALLELISM"] = "False"


class GraphState(TypedDict, total=False):
    """Shared state passed between :langgraph:`LangGraph <reference/graphs>` nodes."""
    question: str #: Original user question as received by the agent.
    contextual_question: str #: Question enriched with context (if needed) for retrieval/generation.
    transform_query_count: int #: How many times the question has been rewritten so far.
    response: str #: Latest assistant response (when available).
    language: str #: Language used in the conversation, optional.
    messages: Annotated[List[AnyMessage], add_messages] #: Rolling chat history used for context (:class:`list` of :class:`~langchain_core.messages.base.AnyMessage` ).
    documents: SearchResult #: Retrieved and filtered documents (:class:`list` of :langchain:`Document <core/documents/langchain_core.documents.base.Document.html>`), optional (when applicable).
    document_request: str #: Binary flag to denote whether the user is asking for an entire document.


class SearchAgent:
    """RAG/conversational agent orchestrated with :langgraph:`LangGraph <reference/graphs>`.

    The agent connects an LLM manager (router/generator/grader), a vector store
    search manager (retriever), and a memory/checkpoint backend to compile
    a :langgraph:`CompiledStateGraph <reference/graphs/?h=compiled#langgraph.graph.state.CompiledStateGraph>` that handles
    conversation turns end-to-end.

    Parameters
    ----------
    agent_id : :class:`str`
        Identifier used to load configuration via :class:`~src.agentic_rag.config.manager.AgentConfigManager`.
    """
    agent_id: str #: :class:`str` : Identifier of this agent instance.
    agent_name: str #: :class:`str` : Human-friendly name from configuration.
    config: AgentConfig #: :class:`~src.agentic_rag.config.manager.AgentConfig` : Loaded configuration (system prompt, embeddings, indexing, search, etc.).
    logger: Logger #: :class:`logging.Logger` : Logger for lifecycle and diagnostics.
    llm_manager: LLMManager #: :class:`~src.agentic_rag.managers.llm_manager.LLMManager` :  LLM manager exposing ``router``, ``generator``, and ``grader`` :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`.
    llm: BaseChatModel
    router_config: RunnableConfig
    generator_config: RunnableConfig
    grader_config: RunnableConfig
    node_factory: NodeFactory
    search_manager: SearchManager #: :class:`~src.agentic_rag.managers.search_manager.SearchManager` : Index loader and retriever factory for the vector store.
    retriever: VectorStoreRetriever #: :langchain_core:`VectorStoreRetriever <vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html>` : Configured retriever if the vector store exists.
    examples: List[Dict[str, str]] #: :class:`list` : Few-shot examples loaded for the router prompt.
    memory: AgentMemory #: :class:`~src.agentic_rag.persistence.message.AgentMemory` : Store + checkpointer used by :langgraph:`LangGraph <reference/graphs>`.
    graph: StateGraph #: :langgraph:`StateGraph <reference/graphs/?h=state#langgraph.graph.state.StateGraph>` : Declarative graph (nodes + edges) before compilation.
    compiler: CompiledStateGraph #: :langgraph:`CompiledStateGraph <reference/graphs/?h=compiled#langgraph.graph.state.CompiledStateGraph>` : Executable state machine with persistence.
    def __init__(self, agent_id: str, config_path: str = None, search_manager: Optional[SearchManager] = None):
        self.agent_id = agent_id
        self.config = AgentConfigManager(
            app_id=self.agent_id, config_path=config_path).config
        self.agent_name = self.config.name
        self.logger = get_logger(f"Agent {self.agent_name}")

        self.llm_manager = LLMManager(self.config.llm)
        self.llm = self.llm_manager.llm
        self.router_config = RunnableConfig(configurable={"temperature": self.llm_manager.config.router_temperature})
        self.generator_config = RunnableConfig(configurable={"temperature": self.llm_manager.config.generator_temperature})
        self.grader_config = RunnableConfig(configurable={"temperature": self.llm_manager.config.grader_temperature})

        self.nodes_config = self._load_nodes_config()
        self.node_factory = NodeFactory(prompt_folder=self.config.nodes.prompts)

        # self.summarizer = SummarizationMiddleware(model=self.llm, messages_to_keep=5)
        self.search_manager = SearchManager(
            index_config=self.config.indexing,
            search_config=self.config.search,
        ) if search_manager is None else search_manager
        # self.examples = self._load_examples()
        self.memory = AgentMemory()

        self.graph: StateGraph = self._create_graph()
        self.compiled_graph: CompiledStateGraph = self.graph.compile(
            checkpointer=self.memory.checkpointer,
            # store=self.memory.store
        )

        self.logger.info("Agent initialization completed")

    def _load_nodes_config(self) -> List[NodeConfig]:
        nodes_config_dict = load_yaml(self.config.nodes.config)
        return [NodeConfig.from_config(node_config) for node_config in nodes_config_dict["nodes"]]

    def _get_node_config(self, name: str) -> NodeConfig:
        return next(node_config for node_config in self.nodes_config if node_config.name == name)

    def _load_examples(self) -> List[Dict[str, str]]:
        # TODO: few shot examples should be moved in a more appropriate place
        with open(pathlib.Path.cwd() / self.config.examples.file) as f:
            examples = json.load(f)
        return examples

    def draw_graph(self, filename: str = None) -> None:
        if not filename:
            filename = f"{type(self).__name__}.txt"
        mermaid_syntax = self.compiled_graph.get_graph().draw_mermaid()
        with open(filename, "w") as file:
            file.write(mermaid_syntax)

    def _retrieve(self, state: GraphState) -> Dict:
        question = state["question"]
        self.logger.info(f"Retrieving documents for contextualized question: {question}...")
        documents: SearchResult = self.search_manager.search(question)
        self.logger.info(f"Retrieved {len(documents.chunks)} documents")
        return {"documents": documents}

    def _retrieval_grader(self, state: GraphState) -> Dict:
        self.logger.info(f"Grading {len(state['documents'].chunks)} retrieved documents")
        question = state["question"]
        documents = state["documents"].chunks
        # doc_filter = LLMChainFilter.from_llm(llm=self.llm, prompt=None)
        # filtered_docs = doc_filter.compress_documents(documents=documents, query=question)
        # documents_content = [doc.page_content for doc in state["documents"].chunks]
        # documents_filename = [doc.metadata["filename"] for doc in state["documents"].chunks]  # TODO: check correctness
        input_context = "\n\n".join(f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents))
        runnable = self.node_factory.build_node_from_config(
             self._get_node_config("retrieval_grader"), self.llm,
             structured_output=True, output_schema=output.GradeDocumentsBatch
        )
        response = runnable.invoke(
                     {"question": question, "documents": input_context},
                     config=self.grader_config, with_retry=True,
                 )
        grades = response.grades
        assert len(grades) == len(documents), self.logger.error(f"Length mismatch between sources and grades.")
        filtered_docs = [doc for i, doc in enumerate(documents) if grades[i].binary_score == "yes"]
        # filtered_docs = []
        # for idx, d in enumerate(state["documents"].chunks):
        #     try:
        #         response = runnable.invoke(
        #             {"question": question, "document": documents_content[idx],
        #              "document_filename": documents_filename[idx]},
        #             config=self.grader_config, with_retry=True,
        #         )
        #         grade = response.binary_score
        #         if grade == "yes":
        #             self.logger.info(f"Document {idx + 1} ({documents_filename[idx]}) is relevant to the question.")
        #             filtered_docs.append(d)
        #         else:
        #             self.logger.info(f"Document {idx + 1} ({documents_filename[idx]}) is not relevant to the question.")
        #     except Exception as e:
        #         self.logger.warning(f"Error grading document {idx}: {e}, assuming relevant")
        #         filtered_docs.append(d)
        return {"documents": SearchResult(chunks=list(filtered_docs))}

    def _generate(self, state: GraphState) -> Dict:
        self.logger.info("Generating answer.")
        question = state["question"]
        documents = state["documents"].chunks
        language = state["language"]
        retrieved_docs_as_context = [(f"Filename: {doc.metadata['filename']}\n"
                                      f"Content: {doc.page_content}") for doc in documents]
        context = "\n\n".join([string for string in retrieved_docs_as_context])
        runnable = self.node_factory.build_node_from_config(
            self._get_node_config("generator"), self.llm, structured_output=False,
        )
        response = runnable.invoke(
            {"documents": context, "question": question, "language": language},
            config=self.generator_config,
        )
        return {"response": response, "messages": [AIMessage(content=response)]}


    def _create_graph(self) -> StateGraph:
        """Declare the LangGraph nodes and edges and return the graph.

        Returns
        -------
        :langgraph:`StateGraph <reference/graphs/?h=state#langgraph.graph.state.StateGraph>`
            Graph with nodes/edges set up and terminal edges to :langgraph:`END <reference/constants/?h=end#langgraph.constants.END>`.
        """
        graph = StateGraph(GraphState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("retrieval_grader", self._retrieval_grader)
        graph.add_node("generate", self._generate)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "retrieval_grader")
        graph.add_edge("retrieval_grader", "generate")
        graph.add_edge("generate", END)

        return graph

    def error_handler(self, exception: str) -> Dict:
        """Generate a user-friendly error message via the error handler node.

        Parameters
        ----------
        exception : :class:`str`
            Exception text to summarize for the user.

        Returns
        -------
        :class:`dict`
            Mapping with key ``response`` holding the error message.
        """
        self.logger.info(f"Error Handler Node. {exception}")
        runnable = self.node_factory.build_node_from_config(
            self._get_node_config("error_handler"), self.llm, structured_output=False,
        )
        response = runnable.invoke(
            {"exception": exception, "languages": self.config.allowed_languages},
            config=self.generator_config,
        )
        return {"response": response}

    def _convert_conversation_to_messages(self, conversation: ConversationRequest) -> List[AnyMessage]:
        """Convert a :class:`~src.agentic_rag.utils.chat.ConversationRequest` into LangChain messages.

        Parameters
        ----------
        conversation : :class:`~src.agentic_rag.utils.chat.ConversationRequest`
            Container with history and the current question.

        Returns
        -------
        list of :class:`~langchain_core.messages.base.AnyMessage`
            Tail slice of messages limited by :attr:`~src.agentic_rag.agent.graph.Agent.config` ``.memory.length``.
        """
        messages: List[AnyMessage] = []
        for message in conversation.history:
            if message.role == "user":
                messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                messages.append(AIMessage(content=message.content))
        if conversation.question.role == "user":
            messages.append(HumanMessage(content=conversation.question.content))
        elif conversation.question.role == "assistant":
            messages.append(AIMessage(content=conversation.question.content))
        else:
            messages.append(AnyMessage(content=conversation.question.content))
        return messages[- 2 * self.config.memory.length:]

    async def answer(self, request: ChatRequest, step_logger_fn = None) -> ChatResponse:
        """Run the compiled graph for a user request and return a response.

        Parameters
        ----------
        request : :class:`~src.agentic_rag.utils.chat.ChatRequest`
            Top-level request containing user info and conversation payload.
        step_logger_fn : function, optional
            Logger to trace graph execution steps.

        Returns
        -------
        :class:`~src.agentic_rag.utils.chat.ChatResponse`
            Assistant response with metadata about sources, generation count, and contextual question.
        """
        config: RunnableConfig = \
            {"configurable": {"user_id": request.user_id, "thread_id": request.conversation.id}}
        # memories = nodes.search_memory(question, config, self.store)
        input_state: GraphState = {
                "question": request.conversation.question.content,
                "messages": self._convert_conversation_to_messages(request.conversation),
                "language": self.config.language,
            }
        self.logger.info(f"User {request.user} in conversation {request.conversation.id} sent a request:"
                         f" {request.conversation.question.content}")
        try:
            is_faulted = False
            # response = self.compiled_graph.invoke(
            #     input=input_state,
            #     config=config
            # )
            response = {}
            current_event_list = []
            for event in self.compiled_graph.stream(input=input_state, config=config, stream_mode="debug"):  # type: ignore
                current_event_list.append(event)
                if event["type"]== "checkpoint":
                    if step_logger_fn is not None:
                        try:
                            await step_logger_fn(current_event_list)
                        except Exception as e:
                            self.logger.error(f"Exception while logging step: {e}")
                    if len(event["payload"]["metadata"].get("next", [])) == 0:
                        response = event["payload"].get("values", {})
                    current_event_list = []
            sources = []
            unique_sources = response.get("documents", SearchResult()).extract_unique_chunks()
            for document in unique_sources:
                document_info = {  # TODO: maybe it wil be worth adding more info about retrieved sources
                    "filename": document.metadata.get("filename", "unknown"),
                    "chunk_idx": document.metadata.get("chunk_idx", "unknown"),
                    "headers": document.metadata.get("headers", []),
                }
                sources.append(document_info)
        except Exception as e:
            sources = []
            self.logger.error(f"Error processing request: {str(e)}")
            response = self.error_handler(str(e))
            is_faulted = True
        return ChatResponse(
            role="assistant",
            content=response["response"],
            metadata={
                "sources": sources,
                "contextual_question": response.get("question"),
            },
            is_faulted=is_faulted
        )


