import json
import pathlib
import datetime
from logging import Logger
from typing import TypedDict, Dict, List, Annotated, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnableBinding, Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_core.messages import HumanMessage, AnyMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from cardiology_gen_ai.utils.logger import get_logger

from src.config.manager import AgentConfigManager, AgentConfig
from src.managers.llm_manager import LLMManager
from src.managers.search_manager import SearchManager
from src.persistence.message import AgentMemory
from src.agent import nodes
from src.utils.chat import ChatRequest, ConversationRequest, MessageSchema, ChatResponse


GENERATION_LIMIT = 2


class GraphState(TypedDict, total=False):
    """Shared state passed between :mod:`langgraph` nodes."""
    question: str #: :class:`str` : Original user question as received by the agent.
    contextual_question: str #: :class:`str` : Question enriched with context (if needed) for retrieval/generation.
    transform_query_count: int #: :class:`int` : How many times the question has been rewritten so far.
    response: str #: :class:`str` : Latest assistant response (when available).
    language: Optional[str] #: :class:`str`, optional : Language used in the conversation.
    messages: Annotated[List[AnyMessage], add_messages] #: :class:`list` of :class:`~langchain_core.messages.base.AnyMessage` : Rolling chat history used for context.
    documents: Optional[List[Document]] #: :class:`list` of :class:`~langchain_core.documents.Document`, optional : Retrieved and filtered documents (when applicable).
    document_request: str #: :class:`str` :  Binary flag to denote whether the user is asking for an entire document.
    generation_count: int #: :class:`int` : Number of generation attempts in the current turn.


class Agent:
    """RAG/conversational agent orchestrated with :mod:`langgraph`.

    The agent connects an LLM manager (router/generator/grader), a vector store
    search manager (retriever), and a memory/checkpoint backend to compile
    a :class:`~langgraph.graph.state.CompiledStateGraph` that handles
    conversation turns end-to-end.

    Parameters
    ----------
    agent_id : :class:`str`
        Identifier used to load configuration via :class:`~src.config.manager.AgentConfigManager`.
    """
    agent_id: str #: :class:`str` : Identifier of this agent instance.
    agent_name: str #: :class:`str` : Human-friendly name from configuration.
    config: AgentConfig #: :class:`~src.config.manager.AgentConfig` : Loaded configuration (system prompt, embeddings, indexing, search, etc.).
    logger: Logger #: :class:`logging.Logger` : Logger for lifecycle and diagnostics.
    llm_manager: LLMManager #: :class:`~src.managers.llm_manager.LLMManager` :  LLM manager exposing ``router``, ``generator``, and ``grader`` :class:`~langchain_core.runnables.Runnable`.
    router: Runnable #: :class:`~langchain_core.runnables.Runnable` : Runnable for routing queries.
    generator: Runnable #: :class:`~langchain_core.runnables.Runnable` : Runnable for answer generation.
    grader: Runnable #: :class:`~langchain_core.runnables.Runnable` : Runnable for grading/validation of retrieved context and generated chunks.
    search_manager: SearchManager #: :class:`~src.managers.search_manager.SearchManager` : Index loader and retriever factory for the vector store.
    retriever: VectorStoreRetriever #: :class:`~langchain_core.vectorstores.VectorStoreRetriever` : Configured retriever if the vector store exists.
    examples: Dict[str, str] #: :class:`list` : Few-shot examples loaded for the router prompt.
    memory: AgentMemory #: :class:`~src.persistence.message.AgentMemory` : Store + checkpointer used by :mod:`langgraph`.
    graph: StateGraph #: :class:`~langgraph.graph.StateGraph` : Declarative graph (nodes + edges) before compilation.
    compiler: CompiledStateGraph #: :class:`~langgraph.graph.state.CompiledStateGraph` : Executable state machine with persistence.
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.config = AgentConfigManager(app_id=self.agent_id).config
        self.agent_name = self.config.name
        self.logger = get_logger(f"Agent {self.agent_name}")

        self.llm_manager = LLMManager(self.config.llm)
        self.router = self.llm_manager.router
        self.generator = self.llm_manager.generator
        self.grader = self.llm_manager.grader

        self.search_manager = SearchManager(
            index_config=self.config.indexing,
            search_config=self.config.search,
            embeddings=self.config.embeddings
        )
        self.retriever = self.search_manager.vectorstore.retriever if self.search_manager.vectorstore.vectorstore_exists() else None

        self.examples = self._load_examples()

        self.memory = AgentMemory()
        
        self.graph: StateGraph = self._create_graph()
        self.compiled_graph: CompiledStateGraph = self.graph.compile(
            checkpointer=self.memory.checkpointer,
            store=self.memory.store
        )

        self.logger.info("Agent initialization completed")

    def _load_examples(self):
        """Load few-shot examples for router prompting.

        Returns
        -------
        :class:`list`
            Examples loaded from the path configured at :attr:`config` ``.examples.file``.
        """
        # TODO: few shot examples should be moved in a more appropriate place
        with open(pathlib.Path.cwd() / self.config.examples.file) as f:
            examples = json.load(f)
        return examples

    def draw_graph(self, filename: str = None) -> None:
        """Export the compiled graph as Mermaid syntax.

        Parameters
        ----------
        filename : :class:`str`, optional
            Destination file path.
        """
        if not filename:
            filename = f"{type(self).__name__}.txt"
        mermaid_syntax = self.compiled_graph.get_graph().draw_mermaid()
        with open(filename, "w") as file:
            file.write(mermaid_syntax)

    def _detect_language(self, state: GraphState) -> Dict:
        """Detect the language of the current question.

        Parameters
        ----------
        state : :class:`~src.agent.GraphState`
            Current state; must include ``question``.

        Returns
        -------
        :class:`dict`
            Mapping with key ``\"language\"`` set to the detected language.
        """
        self.logger.info("Detecting language...")
        question = state["question"]
        runnable = nodes.detect_language(self.generator)
        response = runnable.invoke({"text": question})
        language = response.language
        self.logger.info(f"Detected language: {language}")
        return {"language": language}

    def _conversational_agent(self, state: GraphState) -> Dict:
        """Answer a conversational query without retrieval.

        Parameters
        ----------
        state : :class:`~src.agent.GraphState`
            Must contain ``question``, optionally ``messages`` and ``language``.

        Returns
        -------
        :class:`dict`
            Keys: ``response`` (assistant text) and ``messages`` (list with new :class:`~langchain_core.messages.AIMessage`).
        """
        self.logger.info("Agent is ready to answer questions")
        agent_prompt = self.config.system_prompt
        language = state["language"] or self.config.language
        question = state["question"]
        messages = state["messages"]
        history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[:-1]]) if messages else ""
        runnable = nodes.conversational_agent(llm=self.generator, agent_prompt=agent_prompt)
        response = runnable.invoke({"question": question, "language": language, "history": history})
        return {"response": response, "messages": [AIMessage(content=response)]}

    def _contextualize_question(self, state: GraphState) -> Dict:
        """Add minimal context to the question based on history (if needed).

        Parameters
        ----------
        state : :class:`~src.agent.GraphState`
            Must contain ``question``, ``language``, and ``messages``.

        Returns
        -------
        :class:`dict`
            Keys: ``contextual_question``, ``generation_count`` (set to 0), ``transform_query_count`` (set to 0).
        """
        self.logger.info("Generating contextual question...")
        self.logger.info(f"Original question: {state['question']}")
        question = state["question"]
        language = state["language"]
        messages = state["messages"]
        runnable = nodes.contextualize_question(self.generator, self.config.context.system_prompt)
        response = runnable.invoke(
            {"question": question, "language": language, "history": messages}
        )
        self.logger.info(f"Contextual question: {response}")
        return {"generation_count": 0, "transform_query_count": 0, "contextual_question": response}
    
    def _retrieve(self, state: GraphState) -> Dict:
        """Retrieve candidate documents for the contextualized question.

        Parameters
        ----------
        state : :class:`~src.agent.GraphState`
            Must include ``contextual_question``.

        Returns
        -------
        :class:`dict`
            Key ``documents`` with a list of :class:`~langchain_core.documents.Document`.
        """
        question = state["contextual_question"]
        self.logger.info(f"Retrieving documents for contextualized question: {question}...")

        if self.retriever is None:
            self.logger.info("No vectorstore available. Returning empty document list.")
            return {"documents": []}

        documents = self.retriever.invoke(question)
        self.logger.info(f"Retrieved {len(documents)} documents")
        return {"documents": documents}

    def _retrieval_grader(self, state: GraphState) -> Dict:
        """Filter retrieved documents by relevance using the grader runnable.

        Parameters
        ----------
        state : :class:`~src.agent.GraphState`
            Must include ``contextual_question`` and ``documents``.

        Returns
        -------
        :class:`dict`
            Key ``documents`` with only relevant items preserved.
        """
        self.logger.info(f"Grading {len(state['documents'])} retrieved documents")
        question = state["contextual_question"]
        documents_content = [doc.page_content for doc in state["documents"]]
        documents_filename = [doc.metadata['filename'] for doc in state["documents"]]  # TODO: check correctness
        runnable = nodes.retrieval_grader(self.grader)
        filtered_docs = []
        for idx, d in enumerate(state["documents"]):
            try:
                response = runnable.invoke({"question": question, "document": documents_content[idx],
                                            "document_filename": documents_filename[idx]})
                grade = response.binary_score
                if grade == "yes":
                    self.logger.info(f"Document {idx + 1} ({documents_filename[idx]}) is relevant to the question.")
                    filtered_docs.append(d)
                else:
                    self.logger.info(f"Document {idx + 1} ({documents_filename[idx]}) is not relevant to the question.")
            except Exception as e:
                self.logger.warning(f"Error grading document {idx}: {e}, assuming relevant")
                filtered_docs.append(d)
        return {"documents": filtered_docs}

    def _document_request_detector(self, state: GraphState) -> Dict:
        """Detect whether the user explicitly requested documents.

        Parameters
        ----------
        state : :class:`~src.agent.GraphState`
            Must include ``contextual_question``.

        Returns
        -------
        :class:`dict`
            Key ``document_request`` with value ``\"yes\"`` or ``\"no\"``.
        """
        self.logger.info("Checking if user question requires a document.")
        question = state["contextual_question"]
        runnable = nodes.document_request_detector(self.router)
        score = runnable.invoke({"question": question})
        binary_score = score.binary_score
        if binary_score == "yes":
            self.logger.info("User question implies a document request.")
        else:
            self.logger.info("The user question does not imply a document request.")
        return {"document_request": binary_score}

    def _decide_to_generate(self, state: GraphState) -> str:
        """Branch selector after grading: choose next step based on document relevance.

        Parameters
        ----------
        state : :class:`~src.agent.graph.GraphState`
            Must include ``documents`` and ``document_request``.

        Returns
        -------
        :class:`str`
            One of:
            - ``\"all_docs_not_relevant\"``
            - ``\"at_least_one_doc_relevant\"``
            - ``\"generate_document_request_response\"``
        """
        filtered_docs = state["documents"]
        if len(filtered_docs) == 0:
            self.logger.info("All documents marked as not relevant")
            return "all_docs_not_relevant"
        else:
            self.logger.info(f"{len(filtered_docs)} documents marked as relevant")
            if state["document_request"] == "no":
                return "at_least_one_doc_relevant"
            else:
                return "generate_document_request_response"

    def _generate_document_response(self, state: GraphState) -> Dict:
        """Generate a polite response acknowledging a document request.

        Parameters
        ----------
        state : :class:`~src.agent.graph.GraphState`
            Must include ``documents``, ``contextual_question``, and ``language``.

        Returns
        -------
        :class:`dict`
            Key ``response`` with assistant text.
        """
        self.logger.info("Generating document response message.")
        documents = state["documents"]
        question = state["contextual_question"]
        files = list(set([doc.metadata["filename"] for doc in documents]))
        language = state["language"]
        runnable = nodes.generate_document_response(self.generator)
        response = runnable.invoke({"question": question, "documents": files, "language": language})
        self.logger.info(f"Generated response: {response}")
        return {"response": response}

    def _generate(self, state: GraphState) -> Dict:
        """Generate an answer grounded in retrieved documents.

        Parameters
        ----------
        state : :class:`~src.agent.graph.GraphState`
            Must include ``contextual_question``, ``documents``, and ``language``.

        Returns
        -------
        :class:`dict`
            Keys: ``response`` (assistant text) and ``generation_count`` (incremented).
        """

        self.logger.info("Generating answer.")
        question = state["contextual_question"]
        documents = state["documents"]
        language = state["language"]
        retrieved_docs_as_context = [(f"Filename: {doc.metadata['filename']}\n"
                                      f"Content: {doc.page_content}") for doc in documents]
        context = "\n\n".join([string for string in retrieved_docs_as_context])
        runnable = nodes.generate(self.generator)
        response = runnable.invoke({"documents": context, "question": question, "language": language})
        return {"response": response, "generation_count": state["generation_count"] + 1,}

    def _question_rewriter(self, state: GraphState) -> Dict:
        """Rewrite the question to improve retrieval when needed.

        Parameters
        ----------
        state : :class:`~src.agent.graph.GraphState`
            Must include ``contextual_question`` and ``transform_query_count``.

        Returns
        -------
        :class:`dict`
            Keys: ``contextual_question`` (rewritten) and ``transform_query_count`` (incremented).
        """
        self.logger.info("Transforming query.")
        question = state["contextual_question"]
        self.logger.info(f"Original question: {question}")
        runnable = nodes.question_rewriter(self.generator)
        response = runnable.invoke({"question": question})
        return {
            "contextual_question": response,
            "transform_query_count": state["transform_query_count"] + 1,
        }

    def _generate_default_response(self, state: GraphState) -> Dict:
        """Produce a safe default reply when generation should stop.

        Parameters
        ----------
        state : :class:`~src.agent.graph.GraphState`
            Must include ``language`` and ``question``.

        Returns
        -------
        :class:`dict`
            Keys: ``response`` (fallback message) and ``documents`` (empty list).
        """
        self.logger.info("Generating default response.")
        language = state["language"]
        question = state["question"]
        runnable = nodes.generate_default_response(self.generator)
        response = runnable.invoke({"language": language, "question": question})
        return {"response": response, "documents": []}

    def _router(self, state: GraphState) -> str | None:
        """Route the contextual question to conversational or document-based flow.

        Parameters
        ----------
        state : :class:`~src.agent.graph.GraphState`
            Must include ``contextual_question``.

        Returns
        -------
        :class:`str` or ``None``
            - ``\"conversational_question\"`` → node ``conversational_agent``
            - ``\"document_based_question\"`` → node ``document_request_detector``
            - ``None`` if no route is determined
        """
        self.logger.info("Routing question.")
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=self.examples,
            embeddings=self.config.embeddings.model,
            vectorstore_cls=FAISS,
            k=self.config.examples.top_k,
            input_keys=self.config.examples.input_keys,
        )
        prompt_template = PromptTemplate.from_template(self.config.examples.template)
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=prompt_template,
            input_variables=["question"],
            prefix="",
            suffix="",
        )
        example_prompt = few_shot_prompt.format(input=state["contextual_question"])
        runnable = nodes.router(self.router, self.config.indexing.description, example_prompt)
        routing = runnable.invoke({"question": state["contextual_question"]})
        if routing.branch == "conversational":
            self.logger.info("Conversational question is routed to the Agent.")
            return "conversational_question"
        elif routing.branch == "document_based":
            self.logger.info("Question is routed to the RAG Architecture.")
            return "document_based_question"
        else:
            self.logger.info("Question is not routed to any branch.")
            return None

    def _validator(self, state: GraphState) -> str:
        """Validate grounding and task resolution; decide next step.

        Parameters
        ----------
        state : :class:`~src.agent.graph.GraphState`
            Must include ``documents``, ``response``, ``generation_count``, and ``contextual_question``.

        Returns
        -------
        :class:`str`
            One of:
            - ``\"grounded_and_addressed_question\"`` → terminal success
            - ``\"generation_not_grounded\"`` → re-generate
            - ``\"grounded_but_not_addressed_question\"`` → rewrite question
            - ``\"generate_default_response\"`` → stop with fallback
        """
        self.logger.info("Checking hallucinations.")
        documents = state["documents"]
        generation = state["response"]
        generation_count = state["generation_count"]
        question = state["contextual_question"]
        docs_string = "\n\n".join([doc.page_content for doc in documents])
        ground_validator_runnable = nodes.ground_validator(self.grader)
        answer_grader_runnable = nodes.answer_grader(self.grader)
        if generation_count <= GENERATION_LIMIT:
            ground_validation = ground_validator_runnable.invoke({"documents": docs_string, "generation": generation})
            ground_validation_score = ground_validation.binary_score
            if ground_validation_score == "yes":
                self.logger.info("Generated answer is grounded in documents.")
                answer_grade = answer_grader_runnable.invoke({"question": question, "generation": generation})
                answer_question = answer_grade.binary_score
                if answer_question == "yes":
                    self.logger.info(f"Generated answer addresses the question.")
                    return "grounded_and_addressed_question"
                else:
                    self.logger.info(f"Generated answer does not address the question.")
                    return "grounded_but_not_addressed_question"
            else:
                self.logger.info("Generated answer is not grounded in documents")
                return "generation_not_grounded"
        else:
            self.logger.info("Generation count exceeds limit. Generating default response...")
            return "generate_default_response"

    @staticmethod
    def _verify_generation_limit(state: GraphState) -> str:
        """Guard on query-rewrite attempts.

        Parameters
        ----------
        state : :class:`~src.agent.graph.GraphState`
            Must include ``transform_query_count``.

        Returns
        -------
        :class:`str`
            ``\"retrieve\"`` if under the limit, else ``\"generate_default_response\"``.
        """
        if state["transform_query_count"] <= GENERATION_LIMIT:
            return "retrieve"
        else:
            return "generate_default_response"
    
    def _create_graph(self) -> StateGraph:
        """Declare the LangGraph nodes and edges and return the graph.

        Returns
        -------
        :class:`~langgraph.graph.StateGraph`
            Graph with nodes/edges set up and terminal edges to :data:`END`.
        """
        graph = StateGraph(GraphState)

        graph.add_node("language_detector", self._detect_language)
        graph.add_node("conversational_agent", self._conversational_agent)
        graph.add_node("contextualize_question", self._contextualize_question)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("retrieval_grader", self._retrieval_grader)
        graph.add_node("transform_question", self._question_rewriter)
        graph.add_node("generate", self._generate)
        graph.add_node("document_request_detector", self._document_request_detector)
        graph.add_node("generate_document_response", self._generate_document_response)
        graph.add_node("generate_default_response", self._generate_default_response)

        graph.add_edge(START, "language_detector")
        graph.add_edge("language_detector", "contextualize_question")
        graph.add_conditional_edges(
            "contextualize_question",
            self._router,
            {
                "conversational_question": "conversational_agent",
                "document_based_question": "document_request_detector",
            }
        )
        graph.add_edge("document_request_detector", "retrieve")
        graph.add_edge("retrieve", "retrieval_grader")
        graph.add_conditional_edges(
            "retrieval_grader",
            self._decide_to_generate,
            {
                "all_docs_not_relevant": "transform_question",
                "at_least_one_doc_relevant": "generate",
                "generate_document_request_response": "generate_document_response",
            },
        )
        graph.add_edge("generate_document_response", END)
        graph.add_conditional_edges(
            "generate",
            self._validator,
            {
                "grounded_and_addressed_question": END,
                "generation_not_grounded": "generate",
                "grounded_but_not_addressed_question": "transform_question",
                "generate_default_response": "generate_default_response",
            }
        )
        graph.add_conditional_edges(
            "transform_question",
            self._verify_generation_limit,
            {
                "retrieve": "retrieve",
                "generate_default_response": "generate_default_response",
            }
        )
        graph.add_edge("generate_default_response", END)
        graph.add_edge("conversational_agent", END)

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
            Mapping with key ``generation`` holding the error message.
        """
        self.logger.info("Error Handler Node.")
        runnable = nodes.error_handler_node(self.generator, self.config.allowed_languages)
        response = runnable.invoke({"exception": exception})
        return {"generation": response}

    def _convert_conversation_to_messages(self, conversation: ConversationRequest) -> List[AnyMessage]:
        """Convert a :class:`~src.utils.chat.ConversationRequest` into LangChain messages.

        Parameters
        ----------
        conversation : :class:`~src.utils.chat.ConversationRequest`
            Container with history and the current question.

        Returns
        -------
        list of :class:`~langchain_core.messages.base.AnyMessage`
            Tail slice of messages limited by :attr:`~src.agent.Agent.config` ``.memory.length``.
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
    
    def answer(self, request: ChatRequest) -> ChatResponse:
        """Run the compiled graph for a user request and return a response.

        Parameters
        ----------
        request : :class:`~src.utils.chat.ChatRequest`
            Top-level request containing user info and conversation payload.

        Returns
        -------
        :class:`~src.utils.chat.ChatResponse`
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
            response = self.compiled_graph.invoke(
                input=input_state,
                config=config
            )
            attachments = {"sources": []}
            unique_sources = []
            if response.get("documents"):
                for document in response["documents"]:
                    document_info = {  # TODO: maybe it wil be worth adding more info about retrieved sources
                        "filename": document.metadata["filename"],
                        "chunk_id": document.metadata["chunk_id"],
                    }
                    attachments["sources"].append(document_info)
                seen_chunks = set()
                print(attachments["sources"])
                for doc_info in attachments["sources"]:
                    if (doc_info["filename"], doc_info["chunk_id"]) not in seen_chunks:
                        unique_sources.append(doc_info)
        except Exception as e:
            unique_sources = []
            self.logger.error(f"Error processing request: {str(e)}")
            response = self.error_handler(str(e))
            is_faulted = True
        return ChatResponse(
            role="assistant",
            content=response["response"],
            metadata={
                "sources": unique_sources,
                "n_gen": response.get("generation_count"),
                "contextual_question": response.get("contextual_question"),
            },
            is_faulted=is_faulted
        )


if __name__ == "__main__":
    agent = Agent("cardiology_protocols")
    chat_request = ChatRequest(
        user="gaia",
        user_id="2",
        conversation=ConversationRequest(
            id="2",
            chatbotId="1",
            history=[],
            question=MessageSchema(
                id="2",
                role="user",
                content="Quale è la cura per la cardiopatia?",
                datetime=datetime.datetime.now(),
            )
        )
    )
    # metadata["chunk_idx"]
    agent_response = agent.answer(chat_request)
    print(agent_response.content)
