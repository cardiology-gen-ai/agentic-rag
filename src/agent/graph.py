import json
import pathlib
import datetime
from typing import TypedDict, Dict, List, Annotated, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_core.messages import HumanMessage, AnyMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from cardiology_gen_ai.utils.logger import get_logger

from src.config.manager import AgentConfigManager
from src.managers.llm_manager import LLMManager
from src.managers.search_manager import SearchManager
from src.persistence.message import AgentMemory
from src.agent import nodes
from src.agent import output
from src.utils.chat import ChatRequest, ConversationRequest, MessageSchema, ChatResponse


GENERATION_LIMIT = 1


class GraphState(TypedDict, total=False):
    question: str
    contextual_question: str
    transform_query_count: int
    response: str
    language: Optional[str]   # TODO: fix language (automatic detection)
    messages: Annotated[List[AnyMessage], add_messages]
    documents: Optional[List[Document]]
    document_request: str
    generation_count: int


class Agent:
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
        self.retriever = self.search_manager.vectorstore.retriever

        self.examples = self._load_examples()

        self.memory = AgentMemory()
        
        self.graph: StateGraph = self._create_graph()
        self.compiled_graph: CompiledStateGraph = self.graph.compile(
            checkpointer=self.memory.checkpointer,
            store=self.memory.store
        )

        self.logger.info("Agent initialization completed")

    def _load_examples(self):
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

    def _conversational_agent(self, state: GraphState) -> Dict:
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
        self.logger.info("Generating contextual question...")
        self.logger.info(f"Original question: {state["question"]}")
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
        question = state["contextual_question"]
        self.logger.info(f"Retrieving documents for contextualized question: {question}...")
        documents = self.retriever.invoke(question)
        self.logger.info(f"Retrieved {len(documents)} documents")
        return {"documents": documents}

    def _retrieval_grader(self, state: GraphState) -> Dict:
        self.logger.info(f"Grading {len(state["documents"])} retrieved documents")
        question = state["contextual_question"]
        documents_content = [doc.page_content for doc in state["documents"]]
        documents_filename = [doc.metadata["filename"] for doc in state["documents"]]  # TODO: check correctness
        runnable = nodes.retrieval_grader(self.grader)
        filtered_docs = []
        for idx, d in enumerate(state["documents"]):
            try:
                response = runnable.invoke({"question": question, "document": documents_content[idx],
                                            "document_filename": documents_filename[idx]})
                assert isinstance(response, output.GradeDocuments)
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
        self.logger.info("Checking if user question requires a document.")
        question = state["contextual_question"]
        runnable = nodes.document_request_detector(self.router)
        score = runnable.invoke({"question": question})
        assert isinstance(score, output.DocumentRequest)
        binary_score = score.binary_score
        if binary_score == "yes":
            self.logger.info("User question implies a document request.")
        else:
            self.logger.info("The user question does not imply a document request.")
        return {"document_request": binary_score}

    def _decide_to_generate(self, state: GraphState) -> str:
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
        self.logger.info("Generating answer.")
        question = state["contextual_question"]
        documents = state["documents"]
        language = state["language"]
        retrieved_docs_as_context = [(f"Filename: {doc.metadata["filename"]}\n"
                                      f"Content: {doc.page_content}") for doc in documents]
        context = "\n\n".join([string for string in retrieved_docs_as_context])
        runnable = nodes.generate(self.generator)
        response = runnable.invoke({"documents": context, "question": question, "language": language})
        return {"response": response, "generation_count": state["generation_count"] + 1,}

    def _question_rewriter(self, state: GraphState) -> Dict:
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
        self.logger.info("Generating default response.")
        language = state["language"]
        question = state["question"]
        runnable = nodes.generate_default_response(self.generator)
        response = runnable.invoke({"language": language, "question": question})
        return {"response": response, "documents": []}

    def _router(self, state: GraphState) -> str | None:
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
        assert isinstance(routing, output.RouteQuery)
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
        self.logger.info("Checking hallucinations.")
        documents = state["documents"]
        generation = state["response"]
        generation_count = state["generation_count"]
        question = state["contextual_question"]
        docs_string = "\n\n".join([doc.page_content for doc in documents])
        ground_validator_runnable = nodes.ground_validator(self.grader)
        answer_grader_runnable = nodes.answer_grader(self.grader)
        if generation_count <= GENERATION_LIMIT:
            ground_validation = ground_validator_runnable.invoke({"documents": docs_string, "response": generation})
            ground_validation = ground_validation.binary_score
            if ground_validation == "yes":
                self.logger.info("Generated answer is grounded in documents.")
                answer_grade = answer_grader_runnable.invoke({"question": question, "generation": generation})
                assert isinstance(answer_grade, output.GradeAnswer)
                answer_question = answer_grade.binary_score
                if answer_question == "yes":
                    self.logger.info(f"Generated answer addresses the question.")
                    return "grounded_and_addresses_question"
                else:
                    self.logger.info(f"Generated answer does not address the question.")
                    return "grounded_but_does_not_address_question"
            else:
                self.logger.info("Generated answer is not grounded in documents")
                return "generation_not_grounded"
        else:
            self.logger.info("Generation count exceeds limit. Generating default response...")
            return "generate_default_response"

    @staticmethod
    def _verify_generation_limit(state: GraphState) -> str:
        if state["transform_query_count"] <= GENERATION_LIMIT:
            return "retrieve"
        else:
            return "generate_default_response"
    
    def _create_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("conversational_agent", self._conversational_agent)
        graph.add_node("contextualize_question", self._contextualize_question)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("retrieval_grader", self._retrieval_grader)
        graph.add_node("transform_question", self._question_rewriter)
        graph.add_node("generate", self._generate)
        graph.add_node("document_request_detector", self._document_request_detector)
        graph.add_node("generate_document_response", self._generate_document_response)
        graph.add_node("generate_default_response", self._generate_default_response)

        graph.add_edge(START, "contextualize_question")
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
        self.logger.info("Error Handler Node.")
        runnable = nodes.error_handler_node(self.generator, self.config.allowed_languages)
        response = runnable.invoke({"exception": exception})
        return {"generation": response}

    def _convert_conversation_to_messages(
            self, conversation: ConversationRequest) -> List[AnyMessage]:
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
    agent.answer(chat_request)
