from abc import ABC, abstractmethod
from logging import Logger
from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from agentic_rag.agent.state import GraphState
from agentic_rag.agent.services.llm_service import LLMService


class GenerationService:
    def __init__(self, llm_service: LLMService, generator_config: RunnableConfig, agent_prompt: str):
        self.llm_service = llm_service
        self.generator_config = generator_config
        self.agent_prompt = agent_prompt

    def generate_rag_response(self, documents: List[Document], question: str, language: str, history: List) -> str:
        context = "\n\n".join(
            [f"Filename: {doc.metadata['filename']}\n Content: {doc.page_content}" for doc in documents]
        )
        runnable = self.llm_service.build_node("generator", structured_output=False)
        result = runnable.invoke(
            {"documents": context, "question": question, "history": history, "language": language},
            config=self.generator_config,
        )
        return result

    def generate_default_response(self, question: str, language: str) -> str:
        runnable = self.llm_service.build_node("default_response_generator", structured_output=False)
        response = runnable.invoke({"language": language, "question": question}, config=self.generator_config)
        return response

    def generate_document_response(self, question: str, files: List[str], language: str) -> str:
        runnable = self.llm_service.build_node("document_response_generator", structured_output=False)
        response = runnable.invoke(
            {"question": question, "documents": files, "language": language},
            config=self.generator_config,
        )
        return response

    def generate_conversational_response(self, question: str, language: str, history: List) -> str:
        runnable = self.llm_service.build_node(
            "conversational_agent", structured_output=False, agent_prompt=self.agent_prompt,
        )
        response = runnable.invoke(
            {"question": question, "language": language, "history": history}, config=self.generator_config,
        )
        return response


class GenerationServiceNode(ABC):
    def __init__(self, generation_service: GenerationService, logger: Logger):
        self.generation_service = generation_service
        self.logger = logger

    @abstractmethod
    def __call__(self, state: GraphState) -> Dict:
        pass


class GenerateRagResponse(GenerationServiceNode):
    def __call__(self, state: GraphState, **kwargs) -> Dict:
        self.logger.info("Generating answer based on retrieved documents...")
        documents = state["documents"]
        response = self.generation_service.generate_rag_response(
            documents=documents.chunks, question=state["question"], language=state["language"], history=state["messages"],
        )
        formatted_sources = documents.format_sources()
        if formatted_sources:
            response = f"{response}\n\n\n---\n\n{formatted_sources}"
        self.logger.info(f"Generated response: {response}")
        return {"response": response, "generation_count": state["generation_count"] + 1}


class GenerateDefaultResponse(GenerationServiceNode):
    def __call__(self, state: GraphState) -> Dict:
        self.logger.info("Generating default response...")
        response = self.generation_service.generate_default_response(
            question=state["question"], language=state["language"],
        )
        return {"response": response, "generation_count": state["generation_count"] + 1}


class GenerateDocumentResponse(GenerationServiceNode):
    def __call__(self, state: GraphState) -> Dict:
        self.logger.info("Generating document response message...")
        documents = state["documents"].extract_unique_filenames()
        # TODO: maybe add history in the prompt and substitute contextual_question with question
        response = self.generation_service.generate_document_response(
            question=state["contextual_question"], files=documents, language=state["language"],
        )
        self.logger.info(f"Generated response: {response}")
        return {"response": response, "generation_count": state["generation_count"] + 1}


class GenerateConversationalResponse(GenerationServiceNode):
    def __call__(self, state: GraphState) -> Dict:
        self.logger.info("Agent is answering conversational question...")
        response = self.generation_service.generate_conversational_response(
            question=state["question"], language=state["language"], history=state["messages"],
        )
        return {"response": response, "generation_count": state["generation_count"] + 1}