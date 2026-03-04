from abc import ABC, abstractmethod
from logging import Logger
from typing import List, Tuple, Dict

from langchain_core.messages import SystemMessage, RemoveMessage, AnyMessage, HumanMessage, AIMessage
from lingua import Language, LanguageDetectorBuilder

from agentic_rag.agent.state import GraphState
from agentic_rag.agent.services.llm_service import LLMService
from agentic_rag.utils.chat import ConversationRequest

lang_dict = {
    "en": Language.ENGLISH,
    "it": Language.ITALIAN,
    "fr": Language.FRENCH,
    "de": Language.GERMAN,
}
lang_code_dict = {
    Language.ENGLISH: "english",
    Language.ITALIAN: "italian",
    Language.FRENCH: "french",
    Language.GERMAN: "german",
}


class ContextService:
    def __init__(self, llm_service: LLMService, allowed_languages: List[str]):
        self.llm_service = llm_service
        self.allowed_languages = allowed_languages

    def contextualize_question(self, question: str, summary: str, history: List[AnyMessage]) -> str:
        if len(history) <= 1:
            return question
        serialized_history = self.llm_service.serialize_history(messages=history)
        print(len(history), len(serialized_history))
        runnable = self.llm_service.build_node("question_rewriter", structured_output=False)
        result = runnable.invoke({"messages": serialized_history, "summary": summary, "question": question})
        return result

    def summarize_history(self, history: List, summary: str, messages_to_keep: int = 5) -> Tuple[str, List]:
        system_messages = [msg for msg in history if isinstance(msg, SystemMessage)]
        conversation_messages = [msg for msg in history if not isinstance(msg, SystemMessage)]
        if len(conversation_messages) <= messages_to_keep:
            return summary, history
        messages_to_summarize = conversation_messages[:-messages_to_keep]
        messages_in_history = conversation_messages[-messages_to_keep:]
        serialized_history = self.llm_service.serialize_history(messages=messages_to_summarize)
        runnable = self.llm_service.build_node("history_summarizer", structured_output=False)
        result = runnable.invoke({"summary": summary, "messages": serialized_history})
        messages_to_remove = [RemoveMessage(id=m.id) for m in messages_to_summarize]
        summary_message = SystemMessage(content=f"Summary of previous conversation: \n {result}")
        return result, messages_to_remove + system_messages + [summary_message] + messages_in_history

    def detect_language(self, question: str) -> str:
        allowed_language_code = [lang_dict[lang] for lang in self.allowed_languages]
        language_detector = LanguageDetectorBuilder.from_languages(*allowed_language_code).build()
        language_code = language_detector.detect_language_of(question)
        return lang_code_dict.get(language_code) or "english"

    def handle_error(self, exception: str, languages: List[str]):
        runnable = self.llm_service.build_node("error_handler", structured_output=False)
        response = runnable.invoke({"exception": exception, "languages": languages})
        return response

    @staticmethod
    def convert_conversation_to_messages(conversation: ConversationRequest) -> List[AnyMessage]:
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
        return messages


class ContextServiceNode(ABC):
    def __init__(self, context_service: ContextService, logger: Logger):
        self.context_service = context_service
        self.logger = logger

    @abstractmethod
    def __call__(self, state: GraphState) -> Dict:
        pass


class ContextualizeQuestion(ContextServiceNode):
    def __call__(self, state: GraphState) -> Dict:
        self.logger.info("Generating contextual question...")
        contextual_question = self.context_service.contextualize_question(
            question=state["question"], summary=state["summary"], history=state["messages"]
        )
        self.logger.info(f"Contextual question: {contextual_question}")
        return {"contextual_question": contextual_question}


class SummarizeHistory(ContextServiceNode):
    def __call__(self, state: GraphState) -> Dict:
        self.logger.info("Generating summary of previous history...")
        summary = state.get("summary") or ""
        updated_summary = self.context_service.summarize_history(
            history=state["messages"], summary=summary, messages_to_keep=5,
        )
        self.logger.info(f"Summary of previous conversation generated.")
        return {"summary": updated_summary}


class DetectLanguage(ContextServiceNode):
    def __call__(self, state: GraphState) -> Dict:
        self.logger.info("Detecting language...")
        detected_language = self.context_service.detect_language(state["question"])
        self.logger.info(f"Detected language: {detected_language}")
        return {"language": detected_language}
