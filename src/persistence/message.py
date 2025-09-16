import os
import uuid
from dataclasses import field
from datetime import datetime
from typing import Dict, Any, Optional, List
import asyncio

from psycopg import connect
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from cardiology_gen_ai.utils.logger import get_logger

from src.managers.llm_manager import LLMManager
from src.utils.chat import ChatResponse, ChatRequest


logger = get_logger("Agent memory management based on LangGraph")


class ConversationTurn(BaseModel):
    message_id: uuid.UUID | str  # turn_id
    session_id: uuid.UUID | str # conversation_id
    question: str
    metadata: Dict = {}
    response: str | None = ""
    created_at: datetime
    error: str | None = None

    def model_post_init(self, __context: Any):
        self.message_id = uuid.UUID(str(self.message_id))
        self.session_id = uuid.UUID(str(self.session_id))

    @classmethod
    def from_agent(cls, response: ChatResponse, request: ChatRequest) -> "ConversationTurn":
        return ConversationTurn(
            message_id=request.conversation.question.id,
            session_id=request.conversation.id,
            question=request.conversation.question.content,
            metadata=response.metadata,
            response=response.content,
            created_at=request.conversation.question.datetime,
        )


class RetrievalTurn(BaseModel):
    message_id: uuid.UUID | str
    session_id: uuid.UUID | str
    question: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    embedding_name: str = ""

    def model_post_init(self, __context: Any):
        self.message_id = uuid.UUID(str(self.message_id))
        self.session_id = uuid.UUID(str(self.session_id))

    @classmethod
    def from_agent(cls, response: ChatResponse, request: ChatRequest, embedding_name: str) -> "RetrievalTurn":
        return RetrievalTurn(
            message_id=request.conversation.question.id,
            session_id=request.conversation.id,
            question=request.conversation.question.content,
            sources=response.metadata["sources"],
            embedding_name=embedding_name,
        )


class LLMTurn(BaseModel):
    message_id: uuid.UUID | str
    session_id: uuid.UUID | str
    model_name: str
    generator_temperature: float = 0
    router_temperature: float = 0
    grader_temperature: float = 0
    token_used: int | None = None
    latency_ms: float | None = None

    def model_post_init(self, __context: Any):
        self.message_id = uuid.UUID(str(self.message_id))
        self.session_id = uuid.UUID(str(self.session_id))

    @classmethod
    def from_agent(cls, response: ChatResponse, request: ChatRequest, llm_manager: LLMManager,
                   duration: float = None) -> "LLMTurn":
        return LLMTurn(
            message_id=request.conversation.question.id,
            session_id=request.conversation.id,
            model_name=llm_manager.config.model_name,
            generator_temperature=llm_manager.config.generator_temperature,
            router_temperature=llm_manager.config.router_temperature,
            grader_temperature=llm_manager.config.grader_temperature,
            token_used=llm_manager.count_tokens([request.conversation.question.content, response.content]),
            latency_ms=duration
        )


class FeedbackRequest(BaseModel):
    feedback_value: int
    feedback_message: str | None = None


class FeedbackTurn(BaseModel):
    message_id: uuid.UUID | str
    session_id: uuid.UUID | str
    user_id: uuid.UUID | str
    feedback_value: int | None = 0
    feedback_message: str | None = None
    created_at: datetime

    def model_post_init(self, __context: Any):
        self.message_id = uuid.UUID(str(self.message_id))
        self.session_id = uuid.UUID(str(self.session_id))
        self.user_id = uuid.UUID(str(self.user_id))


class AgentMemory:
    def __init__(self, db_connection_string: Optional[str] = None):
        db_connection_string = db_connection_string or os.getenv("POSTGRES_ADMIN_DSN")
        if not db_connection_string:
            raise ValueError("No database connection string provided")
        self.connection = connect(conninfo=db_connection_string, autocommit=True, row_factory=dict_row)
        self.store = PostgresStore(self.connection)
        self.store.setup()
        self.checkpointer = PostgresSaver(self.connection)
        self.checkpointer.setup()

    def save_conversation_turn(self, conversation_turn: ConversationTurn):
        try:
            self.store.put(("conversation", str(conversation_turn.session_id)),
                           str(conversation_turn.message_id),
                           conversation_turn.model_dump(exclude_none=False, mode="json"))
        except Exception as e:
            logger.info(f"Failed to save conversation: {e}")

    @staticmethod
    def item_to_conversation_turn(item: Any) -> ConversationTurn:
        v = item.value or {}
        session_from_ns = item.namespace[1] if getattr(item, "namespace", None) and len(item.namespace) > 1 else None
        payload = {
            "message_id": v.get("message_id") or item.key,
            "session_id": v.get("session_id") or session_from_ns,
            "question": v.get("question") or "",
            "metadata": v.get("metadata") or {},
            "response": v.get("response"),
            "created_at": v.get("created_at") or item.created_at,
            "error": v.get("error"),
        }
        return ConversationTurn.model_validate(payload)

    def get_history(self, session_id: uuid.UUID, limit: int) -> List[ConversationTurn]:
        conversation_items = self.store.search(("conversation", str(session_id)), limit=1000)
        if len(conversation_items) == 0 or conversation_items is None:
            return []
        conversation_items = [self.item_to_conversation_turn(item) for item in conversation_items]
        conversation_items_sorted = sorted(conversation_items, key=lambda item: item.created_at, reverse=True)
        return conversation_items_sorted[:limit]

    def save_retrieval_turn(self, retrieval_turn: RetrievalTurn):
        try:
            self.store.put(("retrieval", str(retrieval_turn.session_id), str(retrieval_turn.message_id)),
                           "results",
                           retrieval_turn.model_dump(exclude_none=False, mode="json"))
        except Exception as e:
            logger.info(f"Failed to save retriever results: {e}")

    def save_llm_turn(self, llm_turn: LLMTurn):
        try:
            self.store.put(("llm", str(llm_turn.session_id), str(llm_turn.message_id)),
                           "results",
                           llm_turn.model_dump(exclude_none=False, mode="json"))
        except Exception as e:
            logger.info(f"Failed to save LLM information: {e}")

    def save_feedback(self, feedback: FeedbackTurn):
        try:
            self.store.put(("feedback", str(feedback.session_id)),
                            str(feedback.message_id),
                            feedback.model_dump(exclude_none=False, mode="json"))
        except Exception as e:
            logger.info(f"Failed to save feedback: {e}")

    def get_session_feedback(self, session_id: uuid.UUID) -> List[FeedbackTurn]:
        session_items = self.store.search(("feedback", str(session_id)), limit=1000)
        session_feedback: List[FeedbackTurn] = []
        for it in session_items:
            session_feedback.append(FeedbackTurn.model_validate(it.value))
        return session_feedback

    def _delete_prefix(self, ns_prefix: tuple, batch_size: int = 500) -> None:
        while True:
            batch = self.store.search(ns_prefix, limit=batch_size)
            if not batch:
                break
            for it in batch:
                self.store.delete(it.namespace, it.key)

    def delete_session(self, session_id: uuid.UUID) -> None:
        sid = str(session_id)
        self._delete_prefix(("conversation", sid))
        self._delete_prefix(("retrieval", sid))
        self._delete_prefix(("llm", sid))
        self._delete_prefix(("feedback", sid))


class AsyncAgentMemory:
    def __init__(self, store: AsyncPostgresStore, checkpointer: AsyncPostgresSaver):
        self.store = store
        self.checkpointer = checkpointer

    @classmethod
    async def create(cls, db_connection_string: Optional[str] = None) -> "AsyncAgentMemory":
        db_connection_string = db_connection_string or os.getenv("POSTGRES_ADMIN_DSN")
        if not db_connection_string:
            raise ValueError("No database connection string provided")
        store_manager = AsyncPostgresStore.from_conn_string(db_connection_string)
        store = await store_manager.__aenter__()
        await store.setup()
        checkpointer_manager = AsyncPostgresSaver.from_conn_string(db_connection_string)
        checkpointer = await checkpointer_manager.__aenter__()
        await checkpointer.setup()
        return cls(store, checkpointer)

    async def _delete_prefix(self, ns_prefix: tuple, batch_size: int = 500) -> None:
        while True:
            batch = await self.store.asearch(ns_prefix, limit=batch_size)
            if not batch:
                break
            for it in batch:
                await self.store.adelete(it.namespace, it.key)

    async def save_conversation_turn(self, conversation_turn: ConversationTurn) -> None:
        await self.store.aput(("conversation", str(conversation_turn.session_id)),
                              str(conversation_turn.message_id),
                              conversation_turn.model_dump(exclude_none=False, mode="json"))

    async def save_retrieval_turn(self, retrieval_turn: RetrievalTurn) -> None:
        await self.store.aput(("retrieval", str(retrieval_turn.session_id), str(retrieval_turn.message_id)),
                              "results", retrieval_turn.model_dump(exclude_none=False, mode="json"))

    async def save_llm_turn(self, llm_turn: LLMTurn) -> None:
        await self.store.aput(("llm", str(llm_turn.session_id), str(llm_turn.message_id)),
                              "results", llm_turn.model_dump(exclude_none=False, mode="json"))

    async def save_feedback(self, feedback: FeedbackTurn) -> None:
        await self.store.aput( ("feedback", str(feedback.session_id)),
                               str(feedback.message_id),
                               feedback.model_dump(exclude_none=False, mode="json"))

    async def get_session_feedback(self, session_id: uuid.UUID) -> List[FeedbackTurn]:
        items = await self.store.asearch(("feedback", str(session_id)), limit=1000)
        return [FeedbackTurn.model_validate(it.value) for it in items]

    async def delete_session(self, session_id: uuid.UUID) -> None:
        sid = str(session_id)
        await self._delete_prefix(("conversation", sid))
        await self._delete_prefix(("retrieval", sid))
        await self._delete_prefix(("llm", sid))
        await self._delete_prefix(("feedback", sid))


if __name__ == "__main__":
    sync = True
    if sync:
        memory = AgentMemory()
    else:
        memory = asyncio.run(AsyncAgentMemory.create())
    print(type(memory.store), type(memory.checkpointer))