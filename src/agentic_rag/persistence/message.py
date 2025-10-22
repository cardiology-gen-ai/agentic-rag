import os
import uuid
from dataclasses import field
from datetime import datetime
from typing import Dict, Any, Optional, List
import asyncio

from psycopg import connect, Connection
from psycopg.rows import dict_row
from pydantic import BaseModel
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from cardiology_gen_ai.utils.logger import get_logger

from src.agentic_rag.managers.llm_manager import LLMManager
from src.agentic_rag.utils.chat import ChatResponse, ChatRequest


logger = get_logger("Agent memory management based on LangGraph")


class ConversationTurn(BaseModel):
    """A single conversation turn persisted in the memory store."""
    message_id: uuid.UUID | str #: :class:`uuid.UUID` or :class:`str` : Unique identifier of the message (a.k.a. *turn_id*).
    session_id: uuid.UUID | str #: :class:`uuid.UUID` or :class:`str` : Conversation (session) identifier.
    question: str #: :class:`str` : User question / input text.
    metadata: Dict = {} #: :class:`dict`, optional : Metadata returned by the agent; defaults to an empty dict.
    response: str | None = "" #: :class:`str`, optional : Assistant response content (may be empty or ``None`` on failure).
    created_at: datetime #: :class:`datetime.datetime` : Timestamp of when the question was created.
    error: str | None = None #: :class:`str`, optional : Error message if the turn failed.

    def model_post_init(self, __context: Any):
        """Normalize ids to :class:`uuid.UUID` after model initialization."""
        self.message_id = uuid.UUID(str(self.message_id))
        self.session_id = uuid.UUID(str(self.session_id))

    @classmethod
    def from_agent(cls, response: ChatResponse, request: ChatRequest) -> "ConversationTurn":
        """Build a :class:`~src.persistence.message.ConversationTurn` from agent I/O payloads.

        Parameters
        ----------
        response : :class:`~src.utils.chat.ChatResponse`
            Agent response envelope.
        request : :class:`~src.utils.chat.ChatRequest`
            Agent request envelope containing conversation context.

        Returns
        -------
        :class:`~src.persistence.message.ConversationTurn`
            New instance populated from the agent's request/response.
        """
        return ConversationTurn(
            message_id=request.conversation.question.id,
            session_id=request.conversation.id,
            question=request.conversation.question.content,
            metadata=response.metadata,
            response=response.content,
            created_at=request.conversation.question.datetime,
        )


class RetrievalTurn(BaseModel):
    """A retrieval step result associated with a conversation turn."""
    message_id: uuid.UUID | str #: :class:`uuid.UUID` or :class:`str` : Unique identifier for the message/turn.
    session_id: uuid.UUID | str #: :class:`uuid.UUID` or :class:`str` : Conversation (session) identifier.
    question: str #: :class:`str` : The query text for which sources were retrieved.
    sources: List[Dict[str, Any]] = field(default_factory=list) #: :class:`list` of :class:`dict` : Retrieved sources with metadata (schema defined by retriever).
    embedding_name: str = "" #: :class:`str` : Identifier of the embedding model/space used.

    def model_post_init(self, __context: Any):
        """Normalize ids to :class:`uuid.UUID` after model initialization."""
        self.message_id = uuid.UUID(str(self.message_id))
        self.session_id = uuid.UUID(str(self.session_id))

    @classmethod
    def from_agent(cls, response: ChatResponse, request: ChatRequest, embedding_name: str) -> "RetrievalTurn":
        """Build a :class:`~src.persistence.message.RetrievalTurn` from agent I/O plus embedding name.

        Parameters
        ----------
        response : :class:`~src.utils.chat.ChatResponse`
            Agent response envelope containing retrieval metadata under ``metadata['sources']``.
        request : :class:`~src.utils.chat.ChatRequest`
            Agent request envelope containing conversation context.
        embedding_name : :class:`str`
            Name of the embedding configuration used by the retriever.

        Returns
        -------
        :class:`~src.persistence.message.RetrievalTurn`
            New instance populated with question and retrieved sources.
        """
        return RetrievalTurn(
            message_id=request.conversation.question.id,
            session_id=request.conversation.id,
            question=request.conversation.question.content,
            sources=response.metadata["sources"],
            embedding_name=embedding_name,
        )


class LLMTurn(BaseModel):
    """Telemetry about the LLM call associated with a user-assistant interaction turn."""
    message_id: uuid.UUID | str #: :class:`uuid.UUID` or :class:`str` : Unique identifier for the message/turn.
    session_id: uuid.UUID | str #: :class:`uuid.UUID` or :class:`str` : Conversation (session) identifier.
    model_name: str #: :class:`str` : The underlying model reported by :class:`~src.managers.llm_manager.LLMManager`.
    generator_temperature: float = 0 #: :class:`float`, optional : Temperature used for the :attr:`~src.managers.llm_manager.LLMManager.generator` runnable.
    router_temperature: float = 0 #: :class:`float`, optional : Temperature used for the :attr:`~src.managers.llm_manager.LLMManager.router` runnable.
    grader_temperature: float = 0 #: :class:`float`, optional : Temperature used for the :attr:`~src.managers.llm_manager.LLMManager.grader` runnable.
    token_used: int | None = None #: :class:`int`, optional : Number of tokens consumed by the model for the turn.
    latency_ms: float | None = None #: :class:`float`, optional : Total wall-clock latency in milliseconds for model generation.

    def model_post_init(self, __context: Any):
        """Normalize ids to :class:`uuid.UUID` after model initialization."""
        self.message_id = uuid.UUID(str(self.message_id))
        self.session_id = uuid.UUID(str(self.session_id))

    @classmethod
    def from_agent(cls, response: ChatResponse, request: ChatRequest, llm_manager: LLMManager,
                   duration: float = None) -> "LLMTurn":
        """Build an :class:`~src.persistence.message.LLMTurn` from agent I/O and a :class:`~src.managers.llm_manager.LLMManager`.

        Parameters
        ----------
        response : :class:`~src.utils.chat.ChatResponse`
            Agent response envelope.
        request : :class:`~src.utils.chat.ChatRequest`
            Agent request envelope.
        llm_manager : :class:`~src.managers.llm_manager.LLMManager`
            Manager providing model name, temperatures, and token counting.
        duration : :class:`float`, optional
            Total generation latency in milliseconds.

        Returns
        -------
        LLMTurn
            New instance populated with model info and basic telemetry.
        """
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
    """Payload received from clients when submitting feedback."""
    feedback_value: int #: :class:`int` : Numeric feedback value (range defined by the application).
    feedback_message: str | None = None #: :class:`str`, optional : Optional free-text feedback message.


class FeedbackTurn(BaseModel):
    """A single feedback item associated with a given turn and user."""
    message_id: uuid.UUID | str #: :class:`uuid.UUID` or :class:`str` :  Message/turn identifier.
    session_id: uuid.UUID | str #: :class:`uuid.UUID` or :class:`str` : Conversation (session) identifier.
    user_id: uuid.UUID | str #: :class:`uuid.UUID` or :class:`str` : Stable identifier for the user who provided the feedback.
    feedback_value: int | None = 0 #: :class:`int`, optional : Numeric feedback value; defaults to ``0``.
    feedback_message: str | None = None #: :class:`str`, optional : Optional free-text feedback.
    created_at: datetime #: :class:`datetime.datetime` : Timestamp of feedback submission.

    def model_post_init(self, __context: Any):
        """Normalize ids to :class:`uuid.UUID` after model initialization."""
        self.message_id = uuid.UUID(str(self.message_id))
        self.session_id = uuid.UUID(str(self.session_id))
        self.user_id = uuid.UUID(str(self.user_id))


class AgentMemory:
    """Synchronous memory manager backed by :postgresql:`PostgreSQL <about>`.

    This class owns a single :psycopg:`psycopg.connect <module.html#psycopg2.connect>`-ed connection and configures both a :langgraph:`PostgresStore <store/?h=postgresstore#langgraph.store.postgres.PostgresStore>` (for key-value namespaces) and a
    :langgraph:`PostgresSaver <reference/checkpoints/?h=postgressa#langgraph.checkpoint.postgres.PostgresSaver>` (used as checkpointer).

    Parameters
    ----------
    db_connection_string : :class:`str`, optional
        If omitted, resolves from ``POSTGRES_ADMIN_DSN`` in the environment.

    Raises
    ------
    ValueError
        If no database connection string can be determined.
    """
    connection: Connection #: :psycopg:`Connection <connection.html#connection>` : Live :postgresql:`PostgreSQL <about>` connection.
    store: PostgresStore #: :langgraph:`PostgresStore <store/?h=postgresstore#langgraph.store.postgres.PostgresStore>` : Key-value store abstraction layered on :postgresql:`PostgreSQL <about>`.
    checkpointer: PostgresSaver #: :langgraph:`PostgresSaver <reference/checkpoints/?h=postgressa#langgraph.checkpoint.postgres.PostgresSaver>`  : Checkpointer for LangGraph workflows.
    def __init__(self, db_connection_string: Optional[str] = None):
        db_connection_string = db_connection_string or os.getenv("POSTGRES_ADMIN_DSN")
        if not db_connection_string:
            raise ValueError("No database connection string provided")
        self.connection = connect(conninfo=db_connection_string, autocommit=True, row_factory=dict_row)
        self.store = PostgresStore(self.connection)
        self.store.setup()
        self.checkpointer = PostgresSaver(self.connection)
        self.checkpointer.setup()

    def save_conversation_turn(self, conversation_turn: ConversationTurn) -> None:
        """Persist a :class:`~src.persistence.message.ConversationTurn` under the ``conversation`` namespace.

        Parameters
        ----------
        conversation_turn : :class:`~src.persistence.message.ConversationTurn`
            The turn to persist.
        """
        try:
            self.store.put(("conversation", str(conversation_turn.session_id)),
                           str(conversation_turn.message_id),
                           conversation_turn.model_dump(exclude_none=False, mode="json"))
        except Exception as e:
            logger.info(f"Failed to save conversation: {e}")

    @staticmethod
    def item_to_conversation_turn(item: Any) -> ConversationTurn:
        """Convert a raw store item into a :class:`~src.persistence.message.ConversationTurn`.

        Parameters
        ----------
        item : :class:`object`
            Store item with attributes ``value``, ``key``, ``namespace``, and ``created_at`` (shape defined by the underlying store).

        Returns
        -------
        :class:`~src.persistence.message.ConversationTurn`
            A validated :class:`~src.persistence.message.ConversationTurn` instance.
        """
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

    def get_history(self, session_id: uuid.UUID, limit: int, reverse: bool = True) -> List[ConversationTurn]:
        """Fetch a slice of conversation history for a session.

        Parameters
        ----------
        session_id : :class:`uuid.UUID`
            The conversation/session identifier.
        limit : :class:`int`
            Maximum number of turns to return (after sorting).
        reverse : :class:`bool`, optional
            If ``True`` (default) returns newest-first order.

        Returns
        -------
        list of :class:`~src.persistence.message.ConversationTurn`
            The most recent ``limit`` conversation turns.
        """
        conversation_items = self.store.search(("conversation", str(session_id)), limit=1000)
        if len(conversation_items) == 0 or conversation_items is None:
            return []
        conversation_items = [self.item_to_conversation_turn(item) for item in conversation_items]
        conversation_items_sorted = sorted(conversation_items, key=lambda item: item.created_at, reverse=reverse)
        return conversation_items_sorted[:limit]

    def save_retrieval_turn(self, retrieval_turn: RetrievalTurn) -> None:
        """Persist a :class:`~src.persistence.message.RetrievalTurn` under the ``retrieval`` namespace."""
        try:
            self.store.put(("retrieval", str(retrieval_turn.session_id), str(retrieval_turn.message_id)),
                           "results",
                           retrieval_turn.model_dump(exclude_none=False, mode="json"))
        except Exception as e:
            logger.info(f"Failed to save retriever results: {e}")

    def save_llm_turn(self, llm_turn: LLMTurn) -> None:
        """Persist an :class:`~src.persistence.message.LLMTurn` under the ``llm`` namespace."""
        try:
            self.store.put(("llm", str(llm_turn.session_id), str(llm_turn.message_id)),
                           "results",
                           llm_turn.model_dump(exclude_none=False, mode="json"))
        except Exception as e:
            logger.info(f"Failed to save LLM information: {e}")

    def save_feedback(self, feedback: FeedbackTurn):
        """Persist a :class:`~src.persistence.message.FeedbackTurn` under the ``feedback`` namespace."""
        try:
            self.store.put(("feedback", str(feedback.session_id)),
                            str(feedback.message_id),
                            feedback.model_dump(exclude_none=False, mode="json"))
        except Exception as e:
            logger.info(f"Failed to save feedback: {e}")

    def get_session_feedback(self, session_id: uuid.UUID) -> List[FeedbackTurn]:
        """Return all feedback items for a given session.

        Parameters
        ----------
        session_id : :class:`uuid.UUID`
            The conversation/session identifier.

        Returns
        -------
        list of :class:`~src.persistence.message.FeedbackTurn`
            All feedback entries stored for the session.
        """
        session_items = self.store.search(("feedback", str(session_id)), limit=1000)
        session_feedback: List[FeedbackTurn] = []
        for it in session_items:
            session_feedback.append(FeedbackTurn.model_validate(it.value))
        return session_feedback

    def _delete_prefix(self, ns_prefix: tuple, batch_size: int = 500) -> None:
        """Delete a namespace subtree in batches.

        Parameters
        ----------
        ns_prefix : :class:`tuple`
            Namespace tuple prefix (e.g., ``("conversation", session_id)``).
        batch_size : :class:`int`, optional
            Maximum number of items to delete per batch (default ``500``).
        """
        while True:
            batch = self.store.search(ns_prefix, limit=batch_size)
            if not batch:
                break
            for it in batch:
                self.store.delete(it.namespace, it.key)

    def delete_session(self, session_id: uuid.UUID) -> None:
        """Remove all stored data for a session across namespaces.

        Parameters
        ----------
        session_id : :class:`uuid.UUID`
            The conversation/session identifier to purge.
        """
        sid = str(session_id)
        self._delete_prefix(("conversation", sid))
        self._delete_prefix(("retrieval", sid))
        self._delete_prefix(("llm", sid))
        self._delete_prefix(("feedback", sid))


class AsyncAgentMemory:
    """Asynchronous memory manager backed by :postgresql:`PostgreSQL <about>`.

    This class wraps :langgraph:`AsyncPostgresStore <store/?h=asyncpostgresstore#langgraph.store.postgres.AsyncPostgresStore>` and :langgraph:`AsyncPostgresSaver <checkpoints/?h=asyncpostgressaver#langgraph.checkpoint.postgres.aio.AsyncPostgresSaver>`
    instances and exposes async helpers mirroring :class:`~src.persistence.message.AgentMemory`.
    """
    store: AsyncPostgresStore #: :langgraph:`AsyncPostgresStore <store/?h=asyncpostgresstore#langgraph.store.postgres.AsyncPostgresStore>` : Initialized asynchronous store instance.
    checkpointer: AsyncPostgresSaver #: :langgraph:`AsyncPostgresSaver <checkpoints/?h=asyncpostgressaver#langgraph.checkpoint.postgres.aio.AsyncPostgresSaver>` : Initialized asynchronous checkpointer instance.
    def __init__(self, store: AsyncPostgresStore, checkpointer: AsyncPostgresSaver):
        self.store = store
        self.checkpointer = checkpointer

    @classmethod
    async def create(cls, db_connection_string: Optional[str] = None) -> "AsyncAgentMemory":
        """Factory that creates and sets up an :class:`~src.persistence.message.AsyncAgentMemory`.

        Parameters
        ----------
        db_connection_string : :class:`str`, optional
            If omitted, resolves from ``POSTGRES_ADMIN_DSN`` in the environment.

        Returns
        -------
        :class:`~src.persistence.message.AsyncAgentMemory`
            A fully initialized asynchronous memory manager.

        Raises
        ------
        ValueError
            If no database connection string can be determined.
        """
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
        """Delete a namespace subtree in batches (async)."""
        while True:
            batch = await self.store.asearch(ns_prefix, limit=batch_size)
            if not batch:
                break
            for it in batch:
                await self.store.adelete(it.namespace, it.key)

    async def save_conversation_turn(self, conversation_turn: ConversationTurn) -> None:
        """Persist a :class:`~src.persistence.message.ConversationTurn` under the ``conversation`` namespace (asynchronous)."""
        await self.store.aput(("conversation", str(conversation_turn.session_id)),
                              str(conversation_turn.message_id),
                              conversation_turn.model_dump(exclude_none=False, mode="json"))

    async def save_retrieval_turn(self, retrieval_turn: RetrievalTurn) -> None:
        """Persist a :class:`~src.persistence.message.RetrievalTurn` under the ``retrieval`` namespace (asynchronous)."""
        await self.store.aput(("retrieval", str(retrieval_turn.session_id), str(retrieval_turn.message_id)),
                              "results", retrieval_turn.model_dump(exclude_none=False, mode="json"))

    async def save_llm_turn(self, llm_turn: LLMTurn) -> None:
        """Persist an :class:`~src.persistence.message.LLMTurn` under the ``llm`` namespace (asynchronous)."""
        await self.store.aput(("llm", str(llm_turn.session_id), str(llm_turn.message_id)),
                              "results", llm_turn.model_dump(exclude_none=False, mode="json"))

    async def save_feedback(self, feedback: FeedbackTurn) -> None:
        """Persist a :class:`~src.persistence.message.FeedbackTurn` under the ``feedback`` namespace (asynchronous)."""
        await self.store.aput( ("feedback", str(feedback.session_id)),
                               str(feedback.message_id),
                               feedback.model_dump(exclude_none=False, mode="json"))

    async def get_session_feedback(self, session_id: uuid.UUID) -> List[FeedbackTurn]:
        """Return all feedback items for a given session (async)."""
        items = await self.store.asearch(("feedback", str(session_id)), limit=1000)
        return [FeedbackTurn.model_validate(it.value) for it in items]

    async def delete_session(self, session_id: uuid.UUID) -> None:
        """Remove all stored data for a session across namespaces (async)."""
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