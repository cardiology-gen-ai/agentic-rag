from typing import TypedDict, Annotated, List

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from agentic_rag.utils.search import SearchResult


class GraphState(TypedDict, total=False):
    """Shared state passed between :langgraph:`LangGraph <reference/graphs>` nodes."""
    question: str #: Original user question as received by the agent.
    contextual_question: str #: Question enriched with context (if needed) for retrieval/generation.
    summary: str
    transform_query_count: int #: How many times the question has been rewritten so far.
    response: str #: Latest assistant response (when available).
    language: str #: Language used in the conversation, optional.
    messages: Annotated[List[AnyMessage], add_messages] #: Rolling chat history used for context (:class:`list` of :class:`~langchain_core.messages.base.AnyMessage` ).
    documents: SearchResult #: Retrieved and filtered documents (:class:`list` of :langchain:`Document <core/documents/langchain_core.documents.base.Document.html>`), optional (when applicable).
    request_type: str #: Binary flag to denote whether the user is asking for an entire document.
    generation_count: int #: Number of generation attempts in the current turn.
