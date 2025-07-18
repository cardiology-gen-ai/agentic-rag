#!/usr/bin/env python3

from typing import TypedDict, Optional, List, Any, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

class State(TypedDict):
    """
    This class represents the short-term memory of the agentic-rag.
    """
    user_id: str
    conversation_id: str
    
    message: str
    previous_messages: Optional[List[HumanMessage | AIMessage]]
    conversation_summary: Optional[str]

    is_query: bool
    query_type: Optional[str]
    rewritten_query: Optional[str]

    user_context: Optional[str]

    documents: Optional[List[Document]] # Retrieved documents

    feedback: Optional[str]

    # Response field
    response: Optional[str]
    
    # Metadata and tracking
    metadata: Optional[Dict[str, Any]]  # More specific value type
    retrieval_attempts: Optional[int]  # For self-RAG tracking
    generation_attempts: Optional[int]  # For self-RAG tracking
