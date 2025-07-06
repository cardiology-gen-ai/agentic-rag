#!/usr/bin/env python3
"""
State class for the Cardiology Protocols Pipeline.
"""

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class State(TypedDict):
    # Core conversation
    messages: List[BaseMessage]  # history of messages
    
    # Routing information
    query_type: Optional[str] 

    # Query
    query: Optional[str]
    
    # Response generation
    response: Optional[str]  

    # Context
    context: Optional[Dict[str, Any]]

    # Conversation summary
    conversation_summary: Optional[str]
    
    # RAG-specific
    documents: Optional[List[Document]]  # Retrieved documents
    
    # Metadata and tracking
    metadata: Optional[Dict[str, Any]]  # More specific value type
    retrieval_attempts: Optional[int]  # For self-RAG tracking
    generation_attempts: Optional[int]  # For self-RAG tracking

    # Next action
    next_action: Optional[str]

    # Thread ID
    thread_id: Optional[str]

    # User ID
    user_id: Optional[str]
