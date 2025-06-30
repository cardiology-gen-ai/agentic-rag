#!/usr/bin/env python3
"""
State class for the Cardiology Protocols Pipeline.
"""

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class State(TypedDict):
    # Core conversation
    messages: List[BaseMessage]  # More specific than just List
    
    # Routing information
    query_type: Optional[str]  # Might not be set initially
    
    # Response generation
    response: Optional[str]  # Not always present

    # Memory
    context: Optional[Dict[str, Any]]

    # Conversation summary
    conversation_summary: Optional[str]
    
    # RAG-specific
    documents: Optional[List[Document]]  # Retrieved documents
    
    # Metadata and tracking
    metadata: Optional[Dict[str, Any]]  # More specific value type
    retrieval_attempts: Optional[int]  # For self-RAG tracking
    generation_attempts: Optional[int]  # For self-RAG tracking

    # State tracking
    current_state: Optional[str]

    # Next action
    next_action: Optional[str]
