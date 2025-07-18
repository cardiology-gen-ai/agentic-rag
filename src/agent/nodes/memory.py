#!/usr/bin/env python3
"""
Memory node for the Cardiology Protocols Pipeline.
It summarizes previous messages for memort context.
"""

import os
import sys
from typing import Dict, List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage

from sqlite.manager import StateManager
from agent import configs
from agent.state import State

class Memory:
    """Memory node for conversation summarization."""
    
    def __init__(self, llm_model: str, state_manager: StateManager = None):
        self.llm = ChatOllama(model=configs.LLM_MODEL, temperature=configs.MEMORY_LLM_TEMPERATURE)
        self.state_manager = state_manager
    
    def summarize_conversation(self, messages: List[BaseMessage]) -> str:
        """Summarize the conversation history."""
        if not messages:
            return ""
        
        system_prompt = """Summarize this medical conversation in 2-3 sentences, focusing on:
        - Main medical topics discussed
        - Key questions asked
        - Important information provided
        
        Keep it concise and medically relevant."""
        
        # Convert messages to string format
        conversation_parts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # It's a message object
                conversation_parts.append(f"{type(msg).__name__}: {msg.content}")
            else:
                # It's already a string
                conversation_parts.append(f"Message: {msg}")
        conversation = "\n".join(conversation_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Conversation:\n{conversation}")
        ])
        
        result = self.llm.invoke(prompt.format_messages(conversation=conversation))
        return result.content.strip()
    
    def update_state(self, state: State) -> State:
        """Update state with conversation summary."""
        message_history = state.get("previous_messages", [])
        if message_history:
            state["conversation_summary"] = self.summarize_conversation(message_history)

        return state
