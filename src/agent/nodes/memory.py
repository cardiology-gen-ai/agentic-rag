#!/usr/bin/env python3
"""
Memory node for the Cardiology Protocols Pipeline.
Enables the agent to remember previous messages (short memory) and/or user preferences (long memory).

Takes as input:
    - current state with messages

Gives as output:
    - updated states with conversation context
"""

import os
import sys
from typing import Dict, List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage

from sqlite.manager import StateManager
import configs

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
        conversation = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in messages])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Conversation:\n{conversation}")
        ])
        
        result = self.llm.invoke(prompt.format_messages(conversation=conversation))
        return result.content.strip()
    
    def update_state(self, state: Dict) -> Dict:
        """Update state with conversation summary."""
        messages = state.get("messages", [])
        if messages:
            state["conversation_summary"] = self.summarize_conversation(messages)
        
        if self.state_manager:
            self.state_manager.save_state(state)
        
        return state