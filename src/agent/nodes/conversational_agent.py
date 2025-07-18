#!/usr/bin/env python3
"""
Conversational Agent for the Cardiology Protocols Pipeline.
Handles non-medical queries with friendly, helpful responses.

This agent:
1. Responds to greetings, farewells, and general conversation
2. Provides system information and capabilities
3. Guides users towards medical query functionality
4. Maintains conversation context and personality
5. Handles edge cases and fallback scenarios
"""

import os
import sys
from typing import Dict
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from sqlite.manager import StateManager
from agent import configs 
from agent.state import State

class ConversationalAgent:
    """Conversational agent for non-medical queries."""
    
    def __init__(self, llm_model: str = configs.LLM_MODEL, state_manager: StateManager = None):
        self.llm = ChatOllama(model=llm_model, temperature=configs.CONVERSATIONAL_LLM_TEMPERATURE, verbose=False)
        self.state_manager = state_manager
    
    def generate_response(self, state) -> str:
        """Generate conversational response."""
        system_prompt = """You are a helpful assistant for a medical chatbot system. 
        Respond to general questions, greetings, and system inquiries in a friendly manner.
        If users ask about medical topics, guide them to ask specific medical questions.
        
        Keep responses concise and helpful."""
        
        context = f"Previous conversation: {state.get('conversation_summary')}\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"{context}User query: {state.get('message')}")
        ])
        
        result = self.llm.invoke(prompt.format_messages())
        state["previous_messages"].append(state.get("message"))
        return result.content.strip()
    
    def update_state(self, state: State) -> State:
        """Update state with conversational response."""
        response = self.generate_response(state)
        state["response"] = response
        
        return state
