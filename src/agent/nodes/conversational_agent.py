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

current_dir = os.path.dirname(os.path.abspath(__file__))
state_manager_path = os.path.join(current_dir, '../../sqlite')
if os.path.exists(state_manager_path):
    sys.path.insert(0, state_manager_path)

from manager import StateManager


class ConversationalAgent:
    """Conversational agent for non-medical queries."""
    
    def __init__(self, llm_model: str = 'llama3.2:1b', state_manager: StateManager = None):
        self.llm = ChatOllama(model=llm_model, temperature=0.7, verbose=False)
        self.state_manager = state_manager
    
    def generate_response(self, query: str, conversation_summary: str = "") -> str:
        """Generate conversational response."""
        system_prompt = """You are a helpful assistant for a medical chatbot system. 
        Respond to general questions, greetings, and system inquiries in a friendly manner.
        If users ask about medical topics, guide them to ask specific medical questions.
        
        Keep responses concise and helpful."""
        
        context = f"Previous conversation: {conversation_summary}\n" if conversation_summary else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"{context}User query: {query}")
        ])
        
        result = self.llm.invoke(prompt.format_messages())
        return result.content.strip()
    
    def update_state(self, state: Dict) -> Dict:
        """Update state with conversational response."""
        query = state.get("query", "")
        conversation_summary = state.get("conversation_summary", "")
        
        response = self.generate_response(query, conversation_summary)
        state["response"] = response
        
        if self.state_manager:
            self.state_manager.save_state(state)
        
        return state