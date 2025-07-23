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

import re
from typing import Dict, List, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import configs
from state import State

class ConversationalAgent:
    """Conversational agent for non-medical queries."""
    
    def __init__(self, llm_model: str = configs.LLM_MODEL, state_manager = None):
        self.llm = ChatOllama(model=llm_model, temperature=configs.CONVERSATIONAL_LLM_TEMPERATURE, verbose=False)
        self.state_manager = state_manager
    
    def generate(self, query: str, conversation_summary: str) -> str:
        """Generate conversational response."""
        system_prompt = """You are a helpful assistant for a medical chatbot system. 
        Respond to general questions, greetings, and system inquiries in a friendly manner.
        If users ask about medical topics, guide them to ask specific medical questions.
        
        Keep responses concise and helpful."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"Conversation summary: {conversation_summary or 'No previous conversation'}\n\nUser query: {query}")
        ])
        
        result = self.llm.invoke(prompt.format_messages())
        return result.content.strip()
    
    def update_state(self, state: State) -> State:
        """Update state with conversational response."""
        query = state.get('message')
        conversation_summary = state.get('conversation_summary')
        response = self.generate(query, conversation_summary)
        
        # Append the user message to previous messages
        if state.get('previous_messages') is None:
            state['previous_messages'] = []
        state['previous_messages'].append(HumanMessage(content=query))
        
        # Update the message with the response
        state['message'] = response
        
        # Add the AI response to previous messages
        state['previous_messages'].append(AIMessage(content=response))

        return state
