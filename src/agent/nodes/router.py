#!/usr/bin/env python3
"""
Router node for the RAG pipeline.
Enhanced classification for medical queries vs conversational queries.
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


class Router:
    """Router node for medical query classification."""
    
    def __init__(self, llm_model: str, state_manager: StateManager = None):
        self.llm = ChatOllama(model=llm_model, temperature=0.0, verbose=False)
        self.state_manager = state_manager
    
    def classify_query(self, query: str) -> str:
        """Classify query as document_based or conversational."""
        if not query or not query.strip():
            return "conversational"
        
        system_prompt = """You are a router for a medical chatbot. Classify the message into one of these categories:
        - document_based: Questions about specific medical conditions, treatments, protocols, or clinical guidelines. So, if you encounter specific medical keyword it is highly probable that it is document_based.
        - conversational: General greetings, farewells, system questions, or non-medical topics
        
        Respond with only: document_based OR conversational"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        result = self.llm.invoke(prompt.format_messages(query=query))
        response = result.content.strip().lower()
        
        return "document_based" 
    
    def update_state(self, state: Dict) -> Dict:
        """LangGraph node function for query routing."""
        query = state.get("query", "")
        query_type = self.classify_query(query)
        state["query_type"] = query_type
        
        if self.state_manager:
            state_id = self.state_manager.save_state(state)
            state["state_id"] = state_id
        
        return state
    
    def route_query(self, state: Dict) -> str:
        query_type = state.get("query_type", "conversational")
        return query_type
