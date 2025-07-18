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
agent_dir = os.path.join('current_dir', '../')
sys.path.insert(0, agent_dir)

from sqlite.manager import StateManager
from agent import configs
from agent.state import State

class Router:
    """Router node for medical query classification."""
    
    def __init__(self, llm_model: str, state_manager: StateManager = None):
        self.llm = ChatOllama(model=configs.LLM_MODEL, temperature=configs.ROUTER_LLM_TEMPERATURE, verbose=False)
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
        
        return response # this could lead to bugs since it is proned to classify always to conversational 
    
    def route_query(self, state: State) -> str:
        """Route the query to the appropriate node."""
        if state.get("is_query"):
            query = state.get("message")
            query_type = self.classify_query(query)
            state["query_type"] = query_type
            return query_type
        return "conversational"
    
    def update_state(self, state: State) -> State:
        if state.get("is_query"):
            query = state.get("message")
            query_type = self.classify_query(query)
            state["query_type"] = query_type
        
        return state
   
