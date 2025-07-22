#!/usr/bin/env python3
"""
Router node for the RAG pipeline.
Enhanced classification for medical queries vs conversational queries.
"""

import re
from typing import Dict, List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import configs
from state import State

class Router:
    """Router node for medical query classification."""
    
    def __init__(self, 
                 llm_model: str = configs.LLM_MODEL,
                 temperature: float = configs.ROUTER_LLM_TEMPERATURE,
                 state_manager = None):
        self.llm = ChatOllama(model=llm_model, temperature=temperature, verbose=False)
        self.state_manager = state_manager
    
    def classify_query(self, query: str) -> str:
        """Classify query as document_based or conversational."""
        if not query or not query.strip():
            return "conversational"
        
        system_prompt = """You are a query classifier. Your ONLY job is to classify queries into categories.

CRITICAL: You must respond with EXACTLY ONE WORD - nothing else!

Categories:
- document_based: Medical questions, conditions, treatments, clinical guidelines
- conversational: Greetings, farewells, non-medical topics

Examples:
User: "what is myocarditis?" → document_based
User: "hello" → conversational
User: "goodbye" → conversational
User: "heart failure treatment" → document_based

Respond with ONLY: document_based OR conversational"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        result = self.llm.invoke(prompt.format_messages(query=query))
        response = result.content.strip().lower()
        
        # Clean the response to extract only the classification
        if "document_based" in response:
            return "document_based"
        elif "conversational" in response:
            return "conversational"
        else:
            # Default to document_based for safety
            return "document_based" 
    
    def update_state(self, state: State) -> State:
        if state.get("is_query"):
            query = state.get("message")
            query_type = self.classify_query(query)
            if configs.DEBUG:
                print(f"\nQuery classified as: {query_type}")
            state["query_type"] = query_type
        
        return state

    def route_query(self, state: State) -> str:
        """Conditional edge function for LangGraph routing."""
        query_type = state.get("query_type", "document_based")
        if configs.DEBUG:
            print(f"\nRouting to: {query_type}")
        return query_type
