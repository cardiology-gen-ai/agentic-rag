#!/usr/bin/env python3
"""
Final agent for the Cardiology Protocols Pipeline with complete cross-session persistence.
"""

import os
import sys
import uuid
from typing import List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from agent.utils.state import State
from agent.nodes.router import Router
from agent.nodes.memory import Memory
from agent.nodes.conversational_agent import ConversationalAgent
from agent.nodes.rag import RAG
from sqlite.manager import StateManager
import configs

current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.join(current_dir, '../../../data-etl/src')
sys.path.append(vectorstore_path)
from vectorstore import load_vectorstore

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

langfuse = Langfuse(
    public_key=configs.LANGFUSE_PUBLIC_KEY,
    secret_key=configs.LANGFUSE_SECRET_KEY,
    host=configs.LANGFUSE_HOST
)

langfuse_handler = CallbackHandler()

class Agent:
    """Enhanced agent with complete cross-session persistence."""
    
    def __init__(self, state_manager, llm_model: str = configs.LLM_MODEL):
        self.llm_model = llm_model
        self.state_manager = state_manager
        
        # Initialize vectorstore
        try:
            self.vectorstore = load_vectorstore(
                collection_name=configs.QDRANT_COLLECTION_NAME,
                vectorstore_type=configs.VECTORSTORE_TYPE,
                qdrant_url=configs.QDRANT_URL
            )
            print("✓ Vectorstore loaded successfully")
        except Exception as e:
            print(f"⚠️  Vectorstore not available: {e}")
            self.vectorstore = None
        
        # Initialize agents
        self.router = Router(llm_model, self.state_manager)
        self.memory = Memory(llm_model, self.state_manager)
        self.conversational_agent = ConversationalAgent(llm_model, self.state_manager)
        self.rag = RAG(self.vectorstore, llm_model, self.state_manager)
        
        # Build graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("router", self.router.update_state)
        workflow.add_node("memory", self.memory.update_state)
        workflow.add_node("conversational", self.conversational_agent.update_state)
        workflow.add_node("rag", self.rag.update_state)
        
        # Set entry point
        workflow.add_edge(START, "memory")
        workflow.add_edge("memory", "router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self.router.route_query,
            {
                "document_based": "rag",
                "conversational": "conversational"
            }
        )
        
        workflow.add_edge("rag", END)
        workflow.add_edge("conversational", END)
        
        # Compile with checkpointer
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    def process_message(self, state : State) -> str:
        """
        Process a user message with proper cross-session persistence.
        Assumes a state has already been initialized with a thread_id.
        """
        config = {
            "configurable": {"thread_id": state["thread_id"]},
            "callbacks": [langfuse_handler]
        }
        
        # Run the graph with LangFuse callback
        try:
            result = self.graph.invoke(state, config)
            response = result.get("response", "I'm sorry, I couldn't process your request.")
            
            return response
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return "I encountered an error processing your request. Please try again."
    
    