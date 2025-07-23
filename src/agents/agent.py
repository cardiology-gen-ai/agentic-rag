#!/usr/bin/env python3

import os
import sys
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

# Import pipeline components
from state import State
from nodes.memory import Memory
from nodes.router import Router
from nodes.conversational_agent import ConversationalAgent
from nodes.rag import RAG
import configs

current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_dir = os.path.join(current_dir, '../../data-etl/src')
if os.path.exists(vectorstore_dir):
    sys.path.insert(0, vectorstore_dir)

# Try to import vectorstore and self-RAG components
try:
    from vectorstore import load_vectorstore
    VECTORSTORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Vectorstore not available: {e}")
    VECTORSTORE_AVAILABLE = False


class Agent:
    """
    Main pipeline that manages the entire cardiology RAG workflow.
    """
    
    def __init__(self, thread_id: str = None):
        """Initialize the orchestrator with empty components."""
        self.thread_id = thread_id
        self.router = None
        self.memory = None
        self.rag = None
        self.conversational_agent = None
        self.vectorstore = None
        self.graph = None
        
        self.langfuse = Langfuse(
                public_key = configs.LANGFUSE_PUBLIC_KEY,
                secret_key = configs.LANGFUSE_SECRET_KEY,
                host = configs.LANGFUSE_HOST
                )
        self.langfuse_handler = CallbackHandler()
        
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Initialize basic components (always available)
            self.router = Router()
            self.memory = Memory(thread_id=self.thread_id)
            self.conversational_agent = ConversationalAgent()
            
            # Initialize vectorstore and Self-RAG if available
            if VECTORSTORE_AVAILABLE:
                try:
                    self.vectorstore = load_vectorstore(
                        collection_name="cardio_protocols",
                        vectorstore_type="qdrant"
                    )
                    
                    self.rag = RAG(self.vectorstore)
                    
                except Exception as e:
                    self.rag = None
                    self.vectorstore = None
            else:
                self.rag = None
                self.vectorstore = None
            
        except Exception as e:
            raise

    def _build_graph(self):
        """
        Build the LangGraph workflow for the pipeline.
        
        Returns:
            Compiled LangGraph workflow
        """
        
        # Create the workflow
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("memory", self.memory.update_state)
        workflow.add_node("router", self.router.update_state)
        workflow.add_node("conversational_agent", self.conversational_agent.update_state)
        
        # Conditionally add rag node if available
        workflow.add_node("rag", self.rag.update_state)
        
        # Add edges
        workflow.add_edge(START, "memory")
        workflow.add_edge("memory", "router")
        
        # Conditional routing based on query type
        routing_map = {
            "conversational": "conversational_agent",
            "document_based": "rag" if self.rag else "conversational_agent"
        }
        
        workflow.add_conditional_edges(
            "router",
            self.router.route_query,
            routing_map
        )
        
        # Add final edges
        workflow.add_edge("conversational_agent", END)
        if self.rag:
            workflow.add_edge("rag", END)
        
        # Compile the graph
        self.graph = workflow.compile()

    def process_query(self, query: str, user_id: str, thread_id: str) -> str:
        try:
            # Search for already existing conversations in the database (should output a state)
            # Otherwise create a new state
            initial_state = State(
                user_id=user_id,
                thread_id=thread_id,
                message=query,
                previous_messages=[],
                conversation_summary=None,
                is_query=True,
                query_type=None,
                rewritten_query=None,
                documents=None,
                feedback=None,
                metadata={},
                retrieval_attempts=0,
                generation_attempts=0
            )
            
            # Try to restore conversation from database if memory is available
            if self.memory and self.memory.db_manager:
                saved_state = self.memory.load_from_database()
                if saved_state:
                    # Restore previous messages and conversation summary
                    messages = []
                    for msg_data in saved_state.get("messages", []):
                        if msg_data["type"] == "HumanMessage":
                            messages.append(HumanMessage(content=msg_data["content"]))
                        elif msg_data["type"] == "AIMessage":
                            messages.append(AIMessage(content=msg_data["content"]))
                    
                    initial_state["previous_messages"] = messages
                    initial_state["conversation_summary"] = saved_state.get("conversation_summary")
                    initial_state["metadata"] = saved_state.get("metadata", {})

            final_state = self.graph.invoke(initial_state, config={"callbacks": [self.langfuse_handler]})
            
            # Extract and display results
            response = final_state.get("message", "No response generated")
            return response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def cleanup(self):
        """Clean up resources and close database connections"""
        if self.memory:
            self.memory.close()

