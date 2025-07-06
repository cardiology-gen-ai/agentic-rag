#!/usr/bin/env python3
"""
Interactive Chat Orchestrator for the Cardiology Protocols Pipeline.
Provides a command-line interface to interact with the RAG system.
"""

import os
import sys
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
nodes_path = os.path.join(current_dir, 'nodes')
utils_path = os.path.join(current_dir, 'utils')
sqlite_path = os.path.join(current_dir, '../sqlite')
vectorstore_path = os.path.join(current_dir, '../../../data-etl/src')

sys.path.extend([nodes_path, utils_path, sqlite_path, vectorstore_path])

try:
    from state import State
    from router import Router
    from memory import Memory
    from conversational_agent import ConversationalAgent
    from self_rag import SelfRAG
    from manager import StateManager
    from vectorstore import load_vectorstore
except ImportError as e:
    sys.exit(1)


class Orchestrator:
    """Main orchestrator for the cardiology chat system."""
    
    def __init__(self, llm_model: str = "llama3.2:1b"):
        self.llm_model = llm_model
        self.state_manager = StateManager()
        
        # Initialize vectorstore
        try:
            self.vectorstore = load_vectorstore(
                collection_name="cardio_protocols",
                vectorstore_type="qdrant",
                qdrant_url="http://localhost:6333"
            )
            print("✓ Vectorstore loaded successfully")
        except Exception as e:
            self.vectorstore = None
        
        # Initialize agents
        self.router = Router(llm_model, self.state_manager)
        self.memory = Memory(llm_model, self.state_manager)
        self.conversational_agent = ConversationalAgent(llm_model, self.state_manager)
        self.self_rag = SelfRAG(self.vectorstore, llm_model, self.state_manager)
        
        # Build graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("router", self.router.update_state)
        workflow.add_node("memory", self.memory.update_state)
        workflow.add_node("conversational", self.conversational_agent.update_state)
        workflow.add_node("self_rag", self.self_rag.update_state)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self.router.route_query,
            {
                "document_based": "self_rag",
                "conversational": "conversational"
            }
        )
        
        # Add edges to memory and end
        workflow.add_edge("conversational", "memory")
        workflow.add_edge("self_rag", "memory")
        workflow.add_edge("memory", END)
        
        # Compile with checkpointer
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    def process_message(self, user_input: str, thread_id: str = None) -> str:
        """Process a user message and return the response."""
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        # Create initial state
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get current state or create new one
        try:
            current_state = self.graph.get_state(config).values
            messages = current_state.get("messages", [])
        except:
            messages = []
        
        # Add human message
        messages.append(HumanMessage(content=user_input))
        
        initial_state = {
            "messages": messages,
            "query": user_input,
            "thread_id": thread_id,
            "user_id": "default_user"
        }
        
        # Run the graph
        try:
            result = self.graph.invoke(initial_state, config)
            response = result.get("response", "I'm sorry, I couldn't process your request.")
            
            # Add AI message to conversation
            messages.append(AIMessage(content=response))
            
            return response
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return "I encountered an error processing your request. Please try again."
    
    def start_chat(self):
        """Start interactive chat session."""
        print("\n" + "="*60)
        print("🏥 CARDIOLOGY PROTOCOLS ASSISTANT")
        print("="*60)
        print("Welcome! I can help you with:")
        print("• Cardiology guidelines and protocols")
        print("• Treatment recommendations")
        print("• Clinical questions")
        print("• General conversation")
        print("\nType 'quit', 'exit', or 'bye' to end the conversation")
        print("-"*60 + "\n")
        
        thread_id = str(uuid.uuid4())
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nAssistant: Goodbye! Take care!")
                    break
                
                # Process message
                print("\nAssistant: ", end="", flush=True)
                response = self.process_message(user_input, thread_id)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")


def main():
    """Main function to run the chat interface."""
    try:
        orchestrator = Orchestrator()
        orchestrator.start_chat()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Failed to start orchestrator: {e}")
        print("Make sure all dependencies are installed and Qdrant is running")


if __name__ == "__main__":
    main()