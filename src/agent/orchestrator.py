#!/usr/bin/env python3
"""
Final Orchestrator for the Cardiology Protocols Pipeline with complete cross-session persistence.
"""

import os
import sys
import uuid
import sqlite3
from typing import List, Optional, Dict
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
    print(f"Import error: {e}")
    sys.exit(1)


class Orchestrator:
    """Enhanced orchestrator with complete cross-session persistence."""
    
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
            print(f"⚠️  Vectorstore not available: {e}")
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
    
    def load_conversation(self, thread_id: str) -> List:
        """Load existing conversation from SQLite."""
        return self.state_manager.load_thread_messages(thread_id)
    
    def list_conversations(self):
        """List all available conversations."""
        threads = self.state_manager.list_threads()
        
        if not threads:
            print("No previous conversations found.")
            return []
        
        print("\nPrevious Conversations:")
        print("-" * 80)
        print(f"{'#':<3} {'Thread ID':<12} {'First Message':<19} {'Last Message':<19} {'Msgs':<5}")
        print("-" * 80)
        
        for i, (thread_id, first_msg, last_msg, msg_count) in enumerate(threads, 1):
            print(f"{i:<3} {thread_id[:10]+'...':<12} {first_msg[:19]:<19} {last_msg[:19]:<19} {msg_count:<5}")
        
        return threads
    
    def process_message(self, user_input: str, thread_id: str = None) -> str:
        """Process a user message with proper cross-session persistence."""
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Try to get messages from LangGraph memory first
        try:
            current_state = self.graph.get_state(config).values
            messages = current_state.get("messages", [])
        except:
            messages = []
        
        # CRITICAL: If no messages in LangGraph memory, load from SQLite
        if not messages and self.state_manager.thread_exists(thread_id):
            messages = self.load_conversation(thread_id)
            print(f"Restored {len(messages)} messages from previous session")
        
        # Add new human message
        messages.append(HumanMessage(content=user_input))
        
        # Get conversation summary if available
        conversation_summary = self.state_manager.get_thread_summary(thread_id)
        
        initial_state = {
            "messages": messages,
            "query": user_input,
            "thread_id": thread_id,
            "user_id": "default_user",
            "conversation_summary": conversation_summary
        }
        
        # Run the graph
        try:
            result = self.graph.invoke(initial_state, config)
            response = result.get("response", "I'm sorry, I couldn't process your request.")
            
            return response
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return "I encountered an error processing your request. Please try again."
    
    def start_chat(self, resume_thread: str = None):
        """Start interactive chat session with optional thread resumption."""
        print("\n" + "="*60)
        print("CARDIOLOGY PROTOCOLS ASSISTANT")
        print("="*60)
        
        # Handle conversation resumption
        if resume_thread:
            if self.state_manager.thread_exists(resume_thread):
                print(f"Resuming conversation: {resume_thread[:8]}...")
                thread_id = resume_thread
                # Load and display last few messages for context
                messages = self.load_conversation(thread_id)
                if messages:
                    print(f"Found {len(messages)} previous messages")
                    print("Recent context:")
                    for msg in messages[-4:]:  # Show last 4 messages
                        role = "You" if isinstance(msg, HumanMessage) else "Assistant"
                        content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                        print(f"  {role}: {content}")
                    print()
            else:
                print(f"Thread {resume_thread} not found. Starting new conversation.")
                thread_id = str(uuid.uuid4())
        else:
            # Ask if user wants to resume
            threads = self.state_manager.list_threads()
            if threads:
                print("Previous conversations available.")
                choice = input("Resume previous conversation? (y/n): ").strip().lower()
                if choice == 'y':
                    self.list_conversations()
                    try:
                        selection = input("Enter conversation number (or press Enter for new): ").strip()
                        if selection.isdigit():
                            idx = int(selection) - 1
                            if 0 <= idx < len(threads):
                                thread_id = threads[idx][0]
                                print(f"Resuming conversation: {thread_id[:8]}...")
                                # Show context
                                messages = self.load_conversation(thread_id)
                                if messages:
                                    print("Recent context:")
                                    for msg in messages[-4:]:
                                        role = "You" if isinstance(msg, HumanMessage) else "Assistant"
                                        content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                                        print(f"  {role}: {content}")
                                    print()
                            else:
                                thread_id = str(uuid.uuid4())
                        else:
                            thread_id = str(uuid.uuid4())
                    except:
                        thread_id = str(uuid.uuid4())
                else:
                    thread_id = str(uuid.uuid4())
            else:
                thread_id = str(uuid.uuid4())
        
        print("I can help you with:")
        print("• Cardiology guidelines and protocols")
        print("• Treatment recommendations")
        print("• Clinical questions")
        print("• General conversation")
        print("\nType 'quit', 'exit', or 'bye' to end the conversation")
        print(f"Thread ID: {thread_id[:8]}...")
        print("-"*60 + "\n")
        
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
    """Main function with conversation management options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cardiology Protocols Assistant')
    parser.add_argument('--resume', type=str, help='Resume conversation with thread ID')
    parser.add_argument('--list', action='store_true', help='List all conversations')
    
    args = parser.parse_args()
    
    try:
        orchestrator = Orchestrator()
        
        if args.list:
            orchestrator.list_conversations()
            return
        
        orchestrator.start_chat(resume_thread=args.resume)
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Failed to start orchestrator: {e}")
        print("Make sure all dependencies are installed and Qdrant is running")


if __name__ == "__main__":
    main()