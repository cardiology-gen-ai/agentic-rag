#!/usr/bin/env python3
"""
Pipeline orchestrator for the Cardiology Protocols RAG Pipeline.
Orchestrates the flow between memory management, query routing, 
conversational agent, and self-RAG components using LangGraph.
"""

import os
import sys
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

# Import pipeline components
from state import State
from memory import Memory
from router import Router
from conversational_agent import ConversationalAgent

current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_dir = os.path.join(current_dir, '../../data-etl/src')
if os.path.exists(vectorstore_dir):
    sys.path.insert(0, vectorstore_dir)

# Try to import vectorstore and self-RAG components
try:
    from vectorstore import load_vectorstore
    from self_rag import SelfRAG
    VECTORSTORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Vectorstore/Self-RAG components not available: {e}")
    VECTORSTORE_AVAILABLE = False


class Orchestrator:
    """
    Main pipeline orchestrator that manages the entire cardiology RAG workflow.
    
    This orchestrator:
    1. Initializes all pipeline components
    2. Builds a LangGraph workflow
    3. Routes queries based on type (conversational vs document-based)
    4. Manages memory and conversation context
    5. Provides a unified interface for the pipeline
    """
    
    def __init__(self):
        """Initialize the orchestrator with empty components."""
        self.router = None
        self.memory = None
        self.self_rag = None
        self.conversational_agent = None
        self.vectorstore = None
        self.graph = None
        
        print("🏥 Cardiology RAG Pipeline Orchestrator initialized")

    def initialize_components(self):
        """Initialize all pipeline components."""
        print("\n🚀 Initializing Cardiology RAG Pipeline")
        print("=" * 60)
        
        try:
            # Initialize basic components (always available)
            print("1. Initializing Router...")
            self.router = Router()
            
            print("2. Initializing Memory...")
            self.memory = Memory(max_tokens=2000)
            
            print("3. Initializing Conversational Agent...")
            self.conversational_agent = ConversationalAgent()
            
            # Initialize vectorstore and Self-RAG if available
            if VECTORSTORE_AVAILABLE:
                print("4. Loading Vectorstore...")
                try:
                    self.vectorstore = load_vectorstore(
                        collection_name="cardio_protocols",
                        vectorstore_type="qdrant"
                    )
                    
                    print("5. Initializing Self-RAG...")
                    self.self_rag = SelfRAG(self.vectorstore)
                    
                    print("✅ All components initialized successfully!")
                    
                except Exception as e:
                    print(f"❌ Error initializing vectorstore/Self-RAG: {e}")
                    print("   Pipeline will run in conversational-only mode")
                    self.self_rag = None
                    self.vectorstore = None
            else:
                print("4. ❌ Vectorstore not available - running in conversational-only mode")
                print("   Install required dependencies to enable RAG functionality")
                self.self_rag = None
                self.vectorstore = None
            
            print("\n📊 Component Status:")
            print(f"   ✅ Router: {'Available' if self.router else 'Not Available'}")
            print(f"   ✅ Memory: {'Available' if self.memory else 'Not Available'}")
            print(f"   ✅ Conversational Agent: {'Available' if self.conversational_agent else 'Not Available'}")
            print(f"   {'✅' if self.vectorstore else '❌'} Vectorstore: {'Available' if self.vectorstore else 'Not Available'}")
            print(f"   {'✅' if self.self_rag else '❌'} Self-RAG: {'Available' if self.self_rag else 'Not Available'}")
            
        except Exception as e:
            print(f"❌ Error during component initialization: {e}")
            raise

    def build_graph(self):
        """
        Build the LangGraph workflow for the pipeline.
        
        Returns:
            Compiled LangGraph workflow
        """
        print("\n🔧 Building LangGraph Workflow...")
        
        if not all([self.router, self.memory, self.conversational_agent]):
            raise ValueError("Essential components not initialized. Run initialize_components() first.")
        
        # Create the workflow
        workflow = StateGraph(State)
        
        # Add nodes
        print("   Adding workflow nodes...")
        workflow.add_node("memory", self.memory.memory_management_node)
        workflow.add_node("router", self.router.router_node)
        workflow.add_node("conversational_agent", self.conversational_agent.conversational_agent_node)
        
        # Conditionally add self_rag node if available
        if self.self_rag:
            workflow.add_node("self_rag", self.self_rag.selfRAG_node)
            print("   ✅ Self-RAG node added")
        else:
            print("   ⚠️  Self-RAG node not available - using conversational fallback")
        
        # Add edges
        print("   Connecting workflow edges...")
        workflow.add_edge(START, "memory")
        workflow.add_edge("memory", "router")
        
        # Conditional routing based on query type
        routing_map = {
            "conversational": "conversational_agent",
            "document_based": "self_rag" if self.self_rag else "conversational_agent"
        }
        
        workflow.add_conditional_edges(
            "router",
            self.router.route_query,
            routing_map
        )
        
        # Add final edges
        workflow.add_edge("conversational_agent", END)
        if self.self_rag:
            workflow.add_edge("self_rag", END)
        
        # Compile the graph
        print("   Compiling workflow...")
        self.graph = workflow.compile()
        
        print("✅ LangGraph workflow compiled successfully!")
        print(f"   Routing: {routing_map}")
        
        return self.graph

    def process_query(self, query: str, context: dict = None) -> dict:
        """
        Process a user query through the complete pipeline.
        
        Args:
            query: User query string
            context: Optional context dictionary
            
        Returns:
            dict: Final state with response and metadata
        """
        if not self.graph:
            raise ValueError("Workflow not built. Run build_graph() first.")
        
        print(f"\n🔄 Processing Query: '{query[:50]}...'")
        print("=" * 60)
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query_type": None,
            "response": None,
            "context": context or {},
            "conversation_summary": None,
            "documents": None,
            "metadata": {},
            "current_state": None,
            "next_action": None
        }
        
        try:
            # Process through the workflow
            final_state = self.graph.invoke(initial_state)
            
            # Extract and display results
            response = final_state.get("response", "No response generated")
            query_type = final_state.get("query_type", "unknown")
            metadata = final_state.get("metadata", {})
            
            print(f"\n📋 Results:")
            print(f"   Query Type: {query_type}")
            print(f"   Response: {response[:100]}...")
            print(f"   Metadata: {metadata}")
            
            return final_state
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            # Return error state
            return {
                **initial_state,
                "response": f"I encountered an error processing your query: {str(e)}",
                "metadata": {"error": str(e), "status": "failed"}
            }

    def process_conversation(self, messages: list) -> dict:
        """
        Process a multi-turn conversation through the pipeline.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            dict: Final state with response and metadata
        """
        if not self.graph:
            raise ValueError("Workflow not built. Run build_graph() first.")
        
        print(f"\n💬 Processing Conversation: {len(messages)} messages")
        print("=" * 60)
        
        # Create initial state with conversation history
        initial_state = {
            "messages": messages,
            "query_type": None,
            "response": None,
            "context": {},
            "conversation_summary": None,
            "documents": None,
            "metadata": {},
            "current_state": None,
            "next_action": None
        }
        
        try:
            # Process through the workflow
            final_state = self.graph.invoke(initial_state)
            return final_state
            
        except Exception as e:
            print(f"❌ Error processing conversation: {e}")
            return {
                **initial_state,
                "response": f"I encountered an error processing the conversation: {str(e)}",
                "metadata": {"error": str(e), "status": "failed"}
            }

    def get_system_status(self) -> dict:
        """
        Get the current status of all pipeline components.
        
        Returns:
            dict: System status information
        """
        return {
            "router": self.router is not None,
            "memory": self.memory is not None,
            "conversational_agent": self.conversational_agent is not None,
            "vectorstore": self.vectorstore is not None,
            "self_rag": self.self_rag is not None,
            "graph": self.graph is not None,
            "vectorstore_available": VECTORSTORE_AVAILABLE,
            "mode": "full_rag" if self.self_rag else "conversational_only"
        }

    def initialize_and_build(self):
        """
        Convenience method to initialize components and build the graph in one step.
        
        Returns:
            Compiled LangGraph workflow
        """
        self.initialize_components()
        return self.build_graph()


def test_orchestrator():
    """Test function for the orchestrator."""
    print("🧪 Testing Pipeline Orchestrator")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Initialize and build pipeline
    try:
        graph = orchestrator.initialize_and_build()
        print("✅ Pipeline built successfully!")
        
        # Get system status
        status = orchestrator.get_system_status()
        print(f"\n📊 System Status: {status}")
        
        # Test queries
        test_queries = [
            "Hello! I'm new here.",  # Conversational
            "What is the ESC protocol for acute MI?",  # Document-based
            "Thank you for your help!",  # Conversational
            "How do you manage atrial fibrillation?",  # Document-based
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Test Query {i}: '{query}'")
            print(f"{'='*60}")
            
            result = orchestrator.process_query(query)
            
            print(f"\nResult {i}:")
            print(f"   Response: {result.get('response', 'No response')[:150]}...")
            print(f"   Query Type: {result.get('query_type', 'Unknown')}")
            print(f"   Status: {result.get('metadata', {}).get('status', 'Unknown')}")
        
        # Test conversation
        print(f"\n{'='*60}")
        print("Testing Multi-turn Conversation")
        print(f"{'='*60}")
        
        conversation_messages = [
            HumanMessage(content="Hi there!"),
            AIMessage(content="Hello! How can I help with cardiology?"),
            HumanMessage(content="What can you tell me about heart failure?")
        ]
        
        conv_result = orchestrator.process_conversation(conversation_messages)
        print(f"Conversation result: {conv_result.get('response', 'No response')[:100]}...")
        
    except Exception as e:
        print(f"❌ Error testing orchestrator: {e}")
        return False
    
    return True


def interactive_chat():
    """Interactive chat mode for real-time conversation with the pipeline."""
    print("💬 Interactive Cardiology Assistant Chat")
    print("=" * 60)
    print("Welcome to the ESC Cardiology Protocols Assistant!")
    print("Ask me about cardiology guidelines, protocols, and procedures.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'help' for assistance or 'status' to check system status.")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    try:
        # Initialize and build pipeline
        print("\n🚀 Starting up the assistant...")
        graph = orchestrator.initialize_and_build()
        print("✅ Assistant ready!")
        
        # Get system status
        status = orchestrator.get_system_status()
        mode = "🔬 Full RAG Mode" if status["self_rag"] else "💬 Conversational Mode"
        print(f"\n🏥 Running in: {mode}")
        
        if not status["self_rag"]:
            print("⚠️  Note: RAG functionality unavailable. Install vectorstore dependencies for full features.")
        
        print(f"\n{'='*60}")
        
        # Conversation loop
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n🧑‍⚕️ You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n🏥 Assistant: Thank you for using the Cardiology Assistant. Take care!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n🏥 Assistant: " + orchestrator.conversational_agent.get_help_response())
                    continue
                
                elif user_input.lower() == 'status':
                    status = orchestrator.get_system_status()
                    print(f"\n📊 System Status:")
                    for component, available in status.items():
                        status_icon = "✅" if available else "❌"
                        print(f"   {status_icon} {component.replace('_', ' ').title()}: {available}")
                    continue
                
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("\n🔄 Conversation history cleared.")
                    continue
                
                elif not user_input:
                    print("Please enter a question or command.")
                    continue
                
                # Add user message to history
                conversation_history.append(HumanMessage(content=user_input))
                
                # Process query
                print("\n🤔 Assistant: Processing your question...")
                
                # Use conversation history for context
                result = orchestrator.process_conversation(conversation_history)
                
                # Get response
                response = result.get("response", "I'm sorry, I couldn't generate a response.")
                query_type = result.get("query_type", "unknown")
                
                # Display response with type indicator
                type_icon = "🔬" if query_type == "document_based" else "💬"
                print(f"\n🏥 Assistant {type_icon}: {response}")
                
                # Add assistant response to history
                conversation_history.append(AIMessage(content=response))
                
                # Optional: Show metadata in verbose mode
                metadata = result.get("metadata", {})
                if os.environ.get("VERBOSE") == "1":
                    print(f"\n📊 Debug Info: Type={query_type}, Metadata={metadata}")
                
                # Manage conversation length (keep last 10 messages)
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
            except KeyboardInterrupt:
                print("\n\n🏥 Assistant: Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with a different question.")
    
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {e}")
        print("Please check your setup and try again.")


def show_menu():
    """Display the main menu for mode selection."""
    print("🏥 Cardiology Protocols RAG Pipeline")
    print("=" * 60)
    print("Please choose a mode:")
    print()
    print("1. 💬 Interactive Chat Mode")
    print("   - Real-time conversation with the assistant")
    print("   - Ask questions about cardiology protocols")
    print("   - Type commands for help and status")
    print()
    print("2. 🧪 Test Mode")
    print("   - Run automated tests on the pipeline")
    print("   - Verify component functionality")
    print("   - Display system diagnostics")
    print()
    print("3. ❌ Exit")
    print()


def main():
    """Main function with mode selection."""
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                interactive_chat()
                break
            elif choice == "2":
                success = test_orchestrator()
                if success:
                    print("\n🎉 Test completed successfully!")
                else:
                    print("\n❌ Test failed!")
                
                # Ask if user wants to continue
                continue_choice = input("\nWould you like to return to the main menu? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
            elif choice == "3":
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid choice. Please enter 1, 2, or 3.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
