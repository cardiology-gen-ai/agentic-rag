#!/usr/bin/env python3

import os, sys

# Add the current directory (src) to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from agent.agent import Agent
from sqlite.manager import StateManager
from agent.utils.state import State

from typing import List
import uuid
from langchain_core.messages import HumanMessage

def load_conversation(state_manager, thread_id) -> List:
        """Load existing conversation from SQLite."""
        return state_manager.load_thread_messages(thread_id)
    
def list_conversations(state_manager):
    """List all available conversations."""
    threads = state_manager.list_threads()
    
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


def main():
    """Main function with conversation management options."""
    try:
        state_manager = StateManager()
        agent = Agent(state_manager = state_manager)
        state = State()

        print("\n" + "="*60)
        print("CARDIOLOGY PROTOCOLS ASSISTANT")
        print("="*60)
        # Ask if user wants to resume
        threads = state_manager.list_threads()
        if threads:
            print("Previous conversations available.")
            choice = input("Resume previous conversation? (y/n): ").strip().lower()
            if choice == 'y':
                list_conversations(state_manager=state_manager)
                try:
                    selection = input("Enter conversation number (or press Enter for new): ").strip()
                    if selection.isdigit():
                        idx = int(selection) - 1
                        if 0 <= idx < len(threads):
                            thread_id = threads[idx][0]
                            print(f"Resuming conversation: {thread_id[:8]}...")
                            # Show context
                            messages = load_conversation(thread_id)
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

        state['thread_id'] = thread_id 
        
        while True:
            try:
                query = input("You: ").strip()
                
                if not query:
                    continue
                
                # Check for exit commands
                if query.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nAssistant: Goodbye! Take care!")
                    break

                state['query'] = query
                
                # Process message
                print("\nAssistant: ", end="", flush=True)
                response = agent.process_message(state)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")

        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Failed to start agent: {e}")
        print("Make sure all dependencies are installed and Qdrant is running")


if __name__ == "__main__":
    main()