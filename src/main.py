#!/usr/bin/env python3

import os, sys

# Add the current directory (src) to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from agent.agent import Agent
from sqlite.manager import StateManager

from typing import List
import uuid
from langchain_core.messages import HumanMessage
from agent.state import State

def main():
    """Main function with conversation management options."""
    try:
        state_manager = StateManager()
        agent = Agent(state_manager = state_manager)
        state = State()

        user_id = "admin" # for developing purpose
        state["user_id"] = user_id
        state["previous_messages"] = []
        state["conversation_summary"] = None
        state["is_query"] = False
        state["query_type"] = None
        state["rewritten_query"] = None
        state["user_context"] = None
        state["documents"] = None
        state["feedback"] = None
        state["response"] = None
        state["metadata"] = None
        state["retrieval_attempts"] = None
        state["generation_attempts"] = None
        
        # Track message IDs for feedback
        last_ai_message_id = None
        
        print("\n" + "="*60)
        print("CARDIOLOGY PROTOCOLS ASSISTANT")
        print("="*60)
        # Ask if user wants to resume
        conversations = state_manager.get_user_conversations(user_id)
        if conversations:
            print("Previous conversations available.")
            choice = input("Resume previous conversation? (y/n): ").strip().lower()
            if choice == 'y':
                state_manager.list_conversations(user_id)
                try:
                    selection = input("Enter conversation number (or press Enter for new): ").strip()
                    if selection.isdigit():
                        idx = int(selection) - 1
                        if 0 <= idx < len(conversations):
                            conversation_id = conversations[idx][0]
                            print(f"Resuming conversation: {conversation_id[:8]}...")
                            # Show context
                            messages = state_manager.get_conversation_messages(conversation_id)
                            if messages:
                                print("Recent context:")
                                for msg in messages[-4:]:
                                    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
                                    content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                                    print(f"  {role}: {content}")
                                print()
                        else:
                            conversation_id = str(uuid.uuid4())
                    else:
                        conversation_id = str(uuid.uuid4())
                except:
                    conversation_id = str(uuid.uuid4())
            else:
                conversation_id = str(uuid.uuid4())
        else:
            conversation_id = str(uuid.uuid4())

        print("I can help you with:")
        print("• Cardiology guidelines and protocols")
        print("• Treatment recommendations")
        print("• Clinical questions")
        print("• General conversation")
        print("\nType 'quit', 'exit', or 'bye' to end the conversation")
        print(f"Thread ID: {conversation_id[:8]}...")
        print("-"*60 + "\n")

        while True:
            try:
                query = input("You: ").strip()
                query_id = str(uuid.uuid4())
                
                if not query:
                    continue

                if query == '/feedback':
                    value_input = input('\nPositive/Negative: ').strip().lower()
                    if value_input not in ['Positive', 'Negative']:
                        print('Please enter either "Positive" or "Negative"')
                        continue
                    
                    value = 1 if value_input == 'positive' else 0
                    comment = input('\nComment (optional): ').strip()
                    comment = comment if comment else None
                    
                    try:
                        state_manager.add_feedback(state.get("message"), value, comment)
                        print('\nThanks for your feedback!')
                    except Exception as e:
                        print(f'\nError saving feedback: {e}')
                    continue
                
                # Check for exit commands
                if query.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nAssistant: Goodbye! Take care!")
                    break

                state['message'] = query
                state['conversation_id'] = conversation_id
                state['query_id'] = query_id
                
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
