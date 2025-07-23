#!/usr/bin/env python3

import uuid
from agent import Agent
from database import SyncDatabaseManager
from langchain_core.messages import HumanMessage, AIMessage

def interactive_chat():
    """Interactive chat mode for real-time conversation with the pipeline."""
    print("\n" + "="*60)
    print("CARDIOLOGY PROTOCOLS ASSISTANT")
    print("="*60)
    
    # Initialize database manager
    db_manager = SyncDatabaseManager()
    try:
        db_manager.initialize()
    except Exception as e:
        db_manager = None
    
    # Create admin user ID
    user_id = 'admin'
    thread_id = None
    
    # Ask if user wants to resume previous conversation
    if db_manager:
        try:
            # Check for existing conversations
            with db_manager.conn.cursor() as cur:
                cur.execute("""
                    SELECT thread_id, conversation_summary, created_at, updated_at
                    FROM conversations 
                    ORDER BY updated_at DESC 
                    LIMIT 10;
                """)
                conversations = cur.fetchall()
            
            if conversations:
                print("Previous conversations available:")
                for i, (thread_id, summary, created_at, updated_at) in enumerate(conversations, 1):
                    summary_preview = (summary[:50] + "...") if summary and len(summary) > 50 else (summary or "No summary")
                    print(f"  {i}. {thread_id[:8]}... - {summary_preview} (Updated: {updated_at.strftime('%Y-%m-%d %H:%M')})")
                
                choice = input("\nResume previous conversation? (y/n): ").strip().lower()
                if choice == 'y':
                    try:
                        selection = input("Enter conversation number (or press Enter for new): ").strip()
                        if selection.isdigit():
                            idx = int(selection) - 1
                            if 0 <= idx < len(conversations):
                                thread_id = conversations[idx][0]
                                print(f"Resuming conversation: {thread_id[:8]}...")
                                
                                # Show recent context from database
                                with db_manager.conn.cursor() as cur:
                                    cur.execute("""
                                        SELECT message_type, content, timestamp
                                        FROM messages m
                                        JOIN conversations c ON m.thread_id = c.id
                                        WHERE c.thread_id = %s
                                        ORDER BY timestamp DESC
                                        LIMIT 4;
                                    """, (thread_id,))
                                    recent_messages = cur.fetchall()
                                
                                if recent_messages:
                                    print("Recent context:")
                                    for msg_type, content, timestamp in reversed(recent_messages):
                                        role = "You" if msg_type == "HumanMessage" else "Assistant"
                                        content_preview = content[:60] + "..." if len(content) > 60 else content
                                        print(f"  {role}: {content_preview}")
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
                print("No previous conversations found. Starting new conversation.")
        except Exception as e:
            print(f"⚠️  Could not load conversation history: {e}")
            thread_id = str(uuid.uuid4())
    else:
        thread_id = str(uuid.uuid4())
    
    print("\nI can help you with:")
    print("• Cardiology guidelines and protocols")
    print("• Treatment recommendations")
    print("• Clinical questions")
    print("• General conversation")
    print("\nType 'quit' to end the conversation")
    print("Type '/feedback' for rating responses")
    print(f"User ID: {user_id}")
    print(f"Thread ID: {thread_id[:8]}...")
    print("-"*60 + "\n")
    
    # Initialize agent with thread_id
    agent = Agent(thread_id=thread_id)
    agent._initialize_components()
    agent._build_graph()
    
    try:
        while True:
            try:
                # Get user input
                query = input("You: ").strip()
                query_id = str(uuid.uuid4())
                
                if not query:
                    continue
                
                # Handle feedback command
                if query == '/feedback':
                    value_input = input('\nPositive/Negative: ').strip().lower()
                    if value_input not in ['positive', 'negative']:
                        print('Please enter either "Positive" or "Negative"')
                        continue
                    
                    value = True if value_input == 'positive' else False
                    comment = input('\nComment (optional): ').strip()
                    comment = comment if comment else None
                    
                    try:
                        if agent.memory and db_manager:
                            feedback_id = agent.memory.save_feedback(value, comment)
                            print('\nThanks for your feedback!')
                        else:
                            print('\n⚠️  Feedback system unavailable')
                    except Exception as e:
                        print(f'\nError saving feedback: {e}')
                    continue
                
                # Check for exit commands
                if query.lower() in ['quit']:
                    print("\nAssistant: Goodbye! Take care!")
                    break
                
                # Process query using the updated agent
                response = agent.process_query(query, user_id, thread_id)
                print("\nAssistant: ", end="", flush=True)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")
                
        # Clean up when conversation ends
        agent.cleanup()
        if db_manager:
            db_manager.close()
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Failed to start agent: {e}")
        print("Make sure all dependencies are installed and PostgreSQL is running")


def main():
    """Main function."""
    try:
        interactive_chat()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()
