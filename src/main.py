#!/usr/bin/env python3

import os, sys
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_root)

from src.agent.production.graph import Agent

import uuid

from datetime import datetime

def show_help():
    """Display available commands"""
    print("\nAvailable Commands:")
    print("  /help      - Show this help message")
    print("  /new       - Start a new chat session")
    print("  /logout    - Logout and exit")
    print("  /feedback positive|negative [comment] - Provide feedback on last response")
    print("  /sessions  - Show available sessions")
    print("  /status    - Show current session status")
    print("  /stats     - Show feedback statistics")
    print("\nJust type your medical questions to chat with the agent.")

def show_status():
    """Show current session status"""
    logger = logging.getLogger(__name__)
    print(f"User: {current_user}")
    print(f"Current Thread ID: {current_thread_id}")
    
    if agent and current_thread_id:
        session = agent.data_layer.get_session(current_thread_id)
        if session:
            print(f"Messages in session: {session.message_count}")
            print(f"Session created: {session.created_at}")
            print(f"Last updated: {session.updated_at}")
            logger.info(f"Status check - User: {current_user}, Thread: {current_thread_id}, Messages: {session.message_count}")
        else:
            print("Session not found in database")
            logger.info(f"Status check - User: {current_user}, Thread: {current_thread_id}, Session not found")
    else:
        print("No active session")
        logger.info(f"Status check - User: {current_user}, Thread: {current_thread_id}, No active session")

def list_sessions():
    """List available sessions for current user"""
    logger = logging.getLogger(__name__)
    
    if not agent:
        print("No agent available.")
        return None
    
    sessions = agent.data_layer.get_user_sessions_with_metadata(current_user)
    
    if not sessions:
        print("No previous sessions found.")
        logger.info(f"No previous sessions found for user: {current_user}")
        return None
    
    session_count = len(sessions)
    logger.info(f"Listing {session_count} sessions for user: {current_user}")
    print(f"\nSessions for {current_user}:")
    for i, session in enumerate(sessions, 1):
        print(f"  {i}. {session.title} ({session.message_count} messages) - {session.updated_at[:10]}")
    
    choice = input("\nEnter session number to recover (or press Enter for new): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            selected_session = sessions[idx]
            logger.info(f"User selected session: {selected_session.session_id}")
            return selected_session.session_id
    logger.info("User chose to create new session")
    return None

def create_new_session():
    """Create new session"""
    logger = logging.getLogger(__name__)
    
    if not agent:
        logger.error("No agent available")
        return None
    
    # Ensure user exists in the database
    if not agent.data_layer.get_user(current_user):
        agent.data_layer.create_user(current_user, current_user)
        logger.info(f"Created new user: {current_user}")
    
    # Create new session
    thread_id = agent.data_layer.create_session(current_user)
    if thread_id:
        logger.info(f"Created new session {thread_id} for user {current_user}")
        print(f"Created new session: {thread_id[:8]}...")
        return thread_id
    else:
        logger.error("Failed to create new session")
        print("Failed to create new session")
        return None

def save_feedback(is_positive, comment=None):
    """Save feedback for current session"""
    logger = logging.getLogger(__name__)
    
    if not agent or not current_thread_id:
        print("No active session to save feedback for.")
        return
    
    # Generate a message_id (in a real app, you'd track the actual message ID from the AI response)
    message_id = str(uuid.uuid4())
    
    feedback_id = agent.data_layer.save_feedback(
        session_id=current_thread_id,
        user_id=current_user,
        message_id=message_id,
        is_positive=is_positive,
        comment=comment
    )
    
    if feedback_id:
        logger.info(f"Feedback saved for session {current_thread_id}: {'positive' if is_positive else 'negative'}")
        print("Feedback saved successfully.")
    else:
        logger.error("Failed to save feedback")
        print("Failed to save feedback.")

def handle_command(command):
    """Handle chat commands"""
    global current_user, current_thread_id, agent, config
    
    command = command.strip().lower()
    
    if command == "/logout":
        logger = logging.getLogger(__name__)
        logger.info(f"User {current_user} logging out")
        current_user = None
        current_thread_id = None
        agent = None
        print("Logged out successfully.")
        return "logout"
    
    elif command == "/new":
        logger = logging.getLogger(__name__)
        current_thread_id = create_new_session()
        if current_thread_id:
            config = {"configurable": {"thread_id": current_thread_id, "user_id": current_user}}
            logger.info(f"Started new session: {current_thread_id}")
        return "continue"
    
    elif command == "/help":
        show_help()
        return "continue"
    
    elif command == "/status":
        show_status()
        return "continue"
    
    elif command == "/sessions":
        logger = logging.getLogger(__name__)
        thread_id = list_sessions()
        if thread_id:
            current_thread_id = thread_id
            config = {"configurable": {"thread_id": current_thread_id, "user_id": current_user}}
            logger.info(f"Switched to session: {thread_id}")
            print(f"Switched to session: {thread_id[:8]}...")
            print("Note: Previous messages are maintained by LangGraph checkpointer and will be available in conversation.")
        return "continue"
    
    elif command.startswith("/feedback"):
        parts = command.split(maxsplit=2)
        if len(parts) < 2:
            print("Usage: /feedback [positive|negative] [optional comment]")
            return "continue"
        
        feedback_type = parts[1].lower()
        comment = parts[2] if len(parts) > 2 else None
        
        if feedback_type in ["positive", "pos", "good", "yes"]:
            save_feedback(True, comment)
        elif feedback_type in ["negative", "neg", "bad", "no"]:
            save_feedback(False, comment)
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"Invalid feedback type provided: {feedback_type}")
            print("Feedback type must be 'positive' or 'negative'")
        return "continue"
    
    elif command == "/stats":
        logger = logging.getLogger(__name__)
        if not agent:
            print("No agent available.")
            return "continue"
        
        # Get feedback stats for current user
        stats = agent.data_layer.get_feedback_stats(user_id=current_user)
        print(f"\nFeedback Statistics for {current_user}:")
        print(f"  Total feedback: {stats['total']}")
        print(f"  Positive: {stats['positive']}")
        print(f"  Negative: {stats['negative']}")
        
        if current_thread_id:
            session_stats = agent.data_layer.get_feedback_stats(session_id=current_thread_id)
            print(f"\nCurrent Session Feedback:")
            print(f"  Total feedback: {session_stats['total']}")
            print(f"  Positive: {session_stats['positive']}")
            print(f"  Negative: {session_stats['negative']}")
        
        logger.info(f"Displayed feedback stats for user: {current_user}")
        return "continue"
    
    return None

def initialize():
    """Session selection menu"""
    global current_thread_id, agent, config
    logger = logging.getLogger(__name__)
    
    agent = Agent(agent_id='cardiology')
    current_thread_id = create_new_session()
    config = {"configurable": {"thread_id": current_thread_id, "user_id": current_user}}
    return config
        
def chat_loop():
    """Main chat loop"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting chat loop for session: {current_thread_id}")
    print(f"\n=== Chat Session {current_thread_id[:8]}... ===")
    print("Type your questions or use commands (type /help for available commands)")
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                result = handle_command(user_input)
                if result == "logout":
                    return False
                elif result == "continue":
                    continue
            
            # Process question with agent
            try:
                # Get response from agent
                response = agent.answer(user_input, config)

                print("\nAgent: ", end="", flush=True)
                print(response)
                print()
                
                # Update session activity and message count
                if current_thread_id:
                    agent.data_layer.update_session_activity(current_thread_id, increment_messages=True)
                    agent.data_layer.update_user_activity(current_user)
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print(f"Error processing question: {e}")
                print("Please try again or type /help for available commands.\n")
    
    except KeyboardInterrupt:
        logger.info("User interrupted chat loop")
        print("\n\nReturning to main menu...")
        return True

if __name__ == "__main__":
    # Simple authentication
    current_user = 'admin'
    current_thread_id = None
    agent = None
    config = None
    
    try:
        config = initialize()
        chat_loop()
    
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Application interrupted by user")
        print("\n\nGoodbye!")
    
    logger = logging.getLogger(__name__)
    logger.info("Chat application ended")
    print("Chat session ended.")




