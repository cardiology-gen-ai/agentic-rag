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
    print("\nJust type your medical questions to chat with the agent.")

def show_status():
    """Show current session status"""
    logger = logging.getLogger(__name__)
    print(f"User: {current_user}")
    print(f"Current Thread ID: {current_thread_id}")
    if current_user in user_sessions and current_thread_id in user_sessions[current_user]:
        messages = user_sessions[current_user][current_thread_id]
        print(f"Messages in session: {len(messages)}")
        logger.info(f"Status check - User: {current_user}, Thread: {current_thread_id}, Messages: {len(messages)}")
    else:
        print("No messages in current session")
        logger.info(f"Status check - User: {current_user}, Thread: {current_thread_id}, No messages")

def list_sessions():
    """List available sessions for current user"""
    logger = logging.getLogger(__name__)
    if current_user not in user_sessions or not user_sessions[current_user]:
        print("No previous sessions found.")
        logger.info(f"No previous sessions found for user: {current_user}")
        return
    
    session_count = len(user_sessions[current_user])
    logger.info(f"Listing {session_count} sessions for user: {current_user}")
    print(f"\nSessions for {current_user}:")
    for i, thread_id in enumerate(user_sessions[current_user].keys(), 1):
        msg_count = len(user_sessions[current_user][thread_id])
        print(f"  {i}. {thread_id[:8]}... ({msg_count} messages)")
    
    choice = input("\nEnter session number to recover (or press Enter for new): ").strip()
    if choice.isdigit():
        session_list = list(user_sessions[current_user].keys())
        idx = int(choice) - 1
        if 0 <= idx < len(session_list):
            selected_thread = session_list[idx]
            logger.info(f"User selected session: {selected_thread}")
            return selected_thread
    logger.info("User chose to create new session")
    return None

def create_new_session():
    """Create new session"""
    logger = logging.getLogger(__name__)
    thread_id = str(uuid.uuid4())
    if current_user not in user_sessions:
        user_sessions[current_user] = {}
    user_sessions[current_user][thread_id] = []
    logger.info(f"Created new session {thread_id} for user {current_user}")
    print(f"Created new session: {thread_id[:8]}...")
    return thread_id

def save_feedback(is_positive, comment=None):
    """Save feedback for current session"""
    logger = logging.getLogger(__name__)
    if current_thread_id not in feedback_data:
        feedback_data[current_thread_id] = []
    
    feedback_data[current_thread_id].append({
        "is_positive": is_positive,
        "comment": comment,
        "timestamp": datetime.now().isoformat()
    })
    logger.info(f"Feedback saved for session {current_thread_id}: {'positive' if is_positive else 'negative'}")
    print("Feedback saved successfully.")

def handle_command(command):
    """Handle chat commands"""
    global current_user, current_thread_id, agent
    
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
        if agent:
            agent.thread_id = current_thread_id
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
            if agent:
                agent.thread_id = current_thread_id
            logger.info(f"Switched to session: {thread_id}")
            print(f"Switched to session: {thread_id[:8]}...")
            # Display previous messages
            if current_user in user_sessions and current_thread_id in user_sessions[current_user]:
                messages = user_sessions[current_user][current_thread_id]
                print("\nPrevious messages:")
                for msg in messages[-5:]:  # Show last 5 messages
                    print(f"{msg['role'].title()}: {msg['content']}")
                print()
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
    
    return None

def initialize():
    """Session selection menu"""
    global current_thread_id, agent
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
    
    # In-memory session storage
    user_sessions = {}  # {username: {thread_id: [messages]}}
    feedback_data = {}  # {thread_id: [feedback]}
    
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




