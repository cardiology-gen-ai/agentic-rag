#!/usr/bin/env python3
import os, sys
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from langgraph.graph import START, END, StateGraph # type: ignore
from langchain_core.messages import HumanMessage, AIMessage # type: ignore
from langgraph.checkpoint.memory import InMemorySaver # type: ignore
from langgraph.store.memory import InMemoryStore # type: ignore
import uuid

from datetime import datetime

from src.utils.state import State
from src.agent import nodes

vectorstore_dir = os.path.join(project_root, '../data-etl/src')
if os.path.exists(vectorstore_dir):
    sys.path.insert(0, vectorstore_dir)

from vectorstore import load_vectorstore

class Agent():
    def __init__(self, agent_id: str, user_id: str, log_level: str = "INFO"):
        self.agent_id = agent_id
        self.user_id = user_id
        self.thread_id = '1'
        self._setup_logging(log_level)
        self.logger = logging.getLogger(f"{__name__}.Agent.{agent_id}")
        self.logger.info(f"Initializing agent with ID: {agent_id}")
        
        self.graph = self._create_graph()
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.compiled_graph = self.graph.compile(checkpointer = self.checkpointer, store = self.store)
        self.vectorstore = load_vectorstore(
            collection_name="cardio_protocols",
            vectorstore_type="qdrant"
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type = 'similarity',
            search_kwargs = {"k": 5}
        )
        self.logger.info("Agent initialization completed")
    
    def _setup_logging(self, log_level: str):
        """Configure logging for the agent system"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _retrieve(self, state: State) -> dict:
        human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        question = human_messages[-1].content
        # Ensure question is a string, not a list
        if isinstance(question, list):
            question = ' '.join(str(item) for item in question)
        elif not isinstance(question, str):
            question = str(question)
        
        self.logger.info(f"Retrieving documents for question: {question[:100]}...")
        documents = self.retriever.invoke(question)
        # Extract content from Document objects to match State schema
        document_contents = [doc.page_content for doc in documents]
        self.logger.info(f"Retrieved {len(document_contents)} documents")
        return {'documents': document_contents}
    
    def _create_graph(self):
        graph = StateGraph(State)

        graph.add_node('conversational_agent', nodes.conversational_agent)
        graph.add_node('retrieve', self._retrieve)
        graph.add_node('transform_question', nodes.question_rewriter)
        graph.add_node('generate', nodes.generate)

        graph.add_conditional_edges(
            START,
            nodes.route_question,
            {
                "conversational": "conversational_agent",
                "document_based": "retrieve",
            }
        )
        graph.add_conditional_edges(
            "retrieve",
            nodes.retrieval_grader,
            {
                "all_docs_not_relevant": "transform_question",
                "at_least_one_doc_relevant": "generate",
            },
        )
        graph.add_conditional_edges(
            "generate",
            nodes.ground_validator,
            {
                "grounded_and_addressed_question": END,
                "generation_not_grounded": "generate",
                "grounded_but_not_addressed_question": "transform_question",
            }
        )
        graph.add_edge("transform_question", "retrieve")
        graph.add_edge("conversational_agent", END)

        return graph
    
    def answer(self, question) -> str:
        self.logger.info(f"Processing question: {question[:100]}...")
        config = {"configurable": {"thread_id": self.thread_id, "user_id": self.user_id}}
        response = self.compiled_graph.invoke(
            {'messages': [HumanMessage(content=question)]}, 
            config=config
        )
        answer = response.get('generation') or response.get('response', '')
        self.logger.info(f"Generated answer: {answer[:100]}...")
        return answer
        

# Create the agent instance for LangGraph
agent = Agent(agent_id="test", user_id="default").compiled_graph
    
if __name__ == "__main__":
    import getpass
    
    # Simple authentication
    current_user = None
    current_thread_id = None
    agent = None
    
    # In-memory session storage
    user_sessions = {}  # {username: {thread_id: [messages]}}
    feedback_data = {}  # {thread_id: [feedback]}
    
    def authenticate():
        """Simple authentication function"""
        logger = logging.getLogger(__name__)
        print("=== Cardiology Agent Chat ===")
        print("Please login to continue.")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            username = input("Username: ").strip()
            password = getpass.getpass("Password: ")
            
            if username == "admin" and password == "admin":
                logger.info(f"User {username} authenticated successfully")
                print(f"Welcome, {username}!")
                return username
            else:
                remaining = max_attempts - attempt - 1
                logger.warning(f"Failed authentication attempt for user: {username}")
                if remaining > 0:
                    print(f"Invalid credentials. {remaining} attempts remaining.")
                else:
                    logger.error("Maximum authentication attempts exceeded")
                    print("Authentication failed. Goodbye.")
        return None
    
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
    
    def save_message(role, content):
        """Save message to session history"""
        logger = logging.getLogger(__name__)
        if current_user not in user_sessions:
            user_sessions[current_user] = {}
        if current_thread_id not in user_sessions[current_user]:
            user_sessions[current_user][current_thread_id] = []
        
        user_sessions[current_user][current_thread_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        logger.debug(f"Saved {role} message to session {current_thread_id}: {content[:50]}...")
    
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
    
    def session_menu():
        """Session selection menu"""
        global current_thread_id, agent
        logger = logging.getLogger(__name__)
        
        current_thread_id = create_new_session()
        agent = Agent(agent_id=f"agent_{current_thread_id[:8]}", user_id=current_user)
        agent.thread_id = current_thread_id
        return True
            
    def chat_loop():
        """Main chat loop"""
        logger = logging.getLogger(__name__)
        logger.info(f"Starting chat loop for session: {current_thread_id}")
        print(f"\n=== Chat Session {current_thread_id[:8]}... ===")
        print("Type your questions or use commands (type /help for available commands)")
        print("Press Ctrl+C to return to main menu\n")
        
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
                    print("Agent: ", end="", flush=True)
                    
                    # Save user message
                    save_message("user", user_input)
                    
                    # Get response from agent
                    response = agent.answer(user_input)
                    
                    # Save agent response
                    save_message("assistant", response)
                    
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
    
    # Main application flow
    try:
        # Authentication
        current_user = authenticate()
        if not current_user:
            exit(1)
        
        # Main loop
        while current_user:
            # Session selection
            if not session_menu():
                break
            
            # Chat loop
            if not chat_loop():
                break
    
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Application interrupted by user")
        print("\n\nGoodbye!")
    
    logger = logging.getLogger(__name__)
    logger.info("Chat application ended")
    print("Chat session ended.")




