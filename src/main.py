import uuid
from datetime import datetime, timezone
from time import time
from typing import Optional, cast, Literal

from persistence.db import get_sync_db
from agent.graph import Agent

from cardiology_gen_ai.utils.logger import get_logger

from persistence.message import ConversationTurn, RetrievalTurn, LLMTurn, FeedbackRequest, FeedbackTurn
from persistence.session import SessionDB
from persistence.user import UserORM, UserDB, UserCreateSchema
from utils.chat import MessageSchema, ChatRequest, ConversationRequest, ChatResponse, MessageRequest

logger = get_logger("Agentic RAG application")

app_id = "cardiology_protocols"


def show_help():
    """Display available commands"""
    print("\nAvailable Commands:")
    print("  /help      - Show this help message")
    print("  /new       - Start a new chat session")
    print("  /logout    - Logout and exit")
    print("  /feedback positive|negative [comment] - Provide feedback on last response")
    print("  /sessions  - Show available sessions")
    print("  /status    - Show current session status")
    # print("  /stats     - Show feedback statistics")
    print("\nJust type your medical questions to chat with the agent.")


def show_session_status(thread_id: str, thread_repo: SessionDB):
    """Show current session status"""
    session_info = thread_repo.sync_get_session(session_id=uuid.UUID(thread_id))
    if session_info is not None:
        logger.info(f"Successfully retrieved session data for session {thread_id}.")
        print(f"Messages in session: {session_info.message_count}")
        print(f"Session creation: {session_info.created_at}")
        print(f"Last updated: {session_info.updated_at}")
    else:
        logger.info(f"Session {thread_id} not found")
        print("Session not found in database")


def list_sessions(user_id: str, thread_repo: SessionDB):
    """List available sessions for current user"""
    logger.info(f"Listing sessions for user {user_id}")
    sessions = thread_repo.sync_get_user_sessions(uuid.UUID(user_id))
    if sessions is None or len(sessions) == 0:
        logger.info(f"No sessions for user {user_id}")
        return None
    session_count = len(sessions)
    logger.info(f"Listing {session_count} sessions for user: {user_id}")
    for i, session_i in enumerate(sessions, 1):
        logger.info(f"  {i}. {session_i.title} ({session_i.message_count} messages) - {session_i.updated_at}")
    choice = input("\nEnter session number to recover (or press Enter for new): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            selected_session = sessions[idx]
            logger.info(f"User selected session: {selected_session.session_id}")
            return selected_session.session_id
    logger.info("User chose to create new session")
    return None


def create_new_user(user_repo: UserDB, user_schema: UserCreateSchema):
    logger.info(f"Creating new user with username: {user_schema.username}")
    try:
        current_user_orm = user_repo.sync_create_user(user=user_schema)
        return current_user_orm
    except Exception as e:
        logger.error(f"Failed to create new user: {e}")
        raise


def get_user(user_repo: UserDB, username: Optional[str] = None, email: Optional[str] = None,
             user_id: Optional[uuid.UUID] = None):
    return user_repo.sync_get_user(username=username, email=email, user_id=user_id)


def save_agent_turn(rag_agent: Agent, request: ChatRequest, response: ChatResponse, duration: float = None):
    conversation_turn = ConversationTurn.from_agent(response=response, request=request)
    agent.memory.save_conversation_turn(conversation_turn)
    retrieval_turn = RetrievalTurn.from_agent(
        response=response, request=request, embedding_name=rag_agent.config.embeddings.model_name)
    agent.memory.save_retrieval_turn(retrieval_turn)
    llm_turn = LLMTurn.from_agent(response=response, request=request, llm_manager=rag_agent.llm_manager,
                                  duration=duration)
    agent.memory.save_llm_turn(llm_turn)
    logger.info("Saved agent interaction turn successfully.")


def get_session(thread_id: str, thread_repo: SessionDB):
    logger.info(f"Getting session {thread_id}")
    try:
        current_session = thread_repo.sync_get_session(session_id=uuid.UUID(thread_id))
        return current_session
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise


def rename_session(thread_id: str, thread_repo: SessionDB, new_session_title: str):
    logger.info(f"Renaming session {thread_id} as {new_session_title}")
    try:
        renamed_session = thread_repo.sync_update_session_title(
            session_id=uuid.UUID(thread_id), title=new_session_title)
        return renamed_session
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise


def delete_session(thread_id: str, thread_repo: SessionDB):
    logger.info(f"Deleting session {thread_id}")
    try:
        current_session = thread_repo.sync_delete_session_by_id(session_id=uuid.UUID(thread_id))
        return current_session
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise


def create_new_session(rag_agent: Agent, user: UserORM, thread_repo: SessionDB):
    """Create new session"""
    logger.info(f"Creating new thread for user: {user.username}..")
    try:
        current_thread = thread_repo.create_session(agent=rag_agent, user=user)
        thread_id = current_thread.session_id
        logger.info(f"Thread created with id: {thread_id}")
        welcome_messages = {
            "italian": "Ciao, per favore presentati e descrivi il tuo ruolo e funzionalità.",
            "english": "Hello, please introduce yourself and describe your role and functionalities."
        }
        lang_agent = rag_agent.config.language
        lang_message = lang_agent if lang_agent in welcome_messages.keys() else "english"
        question = MessageSchema(
            id=str(uuid.uuid4()),
            role="user",
            content=welcome_messages[lang_message],
            datetime=datetime.now(timezone.utc),
        )
        request = ChatRequest(
            user=user.username,
            user_id=str(user.user_id),
            conversation=ConversationRequest(
                id=str(thread_id),
                chatbotId=app_id,
                history=[],
                question=question,
            )
        )
        result: ChatResponse = rag_agent.answer(request=request)
        save_agent_turn(rag_agent=rag_agent, request=request, response=result)
        return thread_id
    except Exception as e:
        logger.error(f"Failed to create new thread: {e}")
        raise


def save_feedback(feedback_request: FeedbackRequest, user: UserORM, message_id: str, thread_id: str, rag_agent: Agent):
    """Save feedback for current session"""
    logger.info(f"Processing feedback for message {message_id}")
    try:
        feedback_turn = FeedbackTurn(
            message_id=message_id,
            user_id=user.user_id,
            session_id=thread_id,
            feedback_value=feedback_request.feedback_value,
            feedback_message=feedback_request.feedback_message,
            created_at=datetime.now(timezone.utc),
        )
        rag_agent.memory.save_feedback(feedback_turn)
    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")
        raise


def handle_command(command: str, user: UserORM, thread_id: str, message_id: str, thread_repo: SessionDB, rag_agent: Agent):
    """Handle chat commands"""
    command = command.strip().lower()
    if command == "/logout":
        logger.info(f"User {user.user_id} logging out")
        print("Logged out successfully.")
        return "logout"
    elif command == "/new":
        new_thread_id = create_new_session(rag_agent=rag_agent, user=user, thread_repo=thread_repo)
        if new_thread_id:
            logger.info(f"Started new session: {new_thread_id}")
            return new_thread_id
        return "continue"
    elif command == "/help":
        show_help()
        return "continue"
    elif command == "/status":
        show_session_status(thread_id=thread_id, thread_repo=thread_repo)
        return "continue"
    elif command == "/sessions":
        user_thread_id = list_sessions(user_id=str(user.user_id), thread_repo=thread_repo)
        if user_thread_id:
            logger.info(f"Switched to session: {user_thread_id}")
            print(f"Switched to session: {thread_id}...")
            print("Note: Previous messages are maintained by LangGraph checkpointer and will be available in conversation.")
            return user_thread_id
        return "continue"
    elif command.startswith("/feedback"):
        parts = command.split(maxsplit=2)
        if len(parts) < 2:
            print("Usage: /feedback [positive|negative] [optional comment]")
            return "continue"
        feedback_type = parts[1].lower()
        comment = parts[2] if len(parts) > 2 else None
        if feedback_type in ["positive", "pos", "good", "yes"]:
            feed_request = FeedbackRequest(
                feedback_value=1,
                feedback_message=comment,
            )
            save_feedback(feedback_request=feed_request, user=user, message_id=message_id, thread_id=thread_id,
                          rag_agent=rag_agent)
            return "continue"
        elif feedback_type in ["negative", "neg", "bad", "no"]:
            feed_request = FeedbackRequest(
                feedback_value=0,
                feedback_message=comment,
            )
            save_feedback(feedback_request=feed_request, user=user, message_id=message_id,
                          thread_id=thread_id, rag_agent=rag_agent)
            return "continue"
        else:
            logger.warning(f"Invalid feedback type provided: {feedback_type}")
            print("Feedback type must be 'positive' or 'negative'")
        return "continue"
    return None

def message(request: MessageRequest, user: UserORM, thread_id: str, thread_repo: SessionDB, rag_agent: Agent):
    logger.info(f"Processing message in thread {thread_id}")
    start_time = time()
    request_message = request.message
    message_history = rag_agent.memory.get_history(session_id=uuid.UUID(str(thread_id)),
                                                   limit=rag_agent.config.memory.length)
    history = []
    if len(message_history) > 0:
        for msg in message_history:
            history.append(MessageSchema(
                role="user",
                content=msg.question,
                datetime=msg.created_at,
            ))
            history.append(MessageSchema(
                role="assistant",
                content=msg.response,
                datetime=msg.created_at,
            ))
    try:
        msg_id = str(uuid.uuid4())
        question = MessageSchema(
            id=msg_id,
            role=cast(Literal["user", "assistant", "admin"], user.user_role),
            content=request_message,
            datetime=datetime.now(tz=timezone.utc),
        )
        chat_request = ChatRequest(
            user=user.username,
            user_id=str(user.user_id),
            conversation=ConversationRequest(
                id=str(thread_id),
                chatbotId=app_id,
                history=history,
                question=question,
            )
        )
        result = rag_agent.answer(request=chat_request)
        duration = round(time() - start_time, 2)
        print(result.content)
        if result:
            thread = thread_repo.sync_get_session(session_id=uuid.UUID(str(thread_id)))
            if thread and thread.title is None:
                logger.info(f"Setting thread name from first message")
                thread_repo.sync_update_session_title(session_id=uuid.UUID(str(thread_id)), title=request_message)
        save_agent_turn(rag_agent, chat_request, result, duration = duration)
        _ = thread_repo.sync_update_session_activity(session_id=uuid.UUID(str(thread_id)),
                                                                   increment_messages=True)
        return msg_id
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise


def chat_loop(thread_id: str, user: UserORM, thread_repo: SessionDB, rag_agent: Agent):
    """Main chat loop"""
    logger.info(f"Starting chat loop for session: {thread_id}")
    print(f"\n=== Chat Session {thread_id}... ===")
    print("Type your questions or use commands (type /help for available commands)")
    
    try:
        msg_id = str(uuid.uuid4())
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            # handle commands
            if user_input.startswith("/"):
                result = handle_command(command=user_input, user=user, thread_id=thread_id, thread_repo=thread_repo,
                                        rag_agent=rag_agent, message_id=msg_id)
                if result == "logout":
                    return False
                elif result == "continue":
                    continue
                elif user_input in ["/new", "/sessions"] and result != "continue":
                    thread_id = result
                    continue
            message_request = MessageRequest(message=user_input)
            msg_id = message(request=message_request, user=user, thread_id=thread_id, thread_repo=thread_repo,
                        rag_agent=rag_agent)
    except KeyboardInterrupt:
        logger.info("User interrupted chat loop")
        print("\n\nReturning to main menu...")
        return True


if __name__ == "__main__":
    agent = Agent(agent_id=app_id)
    session_generator = get_sync_db()
    session = next(session_generator)
    logged_user = UserCreateSchema(
        username="gaia",
        email="",
    )
    session_id = "c22c3e13-d229-46e8-a572-c24a8890bd87"
    try:
        session_db = SessionDB(session)
        user_db = UserDB(session)
        current_user = user_db.sync_get_user(username=logged_user.username)
        if current_user is None:
            current_user = create_new_user(user_repo=user_db, user_schema=logged_user)
        if ((session_id and get_session(thread_id=str(session_id), thread_repo=session_db) is None)
                or session_id is None):
            session_id = create_new_session(rag_agent=agent, user=current_user, thread_repo=session_db)
        chat_loop(thread_id=session_id, user=current_user, thread_repo=session_db, rag_agent=agent)
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass
