import uuid
import json
from pathlib import Path
from datetime import datetime, timezone

from cardiology_gen_ai.utils.logger import get_logger

from src.agentic_rag.persistence import get_sync_db
from src.agentic_rag.agent.graph import Agent
from src.agentic_rag.persistence import SessionDB
from src.agentic_rag.persistence.user import UserDB, UserCreateSchema
from src.agentic_rag.utils.chat import MessageSchema, ChatRequest, ConversationRequest
from src.main import create_new_user, get_session, create_new_session

logger = get_logger("Agentic RAG Test")

app_id = "cardiology_protocols"

def get_questions(questions_file: str = "test_questions.json"):
    """Load questions from JSON file"""
    # Look for the file in the parent directory of agentic-rag
    base_path = Path(__file__).parent.parent.parent
    questions_path = base_path / questions_file

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract all questions from the nested structure
    all_questions = []
    for guideline_key, guideline_data in data.get("guidelines", {}).items():
        if "questions" in guideline_data:
            for q in guideline_data["questions"]:
                all_questions.append({
                    "guideline": guideline_key,
                    "guideline_name": guideline_data.get("guideline_name", ""),
                    **q
                })

    logger.info(f"Loaded {len(all_questions)} questions from {questions_path}")
    return all_questions

def save_results(results: dict, output_dir: str = "output"):
    """Save results to JSON file in output directory"""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_file = output_path / "out.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_file}")
    return output_file

if __name__ == "__main__":
    logger.info("Starting test run...")

    agent = Agent(agent_id=app_id)
    session_generator = get_sync_db()
    session = next(session_generator)
    logged_user = UserCreateSchema(
        username="test_user",
        email="test@example.com",
    )
    session_id = None

    try:
        session_db = SessionDB(session)
        user_db = UserDB(session)
        current_user = user_db.sync_get_user(username=logged_user.username)
        if current_user is None:
            current_user = create_new_user(user_repo=user_db, user_schema=logged_user)

        if ((session_id and get_session(thread_id=str(session_id), thread_repo=session_db) is None)
            or session_id is None):
            session_id = create_new_session(rag_agent=agent, user=current_user, thread_repo=session_db)

        # Load questions
        questions = get_questions()

        # Extract agent configuration/hyperparameters
        agent_config = {
            "agent_name": agent.agent_name,
            "llm": {
                "model_name": agent.config.llm.model_name,
                "ollama": agent.config.llm.ollama,
                "nbits": agent.config.llm.nbits,
                "generator_temperature": agent.config.llm.generator_temperature,
                "router_temperature": agent.config.llm.router_temperature,
                "grader_temperature": agent.config.llm.grader_temperature,
            },
            "embeddings": {
                "model_name": agent.config.embeddings.model_name,
                "model_type": agent.config.embeddings.model_type,
            },
            "search": {
                "type": agent.config.search.type.value,
                "top_k": agent.config.search.top_k,
                "fetch_k": agent.config.search.fetch_k,
                "score_threshold": agent.config.search.score_threshold,
            },
            "indexing": {
                "index_name": agent.config.indexing.index_name,
                "description": agent.config.indexing.description,
            },
            "memory": {
                "length": agent.config.memory.length,
            },
            "generation_limit": 2,  # From GENERATION_LIMIT constant
        }

        # Prepare results structure matching input format
        results = {
            "metadata": {
                "test_date": datetime.now(timezone.utc).isoformat(),
                "total_questions": len(questions),
                "agent_id": app_id,
                "session_id": str(session_id)
            },
            "agent_config": agent_config,
            "guidelines": {}
        }

        # Process each question
        for idx, question_data in enumerate(questions, 1):
            question_id = question_data.get("id", f"Q_{idx}")
            question_text = question_data.get("question", "")
            guideline_key = question_data.get("guideline", "unknown")

            logger.info(f"Processing question {idx}/{len(questions)}: {question_id}")

            # Create chat request
            msg_id = str(uuid.uuid4())
            chat_request = ChatRequest(
                user=current_user.username,
                user_id=str(current_user.user_id),
                conversation=ConversationRequest(
                    id=str(session_id),
                    chatbotId=app_id,
                    history=[],
                    question=MessageSchema(
                        id=msg_id,
                        role="user",
                        content=question_text,
                        datetime=datetime.now(timezone.utc),
                    )
                )
            )

            # Get agent response
            try:
                agent_response = agent.answer(chat_request)
                if agent_response:
                    answer = agent_response.content
                    if agent_response.is_faulted:
                        logger.warning(f"⚠ Question {question_id} returned faulted response")
                    logger.info(f"✓ Answered question {question_id}")
                else:
                    answer = "ERROR: Agent returned None"
                    logger.error(f"✗ Agent returned None for question {question_id}")
            except Exception as e:
                logger.error(f"✗ Failed to answer question {question_id}: {e}", exc_info=True)
                answer = f"ERROR: {str(e)}"

            # Organize results by guideline
            if guideline_key not in results["guidelines"]:
                results["guidelines"][guideline_key] = {
                    "guideline_name": question_data.get("guideline_name", ""),
                    "questions": []
                }

            # Add question with answer
            question_result = {
                **question_data,
                "answer": answer,
                "answered_at": datetime.now(timezone.utc).isoformat()
            }
            results["guidelines"][guideline_key]["questions"].append(question_result)

        # Save results
        output_file = save_results(results)
        logger.info(f"✓ Test completed. Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        try:
            next(session_generator)
        except StopIteration:
            pass
