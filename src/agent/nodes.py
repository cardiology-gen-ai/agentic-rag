#!/usr/bin/env python3 
"""
Utility functions that act as nodes in the agent.
"""
import sys, os
import logging
from datetime import datetime 
from pydantic import BaseModel, Field # type: ignore

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langgraph.types import Command # type: ignore
from langchain_ollama import ChatOllama # type: ignore
from langchain_core.messages import HumanMessage, AIMessage # type: ignore

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.utils.state import State

# Module-level logger
logger = logging.getLogger(__name__)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="The relevance score: 'yes' if document is relevant, 'no' if not relevant")

class GradeGrounding(BaseModel):
    """Binary score for grounding check on generation."""
    binary_score: str = Field(description="The grounding score: 'yes' if generation is grounded in facts, 'no' if not grounded")

class GradeAnswer(BaseModel):
    """Binary score for answer addressing question."""
    binary_score: str = Field(description="The answer score: 'yes' if answer addresses the question, 'no' if it doesn't")

def route_question(state: State) -> str:
    logger.info("Starting question routing")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.1, verbose=False)
    human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    question = human_messages[-1].content
    logger.debug(f"Question to route: {question[:100]}...")
    system_prompt = ("""
        You are a query router for a cardiology guidelines system. 
        Determine whether the user's question is a conversational inquiry, meaning it is general, casual, or social. 
        This can include, but it is not limited to greetings (e.g., 'Hello'), small talk (e.g., 'How are you?', 'What information can you give me?, 'What documentation can you provide me?'), 
        personal opinions, and polite expressions of gratitude (e.g., 'Thank you').
        
        If the user question is a conversational inquiry, respond with exactly: conversational
        If the user question is NOT a conversational inquiry (i.e., it's asking for specific medical/clinical information), respond with exactly: document_based
        """)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: {question}"),
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    response = runnable.invoke(
        {'question': question}
    )
    response = response.strip().lower()
    
    # Extract only the routing decision from the response
    if 'conversational' in response:
        route = 'document_based'  # NOT USING conversational FOR TESTING PURPOSES
        logger.info(f"Question routed to: {route} (override)")
        return route
    elif 'document_based' in response:
        route = 'document_based'
        logger.info(f"Question routed to: {route}")
        return route
    else:
        # Default to document_based for unclear responses
        route = 'document_based'
        logger.warning(f"Unclear routing response, defaulting to: {route}")
        return route

def conversational_agent(state: State) -> dict:
    logger.info("Processing conversational query")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.7, verbose=False)
    human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    question = human_messages[-1].content
    messages = state.messages
    history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[:-1]]) if messages else ""
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_prompt = (
        f"Today is {current_datetime}."
        "You are a helpful conversational agent. "
        "Your task is to provide clear, accurate, and helpful responses to user questions and requests. "
        "Keep your responses natural and conversational—avoid being overly formal or robotic. "
        "Be concise but complete—give enough information to be useful without unnecessary details. "
        "If you don't know something, say so clearly rather than guessing. "
        "If a question is unclear, ask for clarification. "
        "Stay focused on being helpful and direct in your responses."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: {question}, \nChat history: {history}"),
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    response = runnable.invoke(
        {"question": question, "history": history}
    )
    logger.info(f"Generated conversational response: {response[:100]}...")
    return {'response': response}

def retrieval_grader(state: State):
    logger.info(f"Grading {len(state.documents)} retrieved documents")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.7, verbose=False)
    human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    question = human_messages[-1].content
    documents = state.documents
    
    system_prompt = """
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document or the document filename contain keyword(s) or semantic meaning related to the question, grade it as relevant.
    
    You MUST respond with ONLY 'yes' or 'no' - nothing else.
    - Use 'yes' if the document is relevant to the question
    - Use 'no' if the document is not relevant to the question
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Retrieved document filename: {document_filename} \n\n Retrieved document: {document} \n\n User question: {question}"
            )
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    filtered_docs = []
    for idx, d in enumerate(documents):
        try:
            response = runnable.invoke(
                {"question": question, "document": d, "document_filename": f"document_{idx}"}
            )
            grade = response.strip().lower()
            if "yes" in grade:
                filtered_docs.append(d)
        except Exception as e:
            logger.warning(f"Error grading document {idx}: {e}, assuming relevant")
            filtered_docs.append(d)  # Default to including the document
    
    # Update the state with filtered documents
    state.documents = filtered_docs
    
    if len(filtered_docs) == 0:
        logger.info("All documents marked as not relevant")
        return "all_docs_not_relevant"
    else:
        logger.info(f"{len(filtered_docs)} documents marked as relevant")
        return "at_least_one_doc_relevant"

def generate(state: State):
    logger.info(f"Generating answer using {len(state.documents)} documents")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.7, verbose=False)
    human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    question = human_messages[-1].content 
    documents = state.documents
    # Documents are now strings (page content), not dicts
    context = "\n\n".join([f"Document {idx+1}:\n{doc}" for idx, doc in enumerate(documents)])
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_prompt = f"""
    Today is {current_datetime}. \n
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question\n
    If you don't know the answer, just say that you don't know.\n
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Retrieved information: \n{documents}\n\nQuestion: \n{question}\n\nAnswer:",
            )
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    response = runnable.invoke(
        {
            'documents': context,
            'question': question,
        }
    )
    logger.info(f"Generated response (attempt {state.generation_count + 1}): {response[:100]}...")
    return {
        'response': response,
        'generation_count': state.generation_count + 1,
    }

def question_rewriter(state: State):
    logger.info("Rewriting question for improved retrieval")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.7, verbose=False)
    human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    question = human_messages[-1].content
    logger.debug(f"Original question: {question[:100]}...")
    system_prompt = """
    You are a question rewriter for a cardiology guidelines system.
    Your task is to rewrite the user's question to make it more suitable for retrieval.
    The rewritten question should be clear, concise, and focused on the key information needed to retrieve relevant documents.
    Avoid unnecessary details or ambiguity in the question.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question:"
            )
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    response = runnable.invoke({'question': question})
    logger.info(f"Rewritten question: {response[:100]}...")
    return {
        'question': response,
        'transform_query_count': state.transform_query_count + 1
    }

def ground_validator(state: State):
    logger.info(f"Validating generation grounding (attempt {state.generation_count})")
    llm = ChatOllama(model="llama3.2:1b", temperature=0.7, verbose=False)
    human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    question = human_messages[-1].content
    documents = state.documents
    generation = state.response
    generation_count = state.generation_count
    docs_string = "\n\n".join(documents)  # documents are now strings
    
    ground_validator_system_prompt = """
    You are a grader assessing whether an LLM generation is grounded and supported by a set of retrieved facts.
    Given a set of facts and a generation, assess whether the generation is grounded in the facts.
    
    You MUST respond with ONLY 'yes' or 'no' - nothing else.
    - Use 'yes' if the generation is grounded in the facts
    - Use 'no' if the generation is not grounded in the facts
    """
    ground_validator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ground_validator_system_prompt),
            (
                'human',
                'Set of facts: \n{documents}\n\n LLM generation: {generation}',
            )
        ]
    )
    ground_validator_runnable = ground_validator_prompt | llm | StrOutputParser()
    
    answer_grader_system_prompt = """
    You are a grader assessing whether an answer addresses and resolves a question.
    
    You MUST respond with ONLY 'yes' or 'no' - nothing else.
    - Use 'yes' if the answer resolves the question
    - Use 'no' if the answer does not address or resolve the question
    """
    answer_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', answer_grader_system_prompt),
            (
                'human',
                'User question: \n\n {question} \n\n LLM generation: {generation}',
            )
        ]
    )
    answer_grader_runnable = answer_grader_prompt | llm | StrOutputParser()

    if generation_count <= 3:
        try:
            ground_validation_response = ground_validator_runnable.invoke(
                {'documents': docs_string, 'generation': generation}
            )
            ground_validation = ground_validation_response.strip().lower()
            
            if "yes" in ground_validation:
                try:
                    answer_grade_response = answer_grader_runnable.invoke(
                        {'question': question, 'generation': generation}
                    )
                    answer_question = answer_grade_response.strip().lower()
                    
                    if "yes" in answer_question:
                        logger.info("Generation is grounded and addresses the question")
                        return 'grounded_and_addressed_question'
                    else:
                        logger.info("Generation is grounded but does not address the question")
                        return 'grounded_but_not_addressed_question'
                except Exception as e:
                    logger.warning(f"Error in answer grading: {e}, assuming question is addressed")
                    return 'grounded_and_addressed_question'
            else:
                logger.info("Generation is not grounded in the provided documents")
                return 'generation_not_grounded'
        except Exception as e:
            logger.warning(f"Error in ground validation: {e}, assuming generation is grounded")
            return 'grounded_and_addressed_question'
    else:
        logger.info("Maximum generation attempts reached, accepting current generation")
        return 'grounded_and_addressed_question'






