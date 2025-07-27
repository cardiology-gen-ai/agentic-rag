#!/usr/bin/env python3 
"""
Utility functions that act as nodes in the agent.
"""

from datetime import datetime 
from pydantic import BaseModel # type: ignore

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str

class GradeGrounding(BaseModel):
    """Binary score for grounding check on generation."""
    binary_score: str

class GradeAnswer(BaseModel):
    """Binary score for answer addressing question."""
    binary_score: str

def route_question(llm):
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
    route_question = prompt | llm | StrOutputParser()
    return route_question

def conversational_agent(llm):
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
    conversational_agent = prompt | llm | StrOutputParser()
    return conversational_agent

def retrieval_grader(llm):
    parser = PydanticOutputParser(pydantic_object=GradeDocuments)
    system_prompt = """
    You are a grader assessing relevance of a retrieved document to a user question.\n
    If the document or the document filename contain keyword(s) or semantic meaning related to the question, grade it as relevant.\n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Retrieved document filename: {document_filename} \n\n Retrieved document: {document} \n\n User question: {question}"
            )
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    retrieval_grader = prompt | llm | parser
    return retrieval_grader

def generate(llm):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_prompt = """
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
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

def question_rewriter(llm):
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
    question_rewriter = prompt | llm | StrOutputParser()
    return question_rewriter

def ground_validator(llm):
    parser = PydanticOutputParser(pydantic_object=GradeGrounding)
    system_prompt = """
    You are a grader assessing whether an LLM generation is grounded and supported by a set of retrieved facts.\n
    Given a set of facts and a generation, assess whether the generation is grounded in the facts.\n
    Provide a binary score: 'yes' if the generation is grounded in the facts, 'no' otherwise.\n
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                'human',
                'Set of facts: \n{documents}\n\n LLM generation: {generation}',
            )
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    ground_validator = prompt | llm | parser
    return ground_validator

def answer_grader(llm):
    parser = PydanticOutputParser(pydantic_object=GradeAnswer)
    system_prompt = """
    You are a grader assessing whether an answer addresses and resolves a question\n
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.\n
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            (
                'human',
                'User question: \n\n {question} \n\n LLM generation: {generation}',
            )
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    answer_grader = prompt | llm | parser
    return answer_grader






