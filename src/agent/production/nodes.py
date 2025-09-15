from datetime import datetime

from langchain_core.runnables import RunnableBinding
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import List, Union

from src.agent.production import output


def contextualize_question(llm: RunnableBinding, context_prompt: str):
    system_prompt = """
    Given the provided history, enrich the last human message with the information from the history 
    such that it can be understood without the chat history. 
    Users may ask about specific topics or ask to retrieve a particular document. 
    In both cases, the reformulated question must preserve the user's original intent.
    Your task is to rewrite the user's question to make it more suitable for retrieval.
    The rewritten question should be clear, concise, and focused on the key information needed to retrieve relevant documents.
    Avoid unnecessary details or ambiguity in the question.
    Do NOT answer the question, just reformulate it if needed, otherwise return it as is. 
    """
    system_prompt += context_prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", " Language: {language} \nHistory: \n {history} \nQuestion: {question}"),
        ]
    )
    contextualizer = prompt | llm | StrOutputParser()
    return contextualizer


def router(llm: RunnableBinding, index_description: str, example_prompt: str):
    structured_llm = llm.with_structured_output(output.RouteQuery, method="function_calling")
    system_prompt = f"""
    You are an expert at routing a human message to a document-based branch or conversational branch. 
    This is the vectorstore description: {index_description}\n
    Determine whether the user's question is a conversational inquiry, meaning it is general, casual, or social. 
    This can include, but it is not limited to greetings (e.g., 'Hello'), small talk 
    (e.g., 'How are you?', 'What information can you give me?, 'What documentation can you provide me?'), 
    personal opinions, and polite expressions of gratitude (e.g., 'Thank you').
    If the user question is a conversational inquiry, respond with exactly: conversational
    If the user question is NOT a conversational inquiry, respond with exactly: document_based
    Follow these examples to decide if the question should be routed to the vectorstore or conversational branch:
    {example_prompt}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: {question}"),
        ]
    )
    question_router = prompt | structured_llm
    return question_router


def conversational_agent(llm: RunnableBinding, agent_prompt: str):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_prompt = f"Today is {current_datetime}."
    system_prompt += agent_prompt
    system_prompt += (
        "Your task is to provide clear, accurate, and helpful responses to user questions and requests. "
        "Keep your responses natural and conversational—avoid being overly formal or robotic. "
        "Be concise but complete—give enough information to be useful without unnecessary details. "
        "If you don't know something, say so clearly rather than guessing. "
        "If a question is unclear, ask for clarification. "
        "Stay focused on being helpful and direct in your responses."
        "Use the provided language in your answer."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}, \nChat history: {history} \nLanguage: {language}"),
        ]
    )
    conversational_agent_runnable = prompt | llm | StrOutputParser()
    return conversational_agent_runnable


def retrieval_grader(llm: RunnableBinding):
    structured_llm = llm.with_structured_output(output.GradeDocuments, method="function_calling")
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
            ("human", "Retrieved document filename: {document_filename} \nRetrieved document: {document} \nUser question: {question}")
        ]
    )
    grader = prompt | structured_llm
    return grader


def document_request_detector(llm: RunnableBinding):
    structured_llm = llm.with_structured_output(output.DocumentRequest, method="function_calling")
    system_prompt = """
    You are a classifier that determines whether a user's question is a request for a document.\n
    Respond with 'yes' if the user is explicitly asking to receive, access, or view a document.\n
    Respond with 'no' if the user is asking about the content, meaning, or purpose of a document, 
    without requesting the document itself.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: {question}"),
        ]
    )
    document_detector = prompt | structured_llm
    return document_detector


def generate_document_response(llm: RunnableBinding):
    # TODO: maybe formulate a default "document response" without letting the model formulate it itself
    system_prompt = """
    You are an assistant for question-answering tasks.\n
    If the user asks for a document (or multiple documents), generate a polite and clear response 
    indicating that you are providing the requested document.\n
    Do NOT mention file names directly.\n
    Instead, refer to them generally (e.g., 'the documents').\n
    Use the same language as the user's question.\n
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question} \nAvailable documents: {documents} \nLanguage: {language}"),
        ]
    )
    document_response = prompt | llm | StrOutputParser()
    return document_response

def generate(llm: RunnableBinding):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_prompt = f"""
    Today is {current_datetime}. \n
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question\n
    If you don't know the answer, just say that you don't know. \nUse the language of the question in your answer.\n
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Retrieved information: \n{documents} \nQuestion: \n{question} \nLanguage:{language} \nAnswer:")
        ]
    )
    generator = prompt | llm | StrOutputParser()
    return generator


def question_rewriter(llm: RunnableBinding):
    system_prompt = "You a question re-writer that converts an input question to a better version that is optimized \n"
    system_prompt += "for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    rewriter = prompt | llm | StrOutputParser()
    return rewriter

def generate_default_response(llm: RunnableBinding):
    # TODO: maybe formulate a default "default response" without letting the model formulate it itself
    system_prompt = "You are an assistant for question-answering tasks.\n"
    system_prompt += "Politely inform the user that your knowledge is not sufficient to answer the question and suggest trying another question.\n"
    system_prompt += "Use the language of the question in your response.\n"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question} \nLanguage: {language}\n"),
        ]
    )
    default_response = prompt | llm | StrOutputParser()
    return default_response


def ground_validator(llm: RunnableBinding):
    structured_llm = llm.with_structured_output(output.GradeGrounding, method="function_calling")
    system_prompt = (
        "You are a grader assessing whether an LLM generation is grounded and supported by a set of retrieved facts.\n"
        "Given a set of facts and a generation, assess whether the generation is grounded in the facts.\n"
        "Provide a binary score: 'yes' if the generation is grounded in the facts, 'no' otherwise.\n"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Set of facts: :\n{documents}\n\n LLM generation: {generation}"),
        ]
    )
    groundedness_validator = prompt | structured_llm
    return groundedness_validator


def answer_grader(llm: RunnableBinding):
    structured_llm = llm.with_structured_output(output.GradeAnswer, method="function_calling")
    system_prompt = (
        "You are a grader assessing whether an answer addresses and resolves a question\n"
        "Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.\n"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question} \n\n Answer: {generation}"),
        ]
    )
    grader = prompt | structured_llm
    return grader


def error_handler_node(llm: RunnableBinding, language: str):
    system_prompt = f"""
    You are an error message generator.
    Given an exception return a friendly and helpful error message to the user.
    Do not include technical details unless useful for the user.
    Please use the same language of the exception, but it must be one of the following: {language}.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Exception: {exception}"),
        ]
    )
    error_message_generator = prompt | llm | StrOutputParser()
    return error_message_generator


# TODO: still need to decide how to appropriately handling long-term memory
# def extract_human_to_ai_sequence(messages: List[Union[HumanMessage, AIMessage, ToolMessage]]) -> List:
#     # Step 1: Find the index of the last HumanMessage
#     start_idx = None
#     for i in reversed(range(len(messages))):
#         if isinstance(messages[i], HumanMessage):
#             start_idx = i
#             break
#     if start_idx is None:
#         return []
#     # Step 2: From that human message, collect all messages up to and including the next complete AI response
#     result = []
#     ai_message_count = 0
#     for msg in messages[start_idx:]:
#         result.append(msg)
#         if isinstance(msg, AIMessage) and msg.content.strip():
#             ai_message_count += 1
#             # Stop after a full (non-empty) AI response
#             break
#     return result

# def store_memory(state: State, config: RunnableConfig, store: BaseStore):
#     user_id = config['configurable']['user_id']
#     namespace = (user_id, 'memories') # shared across threads
#     memory_id = str(uuid.uuid4())
#     memory = extract_human_to_ai_sequence(state["messages"])
#     store.put(namespace, memory_id, {'memory': memory})
#     return {'memory': memory}
#
# def search_memory(question, config: RunnableConfig, store: BaseStore):
#     user_id = config['configurable']['user_id']
#     namespace = (user_id, 'memories') # shared across threads
#     memories = store.search(
#         namespace,
#         query = question,
#         limit=3
#     )
#     return memories



