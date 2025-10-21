from datetime import datetime
import re

from langchain_core.runnables import RunnableBinding, RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agent import output


def _strip_think(s: str) -> str:
    # useful for parsing Qwens' model output
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL|re.IGNORECASE)
    s = s.strip().strip("").strip()
    return s


def _get_final(s: str) -> str:
    match = re.search(r"assistantfinal\s*(.*)$", s, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return content
    return s


def detect_language(llm: Runnable) -> Runnable:
    """Build a runnable that detects the language of the text.

    The chain formats instructions, prompts the model, parses the raw string,
    strips ``<think>`` traces, and validates the final JSON into :class:`~src.agent.output.DetectLanguage`.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model to execute the detection.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agent.output.DetectLanguage` instance.
    """
    parser = JsonOutputParser(pydantic_object=output.DetectLanguage)
    format_instructions = "Return ONLY a valid JSON object with exactly one key 'language' whose value is either 'it' or 'en'."
    system_prompt = f"""
    You are a language detector. Decide if the input is Italian or English.  
    Return "it" if the input is in Italian, "en" if the input is in English.\n
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Text:\n{text}"),
        ]
    )
    to_model = RunnableLambda(lambda d: output.DetectLanguage.model_validate(d))
    language_detector = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final) | parser | to_model
    return language_detector


def contextualize_question(llm: Runnable, context_prompt: str) -> Runnable:
    """Build a runnable that minimally adds context to the last user question.

    The chain returns the original question verbatim unless context is truly needed to make it understandable without the prior chat history.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model to perform contextualization.
    context_prompt : str
        Additional system guidance appended to the base rules.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a context-adjusted question string.
    """
    system_prompt = """
    Given the provided history, add context to the last human message ONLY if needed to make it understandable without the chat history.

    IMPORTANT RULES:
    1. If there is NO history or the history is empty, return the question EXACTLY as provided - do not modify it at all
    2. If the question is a greeting, introduction request, or general conversational message, return it EXACTLY as provided
    3. Only add context when the question refers to something mentioned earlier in the conversation
    4. Always preserve the user's original intent and meaning
    5. Do NOT rewrite questions to make them "more suitable for retrieval"
    6. Do NOT answer the question, just add context if truly needed

    Examples:
    - "Hello" → "Hello" (return exactly as is)
    - "Please introduce yourself" → "Please introduce yourself" (return exactly as is)
    - "What about that condition?" (with history about diabetes) → "What about diabetes?" (add context)
    """
    system_prompt += context_prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", " Language: {language} \nHistory: \n {history} \nQuestion: {question}"),
        ]
    )
    contextualizer = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) |  RunnableLambda(_get_final)
    return contextualizer


def router(llm: Runnable, index_description: str, example_prompt: str) -> Runnable:
    """Build a runnable that routes a query to the ``conversational`` or ``document-based`` branch.

    The chain instructs the model, parses JSON, strips ``<think>`` traces, and validates output into :class:`~src.agent.output.RouteQuery`.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model to perform routing.
    index_description : str
        Description of the vectorstore/index available to the agent.
    example_prompt : str
        Few-shot examples guiding the routing behavior.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agent.output.RouteQuery` instance.
    """
    parser = JsonOutputParser(pydantic_object=output.RouteQuery)
    format_instructions = "Return ONLY a valid JSON object with exactly one key 'branch' whose value is either 'conversational' or 'document_based'."
    system_prompt = f"""
    You are an expert at routing a human message to a document-based branch or conversational branch. 
    This is the vectorstore description: {index_description}\n
    Determine whether the user's question is a conversational inquiry, meaning it is general, casual, or social. 
    This can include, but it is not limited to greetings (e.g., 'Hello'), small talk 
    (e.g., 'How are you?', 'What information can you give me?, 'What documentation can you provide me?'), 
    personal opinions, and polite expressions of gratitude (e.g., 'Thank you').
    Return 'conversational' if the user question is a conversational inquiry.
    Return 'document_based' if the user question is NOT a conversational inquiry
    Follow these examples to decide if the question should be routed to the vectorstore or conversational branch:
    {example_prompt}.\n
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: {question}"),
        ]
    )
    to_model = RunnableLambda(lambda d: output.RouteQuery.model_validate(d))
    question_router = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final) | parser | to_model
    return question_router


def conversational_agent(llm: Runnable, agent_prompt: str) -> Runnable:
    """Build a general conversational agent runnable.

    Produces clear, concise answers in the requested language, using the given agent prompt and the current timestamp.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model to generate replies.
    agent_prompt : str
        System guidance appended to the default behavior.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline that yields an assistant message string.
    """
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_prompt = f""
    system_prompt += agent_prompt
    system_prompt = f"""
    Today is {current_datetime}.\n
    {agent_prompt}\n
    Your task is to provide clear, accurate, and helpful responses to user questions and requests. 
    Keep your responses natural and conversational—avoid being overly formal or robotic. 
    Be concise but complete—give enough information to be useful without unnecessary details. 
    If you don't know something, say so clearly rather than guessing. 
    If a question is unclear, ask for clarification. 
    Stay focused on being helpful and direct in your responses.
    Use the provided language in your answer.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}, \nChat history: {history} \nLanguage: {language}"),
        ]
    )
    conversational_agent_runnable = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final)
    return conversational_agent_runnable


def retrieval_grader(llm: Runnable) -> Runnable:
    """Build a runnable that grades document relevance to a question.

    Returns a binary score via :class:`~src.agent.output.GradeDocuments` after JSON parsing and validation.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model used for grading.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agent.output.GradeDocuments` instance.
    """
    parser = JsonOutputParser(pydantic_object=output.GradeDocuments)
    format_instructions = "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."
    system_prompt = f"""
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document or the document filename contain keyword(s) or semantic meaning related to the question, grade it as relevant.
    - Use 'yes' if the document is relevant to the question
    - Use 'no' if the document is not relevant to the question.\n
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Retrieved document filename: {document_filename} \nRetrieved document: {document} \nUser question: {question}")
        ]
    )
    to_model = RunnableLambda(lambda d: output.GradeDocuments.model_validate(d))
    grader = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final) | parser | to_model
    return grader


def document_request_detector(llm: Runnable) -> Runnable:
    """Build a runnable that detects whether a user is explicitly requesting a document.

    Produces a binary score via :class:`~src.agent.output.DocumentRequest` after JSON parsing and validation.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model used for detection.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agent.output.DocumentRequest` instance.
    """
    parser = JsonOutputParser(pydantic_object=output.DocumentRequest)
    format_instructions = "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."
    system_prompt = f"""
    You are a classifier that determines whether a user's question is a request for a document.\n
    Respond with 'yes' if the user is explicitly asking to receive, access, or view a document.\n
    Respond with 'no' if the user is asking about the content, meaning, or purpose of a document, 
    without requesting the document itself.\n
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: {question}"),
        ]
    )
    to_model = RunnableLambda(lambda d: output.DocumentRequest.model_validate(d))
    document_detector = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final) | parser | to_model
    return document_detector


def generate_document_response(llm: Runnable) -> Runnable:
    """Build a runnable that crafts a polite response when the user requests documents.

    The response avoids mentioning specific filenames.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline yielding a response string.
    """
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
    document_response = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final)
    return document_response

def generate(llm: Runnable) -> Runnable:
    """Build a runnable that answers using retrieved context.

    If the answer is unknown, the agent should state it clearly. The response uses the question's language.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline yielding a response string.
    """
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
    generator = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) |  RunnableLambda(_get_final)
    return generator


def question_rewriter(llm: Runnable) -> Runnable:
    """Build a runnable that rewrites a query for better vectorstore retrieval.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline yielding an improved query string.
    """
    system_prompt = """
    You a question re-writer that converts an input question to a better version that is optimized \n
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.\n
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    rewriter = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final)
    return rewriter

def generate_default_response(llm: Runnable) -> Runnable:
    """Build a runnable that returns a polite fallback response.

    The message states that the system lacks sufficient knowledge and suggests
    trying another question, in the user's language.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline yielding a default response.
    """
    # TODO: maybe formulate a default "default response" without letting the model formulate it itself
    system_prompt = """
    You are an assistant for question-answering tasks.\n
    Politely inform the user that your knowledge is not sufficient to answer the question and suggest trying another question.\n
    Use the language of the question in your response.\n
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question} \nLanguage: {language}\n"),
        ]
    )
    default_response = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final)
    return default_response


def ground_validator(llm: Runnable):
    """Build a runnable that checks whether a generation is grounded in retrieved facts.

    Produces a binary score via :class:`~src.agent.output.GradeGrounding` after JSON parsing and validation.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model used for grading.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agent.output.GradeGrounding` instance.
    """
    parser = JsonOutputParser(pydantic_object=output.GradeGrounding)
    format_instructions = "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."
    system_prompt = f"""
    You are a grader assessing whether an LLM generation is grounded and supported by a set of retrieved facts.\n
    Given a set of facts and a generation, assess whether the generation is grounded in the facts.\n
    Give a binary score 'yes' or 'no'. 'Yes' means that the generation is grounded and supported by the facts.\n
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Set of facts: :\n{documents}\n\n LLM generation: {generation}"),
        ]
    )
    to_model = RunnableLambda(lambda d: output.GradeGrounding.model_validate(d))
    groundedness_validator = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final) | parser | to_model
    return groundedness_validator


def answer_grader(llm: Runnable):
    """Build a runnable that checks whether an answer resolves a question.

    Produces a binary score via :class:`~src.agent.output.GradeAnswer` after JSON parsing and validation.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model used for grading.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agent.output.GradeAnswer`.
    """
    parser = JsonOutputParser(pydantic_object=output.GradeAnswer)
    format_instructions = "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."
    system_prompt = f"""
    You are a grader assessing whether an answer addresses and resolves a question\n
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.\n
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "Return only valid JSON."),
            ("human", "Question: {question} \n\n Answer: {generation}"),
        ]
    )
    to_model = RunnableLambda(lambda d: output.GradeAnswer.model_validate(d))
    grader = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final) | parser | to_model
    return grader


def error_handler_node(llm: Runnable, language: str):
    """Build a runnable that turns exceptions into friendly user-facing messages.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model to generate the error message.
    language : :class:`str`
        Target language (must be one of the allowed languages provided to the prompt).

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline yielding a concise, friendly error message.
    """
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
    error_message_generator = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think)
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



