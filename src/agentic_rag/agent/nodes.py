from datetime import datetime
import re
from typing import List, Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agentic_rag.agent import output

# prompt structure: system prompt (role or persona + goal + guardrails + answer structure),
# few-shot examples (if needed), history, documents, user query 


def _strip_think(s: str) -> str:
    # useful for parsing Qwens' model output
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL|re.IGNORECASE)
    s = s.strip().strip("").strip()
    return s


def _get_final(s: str) -> str:
    content = s
    # useful for parsing gpt-oss output
    match_final = re.search(r"assistantfinal\s*(.*)$", content, re.DOTALL)
    if match_final:
        content = match_final.group(1).strip()
    match_json = re.search(r"[Jj][Ss][Oo][Nn]\s*(\{.*)$", content, re.DOTALL)
    if match_json:
        content = match_json.group(1).strip()
    return content


def get_llm_with_structured_output(llm: BaseChatModel, output_schema: BaseModel | Dict | Any, prompt: ChatPromptTemplate):
    llm_with_structured_output = llm.with_structured_output(output_schema, include_raw=True)
    language_detector = prompt | llm_with_structured_output
    return language_detector


def get_chain_with_unstructured_output(llm: BaseChatModel, prompt: ChatPromptTemplate):
    runnable = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) |  RunnableLambda(_get_final)
    return runnable


def get_chain_with_structured_output(llm: BaseChatModel, output_schema: BaseModel | Dict | Any, prompt: ChatPromptTemplate):
    parser = JsonOutputParser(pydantic_object=output_schema)
    to_model = RunnableLambda(lambda d: output_schema.model_validate(d))
    runnable = get_chain_with_unstructured_output(llm, prompt) | parser | to_model
    return runnable


def detect_language(llm: BaseChatModel, structured_output: bool = True) -> Runnable:
    """Build a runnable that detects the language of the text.

    The chain formats instructions, prompts the model, parses the raw string,
    strips ``<think>`` traces, and validates the final JSON into :class:`~src.agentic_rag.agent.output.DetectLanguage`.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model to execute the detection.
    structured_output: bool
        Whether to return a structured output. Default is ``True``.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agentic_rag.agent.output.DetectLanguage` instance.
    """
    format_instructions = "" if structured_output else \
        "Return ONLY a valid JSON object with exactly one key 'language' whose value is either 'it' or 'en'."
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
    return get_llm_with_structured_output(llm, output.DetectLanguage, prompt) if structured_output else (
        get_chain_with_structured_output(llm, output.DetectLanguage, prompt))


def contextualize_question(llm: BaseChatModel, context_prompt: str, messages: List[AnyMessage]) -> Runnable:
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
    # Thought: Given the chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history.
    #
    # Action:I will NOT answer the latest user question. I will reformulate the latest user question as a new question using the chat history so it can be a standalone question which can be understood without the chat history. If the question does not need any reformulation, I will return the question as is in the original format. Formulate the new question in a standalone question format without chat history.
    #
    # Observation: is the reformulated question clear and concise and in a question format?
    #
    # Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. Your standalone question will be used to query a vector database for RAG.
    system_prompt = """
    You are an intelligent agent tasked with understanding ambiguous or unclear user questions in the context of cardiology protocols.
    Your goal is to analyze the user question, alongside the given chat history, 
    to formulate a clear, standalone question that retains the original intent 
    but can be understood independently without requiring prior context or the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    # IMPORTANT RULES:
    # 1. If there is NO history or the history is empty, return the question EXACTLY as provided - do not modify it at all
    # 2. If the question is a greeting, introduction request, or general conversational message, return it EXACTLY as provided
    # 3. Only add context when the question refers to something mentioned earlier in the conversation
    # 4. Always preserve the user's original intent and meaning
    # 5. Do NOT rewrite questions to make them "more suitable for retrieval"
    # 6. Do NOT answer the question, just add context if truly needed
    # """
    system_prompt += context_prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n Question must be in {language} language."),
            ("human", "User Question: {question}\n Chat History: \n"),
        ] + messages
    )
    return get_chain_with_unstructured_output(llm, prompt)


def router(llm: BaseChatModel, index_description: str, example_prompt: str, structured_output: bool = True) -> Runnable:
    """Build a runnable that routes a query to the ``conversational`` or ``document-based`` branch.

    The chain instructs the model, parses JSON, strips ``<think>`` traces, and validates output into :class:`~src.agentic_rag.agent.output.RouteQuery`.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model to perform routing.
    index_description : str
        Description of the vectorstore/index available to the agent.
    example_prompt : str
        Few-shot examples guiding the routing behavior.
    structured_output: bool
        Whether to return a structured output. Default is ``True``.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agentic_rag.agent.output.RouteQuery` instance.
    """
    format_instructions = "" if structured_output else \
        "Return ONLY a valid JSON object with exactly one key 'branch' whose value is either 'conversational' or 'document_based'."
    system_prompt = f"""
    You are an expert at routing a human message to a document_based branch or conversational branch. 
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
    return get_llm_with_structured_output(llm, output.RouteQuery, prompt) if structured_output else (
        get_chain_with_structured_output(llm, output.RouteQuery, prompt))


def conversational_agent(llm: BaseChatModel, agent_prompt: str) -> Runnable:
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
    return get_chain_with_unstructured_output(llm, prompt)


def retrieval_grader(llm: BaseChatModel, structured_output: bool = True) -> Runnable:
    """Build a runnable that grades document relevance to a question.

    Returns a binary score via :class:`~src.agentic_rag.agent.output.GradeDocuments` after JSON parsing and validation.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model used for grading.
    structured_output: bool
        Whether to return a structured output. Default is ``True``.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agentic_rag.agent.output.GradeDocuments` instance.
    """
    format_instructions = "" if structured_output else \
        "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."
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
    return get_llm_with_structured_output(llm, output.GradeDocuments, prompt) if structured_output else (
            get_chain_with_structured_output(llm, output.GradeDocuments, prompt))


def document_request_detector(llm: BaseChatModel, structured_output: bool = True) -> Runnable:
    """Build a runnable that detects whether a user is explicitly requesting a document.

    Produces a binary score via :class:`~src.agentic_rag.agent.output.DocumentRequest` after JSON parsing and validation.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model used for detection.
    structured_output: bool
        Whether to return a structured output. Default is ``True``.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agentic_rag.agent.output.DocumentRequest` instance.
    """
    format_instructions = "" if structured_output else \
        "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."
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
    return get_llm_with_structured_output(llm ,output.DocumentRequest, prompt) if structured_output else (
        get_chain_with_structured_output(llm ,output.DocumentRequest, prompt))


def generate_document_response(llm: BaseChatModel) -> Runnable:
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
    return get_chain_with_unstructured_output(llm, prompt)

def generate(llm: BaseChatModel) -> Runnable:
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
    # Your primary mission is to answer questions based on provided context or chat history.
    # Carefully analyze the given context and ensure your response is concise and directly addresses the question without any additional narration.
    #
    # ###
    #
    # Your final answer should be written concisely (but include important numerical values, technical terms, jargon, and names)
    #
    # # Steps
    #
    # 1. Carefully read and understand the context provided.
    # 2. Identify the key information related to the question within the context.
    # 3. Formulate a concise answer based on the relevant information.
    # 4. Ensure your final answer directly addresses the question.
    # TODO: check if it is a good idea to keep current_datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_prompt = f"""
    Today is {current_datetime}. \n
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question\n
    If you don't know the answer, just say that you don't know. \nUse the language of the question in your answer.\n
    """
    # TODO: maybe also history is needed to correctly answer a question
    # use the following context to answer the query
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Retrieved information: \n{documents} \nQuestion: \n{question} \nLanguage:{language} \nAnswer:")
        ]
    )
    return get_chain_with_unstructured_output(llm, prompt)


# TODO: here implement query rewriting strategies
def question_rewriter(llm: BaseChatModel) -> Runnable:
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
    return get_chain_with_unstructured_output(llm, prompt)


def generate_default_response(llm: BaseChatModel) -> Runnable:
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
    return get_chain_with_unstructured_output(llm, prompt)


# TODO: check prompts of evaluation libraries to correct these prompts
def ground_validator(llm: BaseChatModel, structured_output: bool = True):
    """Build a runnable that checks whether a generation is grounded in retrieved facts.

    Produces a binary score via :class:`~src.agentic_rag.agent.output.GradeGrounding` after JSON parsing and validation.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model used for grading.
    structured_output: bool
        Whether to return a structured output. Default is ``True``.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agentic_rag.agent.output.GradeGrounding` instance.
    """
    format_instructions = "" if structured_output else \
        "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."
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
    return get_llm_with_structured_output(llm, output.GradeGrounding, prompt) if structured_output \
            else get_chain_with_structured_output(llm, output.GradeGrounding, prompt)


def answer_grader(llm: BaseChatModel, structured_output: bool = True):
    """Build a runnable that checks whether an answer resolves a question.

    Produces a binary score via :class:`~src.agentic_rag.agent.output.GradeAnswer` after JSON parsing and validation.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model used for grading.
    structured_output: bool
        Whether to return a structured output. Default is ``True``.

    Returns
    -------
    :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Runnable pipeline producing a validated :class:`~src.agentic_rag.agent.output.GradeAnswer`.
    """
    format_instructions = "" if structured_output else \
        "Return ONLY a valid JSON object with exactly one key 'binary_score' whose value is either 'yes' or 'no'."
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
    return get_llm_with_structured_output(llm, output.GradeAnswer, prompt) if structured_output else (
        get_chain_with_structured_output(llm, output.GradeAnswer, prompt))


def error_handler_node(llm: BaseChatModel, language: List[str]):
    """Build a runnable that turns exceptions into friendly user-facing messages.

    Parameters
    ----------
    llm : :langchain_core:`Runnable <runnables/langchain_core.runnables.base.Runnable.html>`
        Temperature-bound chat model to generate the error message.
    language : :class:`list` of :class:`str`
        Possible target languages (must be one of the allowed languages provided to the prompt).

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
    return get_chain_with_unstructured_output(llm, prompt)


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



