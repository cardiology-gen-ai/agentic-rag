#!/usr/bin/env python3
import os, sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from langgraph.graph import START, END, StateGraph # type: ignore
from langgraph.types import Command # type: ignore

from datetime import datetime

from src.agents.base.graph import BaseAgent
from src.utils.rag_state import RagState
from src.agents.rag.nodes import *
from src.utils.chat import MessageSchema, ConversationSchema, ChatSchema, ChatServiceResponse

vectorstore_dir = os.path.join(project_root, '../data-etl/src')
if os.path.exists(vectorstore_dir):
    sys.path.insert(0, vectorstore_dir)

from vectorstore import load_vectorstore

GENERATION_LIMIT = 3
class RagAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id)

        self.graph = self._create_graph()
        self.compiled_graph = self.graph.compile()
        self.vectorstore = load_vectorstore(
            collection_name="cardio_protocols",
            vectorstore_type="qdrant"
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type = 'similarity',
            search_kwargs = {"k": 5}
        )


    def _route_question(self, state: RagState) -> str:
        question = state.question
        runnable = route_question(self.llm)
        response = runnable.invoke(
            {'question': question}
        )
        response = response.strip().lower()
        return response

    def _conversational_agent(self, state: RagState) -> dict:
        question = state.question
        messages = state.messages
        history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages]) if messages else ""
        runnable = conversational_agent(self.llm)
        response = runnable.invoke(
            {"question": question, "history": history}
        )
        return {"response": response}

    def _retrieve(self, state: RagState) -> dict:
        question = state.question
        documents = self.retriever.invoke(question)
        return {'documents': documents}

    def _retrieval_grader(self, state: RagState) -> dict:
        question = state.question
        documents = state.documents
        runnable = retrieval_grader(self.llm)
        filtered_docs = []
        for idx, d in enumerate(documents):
            score = runnable.invoke(
                {"question": question, "document": d["content"], "document_filename": d["file"]}
            )
            grade = score.binary_score
            if grade == "yes":
                filtered_docs.append(d)
            else:
                continue
        if len(filtered_docs) == 0:
            return Command(
                update = {
                    'documents': filtered_docs,
                    'route': 'all_docs_not_relevant'
                },
                goto = "transform_question"
            )
        else:
            return Command(
                update = {
                    'documents': filtered_docs,
                    'route': 'at_least_one_doc_relevant'
                },
                goto = "generate"
            )

    def _generate(self, state: RagState):
        question = state.question
        documents = state.documents
        retrieved_docs_as_string = [f"Filename: {doc['file']}\nContent: {doc['content']}" for doc in documents]
        context = "\n\n".join([string for string in retrieved_docs_as_string])
        runnable = generate(self.llm)
        response = runnable.invoke(
            {
                'documents': context,
                'question': question,
            }
        )
        return {
            'generation': response,
            'generation_count': state.generation_count + 1,
        }

    def _question_rewriter(self, state: RagState) -> dict:
        question = state.question
        runnable = question_rewriter(self.llm)
        response = runnable.invoke({'question': question})
        return {
            'question': response,
            'transform_query_count': state.transform_query_count + 1
        }
    
    def _ground_validator(self, state: RagState) -> str:
        documents = state.documents
        generation = state.generation
        generation_count = state.generation_count
        question = state.question
        docs_string = "\n\n".join([doc["content"] for doc in documents])
        ground_validator_runnable = ground_validator(self.llm)
        answer_grader_runnable = answer_grader(self.llm)
        if generation_count <= GENERATION_LIMIT:
            ground_validation = ground_validator_runnable.invoke(
                {'documents': docs_string, 'generation': generation}
            )
            ground_validation = ground_validation.binary_score
            if ground_validation == 'yes':
                answer_grade = answer_grader_runnable.invoke(
                    {'question': question, 'generation': generation}
                )
                answer_question = answer_grade.binary_score
                if answer_question == 'yes':
                    return 'grounded_and_addressed_question'
                else:
                    return 'grounded_but_not_addressed_question'
            else:
                return 'generation_not_grounded'
    
    def _create_graph(self):
        graph = StateGraph(RagState)

        graph.add_node('conversational_agent', self._conversational_agent)
        graph.add_node('retrieve', self._retrieve)
        graph.add_node('retrieval_grader', self._retrieval_grader)
        graph.add_node('transform_question', self._question_rewriter)
        graph.add_node('generate', self._generate)

        graph.add_conditional_edges(
            START,
            self._route_question,
            {
                "conversational": "conversational_agent",
                "document_based": "retrieve",
            }
        )
        graph.add_conditional_edges(
            "retrieve",
            self._retrieval_grader,
            {
                "all_docs_not_relevant": "transform_question",
                "at_least_one_doc_relevant": "generate",
            },
        )
        graph.add_conditional_edges(
            "generate",
            self._ground_validator,
            {
                "grounded_and_addressed_question": END,
                "generation_not_grounded": "generate",
                "grounded_but_not_addressed_question": "transform_question",
            }
        )
        graph.add_edge("transform_question", "retrieve")
        graph.add_edge("conversational_agent", END)

        return graph
    
    def answer(self, request_body: ChatSchema) -> ChatServiceResponse:
        state = {
            'messages': self._convert_conversation_to_messages(
                request_body.conversation 
            )
        }
        question = request_body.conversation.question.content
        user = request_body.user 
        _id = request_body.conversation.id

        is_faulted = False 
        result = self.compiled_graph.invoke(
            input = {
                'question': question,
                'user_name': user,
                'messages': state['messages'],
                'transform_query_count': 0,
                'generation_count': 0,
                'contextual_question': '',
                'response': '',
                'generation': '',
                'documents': [],
                'examples': [],
                'document_request': '',
            },
            config = {
                # "callbacks": callbacks,
                # "configurable": {"thread_id": request_body.conversation.id},
            },
        )
        content = result.get('generation') or result.get('response', '')
        return ChatServiceResponse(
            role = 'assistant',
            content = content,
            is_faulted = is_faulted,
        )

# Create the agent instance for LangGraph
agent = RagAgent(agent_id="bt-hr").compiled_graph
    
if __name__ == "__main__":
    agent = RagAgent(agent_id = "bt-hr")
    question_list = ["Hi, how is valvular heart disease diagnosed?"]
    history = []
    for question in question_list:
        user_message = MessageSchema(
            role="user",
            content=question,
            datetime = datetime.strptime("2021-10-01T10:00:02", "%Y-%m-%dT%H:%M:%S"),
        )
        request = ChatSchema(
            user = "christian",
            conversation = ConversationSchema(
                id = "1",
                agentId = "bt-hr",
                history = history,
                question = user_message,
            )
        )
        final_response = agent.answer(request_body = request)
        print(f"Final response: {final_response.content}")




