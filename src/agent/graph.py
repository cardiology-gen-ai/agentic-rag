#!/usr/bin/env python3
import os, sys
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from langgraph.graph import START, END, StateGraph # type: ignore
from langchain_core.messages import HumanMessage, AIMessage # type: ignore
from langgraph.checkpoint.memory import InMemorySaver # type: ignore

from datetime import datetime

from src.utils.state import State
from src.agent import nodes
from src.utils.chat import MessageSchema, ConversationSchema, ChatSchema, ChatServiceResponse

vectorstore_dir = os.path.join(project_root, '../data-etl/src')
if os.path.exists(vectorstore_dir):
    sys.path.insert(0, vectorstore_dir)

from vectorstore import load_vectorstore

class Agent():
    def __init__(self, agent_id: str, log_level: str = "INFO"):
        self.agent_id = agent_id
        self._setup_logging(log_level)
        self.logger = logging.getLogger(f"{__name__}.Agent.{agent_id}")
        self.logger.info(f"Initializing agent with ID: {agent_id}")
        
        self.graph = self._create_graph()
        self.checkpointer = InMemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer = self.checkpointer)
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
        config = {"configurable": {"thread_id": self.agent_id}}
        response = self.compiled_graph.invoke(
            {'messages': [HumanMessage(content=question)]}, 
            config=config
        )
        answer = response.get('generation') or response.get('response', '')
        self.logger.info(f"Generated answer: {answer[:100]}...")
        return answer
        

# Create the agent instance for LangGraph
agent = Agent(agent_id="test").compiled_graph
    
if __name__ == "__main__":
    agent = Agent(agent_id = "test")
    question = "Hi, how is valvular heart disease diagnosed?"
    final_response = agent.answer(question)
    print(f"Final response: {final_response}")




