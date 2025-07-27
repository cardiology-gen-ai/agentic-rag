#!/usr/bin/env python3
from langgraph.graph import START, END, StateGraph # type: ignore

from datetime import datetime

from src.agents.base.graph import BaseAgent
from src.utils.rag_state import RagState
from src.agents.rag.nodes import *
from src.utils.chat import MessageSchema, ConversationSchema, ChatSchema

class RagAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id)

        self.graph = self._create_graph()
        self.compiled_graph = self.graph.compile()

    def _contextualize_question(self):
        pass

    def _route_question(self):
        pass

    def _conversational_agent(self, state: RagState) -> dict:
        system_prompt = """
        # create system prompt
        """
        question = state['question']
        runnable = conversational_agent(self.llm, system_prompt)
        response = runnable.invoke(
            {"question": question}
        )
        return {"response": response}

    def _retrieve(self):
        pass 

    def _retrieval_grader(self):
        pass 

    def _question_rewriter(self):
        pass 

    def _generate(self):
        pass
    
    def _decide_to_generate(self, state: RagState) -> str:
        # Decision logic for retrieval grader
        return "at_least_one_doc_relevant"  # placeholder
    
    def _ground_validator(self, state: RagState) -> str:
        # Decision logic for ground validation
        return "grounded_and_addresses_question"  # placeholder
    
    def _create_graph(self):
        graph = StateGraph(RagState)

        graph.add_node('contextualize_question', self._contextualize_question)
        graph.add_node('conversational_agent', self._conversational_agent)
        graph.add_node('retrieve', self._retrieve)
        graph.add_node('retrieval_grader', self._retrieval_grader)
        graph.add_node('transform_question', self._question_rewriter)
        graph.add_node('generate', self._generate)

        graph.add_edge(START, 'contextualize_question')
        graph.add_conditional_edges(
            "contextualize_question",
            self._route_question,
            {
                "conversational": "conversational_agent",
                "document_based": "retrieve",
            }
        )
        graph.add_edge("retrieve", "retrieval_grader")
        graph.add_conditional_edges(
            "retrieval_grader",
            self._decide_to_generate,
            {
                "all_docs_not_relevant": "transform_question",
                "at_least_one_doc_relevant": "generate",
            },
        )
        graph.add_conditional_edges(
            "generate",
            self._ground_validator,
            {
                "grounded_and_addresses_question": END,
                "generation_not_grounded": "generate",
                "grounded_but_does_not_address_question": "transform_question",
            }
        )
        graph.add_edge("transform_question", "retrieve")

        return graph
    
    def answer(self):
        pass

# Create the agent instance for LangGraph
agent = RagAgent(agent_id="bt-hr").compiled_graph
    
if __name__ == "__main__":
    agent = RagAgent(agent_id = "bt-hr")
    question_list = ["Hi, how is valvular heart disease diagnosed?"]
    history = []
    for question in question_list:
        user_message = MessageSchema(
            role="admin",
            content=question,
            datetime = datetime.strptime("2021-10-01T10:00:02", "%Y-%m-%dT%H:%M:%S"),
        )
        request = ChatSchema(
            user = "christian",
            conversation = ConversationSchema(
                id = "1",
                chatbotId = "bt-hr",
                history = history,
                question = user_message,
            )
        )
        final_response = agent.answer(request_body = request)
        print(f"Final response: {final_response.content}")




