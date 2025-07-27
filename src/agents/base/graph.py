#!/usr/bin/env python3
import os,sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from typing import Callable
from langchain_core.runnables import RunnableConfig # type: ignore
from langchain_core.messages import AIMessage # type: ignore
from langgraph.graph import StateGraph, START, END # type: ignore
from langchain_community.chat_models import ChatOllama # type: ignore

from src.utils.base_state import BaseState
from src.utils.chat import ConversationSchema

class BaseAgent():
    def __init__(self, agent_id: str):
        self.agent_id = agent_id 
        self.llm = ChatOllama(model="llama3.2:1b", temperature=0.7, verbose=False)
        self.graph = self._create_graph()

    def _initialize_storage_files(self):
        pass

    def _create_graph(self) -> StateGraph:
        """
        Creates the graph for the agent.
        This method should be overridden by subclasses to define their specific graph structure.
        """
        raise NotImplementedError(
            "Subclasses must implement their own graph creation logic."
        )

    def _add_edge(self, source: str, target: str):
        self.graph.add_edge(source, target)

    def _remove_edge(self, source: str, target: str):
        if (source, target) in self.graph.edges:
            self.graph.edges.remove((source, target))

    def _add_conditional_edge(self, source: str, node: Callable, branches: dict, after: str | None):
        self.graph.add_conditional_edges(source=after, path=node, path_map=branches)

    def remove_node(self, name: str) -> None:
        if name not in self.graph.nodes:
            return 
        
        self.graph.compiled = False 

        incoming = [(s,t) for s,t in self.graph.edges if t == name]
        outgoing = [(s,t) for s,t in self.graph.edges if s == name]

        for s,_ in incoming:
            self._remove_edge(s,name)
        for _,t in outgoing:
            self._remove_edge(name,t)

        for s,_ in incoming:
            for _,t in outgoing:
                self._add_edge(s,t)

        del self.graph.nodes[name]

    def overwrite_node(self, name: str, node: Callable) -> None:
        if name not in self.graph.nodes:
            return 

        self.graph.add_node(name, node)
    
    def _convert_conversation_to_messages(self, conversation: ConversationSchema) -> list[dict]:
        messages = []
        for message in conversation.history:
            messages.append({"role": message.role, "content": message.content})

        messages.append(
            {
                "role": conversation.question.role,
                "content": conversation.question.content,
            }
        )

        return messages[-2 * 5:] # in realtà dovrei usare un config.json ed inserire il valore 5 come self.config.memory.length
        # Moltiplico per 2 perchè la in config.json scrivo quanti scambi di messaggi devo tenere, ma un singolo scambio
        # contiene due messaggi (uno dell'utente e uno dell'assistente)
        # e.g. in config.json ho length=5 -> devo considerare gli ultimi 10 messaggi

    def answer(self, request_body: dict) -> None:
        """
        Process an incoming conversation request and generate an answer.
        This method should be overridden by subclasses to implement their specific answer generation logic.

        :param request_body: Dictionary containing conversation details
        """
        raise NotImplementedError(
            "Subclasses must implement their own answer generation logic."
        )
