from pathlib import Path
from typing import List

from langchain_core.messages import AnyMessage

from agentic_rag.config.manager import NodePromptConfig
from agentic_rag.managers.llm_manager import LLMManager
from agentic_rag.managers.nodes_manager import NodeFactory, NodeConfig
from agentic_rag.utils.nodes import load_yaml


class LLMService:
    def __init__(self, llm_manager: LLMManager, nodes_prompt_config: NodePromptConfig):
        self.llm = llm_manager.llm
        self.node_factory = NodeFactory(prompt_folder=nodes_prompt_config.prompts)
        self.nodes_config = self._load_nodes_config(nodes_prompt_config)
        self.config = llm_manager.config

    @staticmethod
    def _load_nodes_config(nodes_prompt_config: NodePromptConfig) -> List[NodeConfig]:
        nodes_config_dict = load_yaml(Path(Path.cwd() / nodes_prompt_config.config))
        return [NodeConfig.from_config(node_config) for node_config in nodes_config_dict["nodes"]]

    def _get_node_config(self, name: str) -> NodeConfig:
        return next(node_config for node_config in self.nodes_config if node_config.name == name)

    def build_node(self, name: str, structured_output=False, output_schema=None, **kwargs):
        config = self._get_node_config(name)
        return self.node_factory.build_node_from_config(
            config,
            self.llm,
            structured_output=structured_output,
            output_schema=output_schema,
            **kwargs
        )

    @staticmethod
    def serialize_history(messages: List[AnyMessage]) -> str:
        formatted = []
        for msg in messages:
            role = "User" if msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)