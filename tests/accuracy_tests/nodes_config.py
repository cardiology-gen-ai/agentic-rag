from abc import ABC
from enum import Enum
from importlib.resources import files
from typing import List, Dict, Any, Optional

import mlflow
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
import dspy

from src.agentic_rag.agent.prompts.output_schemas import node_output_schemas
from src.agentic_rag.managers.nodes_manager import NodeConfig, Node, DspyNode, LangChainNode, NodeFactory
from src.agentic_rag.utils.nodes import NodeType


class MetricAggregationFunction(str, Enum):
    average = "average"
    percentage_passing = "percentage"
    all_passing = "all"
    at_least_one_passing = "at_least_one"
    at_least_half_passing = "at_least_half"


class NodeOptimizationConfig(NodeConfig):
    upstream_nodes_name: List[str] = None
    downstream_nodes_name: List[str] = None
    input_keys: Dict[str, str] | List[str] = None
    output_keys: Dict[str, str] | List[str] = None
    invoke_keys: List[str] = None
    metrics: List[str]
    metrics_aggregator: MetricAggregationFunction = MetricAggregationFunction.average

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "NodeOptimizationConfig":
        base_instance = NodeConfig.from_config(config_dict)
        input_keys = \
            {"_expectations": config_dict["keys"].get("input", ""), "_context": config_dict["keys"].get("context", "")}
        output_keys = {"_output": config_dict["keys"].get("output", "")}
        invoke_keys = [k for k in [config_dict["keys"].get("input", ""), config_dict["keys"].get("context", "")] +
                       config_dict["keys"].get("invoke_args", []) if k != ""]
        metrics = config_dict.get("metrics", [])
        metrics_aggregator = MetricAggregationFunction(config_dict.get("metrics_aggregator", "average"))
        return cls(
            **base_instance.model_dump(),
            upstream_nodes_name=config_dict.get("upstream_nodes", []),
            downstream_nodes_name=config_dict.get("downstream_nodes", []),
            input_keys=input_keys,
            output_keys=output_keys,
            invoke_keys=invoke_keys,
            metrics=metrics,
            metrics_aggregator=metrics_aggregator
        )


class TestNode(Node, ABC):
    input_keys: Optional[Dict[str, str]] = None
    output_keys: Optional[Dict[str, str]] = None

    def format_input(self, invoke_dict: Dict[str, Any]) -> Dict[str, Any]:
        input_dict = {}
        if self.input_keys is not None:
            input_dict = {k: invoke_dict.get(v) for k, v in self.input_keys.items()}
        remaining_invoke_dict = {k: v for k, v in invoke_dict.items() if k not in set(self.input_keys.values())}
        return input_dict | remaining_invoke_dict

    def invert_format_input(self, formatted_dict: Dict[str, Any]) -> Dict[str, Any]:
        invoke_dict = {}
        reverse_keys = {alias: original for alias, original in self.input_keys.items()}
        for key, value in formatted_dict.items():
            if key in reverse_keys:
                invoke_dict[reverse_keys[key]] = value
            else:
                invoke_dict[key] = value
        return invoke_dict

    @mlflow.trace()
    def formatted_invoke(self, inputs: Dict, **kwargs) -> str:
        invoke_dict = self.invert_format_input(inputs)
        node_output = self.invoke(invoke_dict, **kwargs)
        if isinstance(node_output, BaseModel):
            node_output_dict = node_output.model_dump()
            return str(list(node_output_dict.values())[0])
        if not isinstance(node_output, str):
            return self.extract_output(node_output)
        return node_output


class NodeChain(TestNode):
    nodes: List[TestNode]
    nodes_config: List[NodeOptimizationConfig]

    def invoke(self, invoke_dict: Dict, **kwargs):
        current_dict = invoke_dict.copy()
        node_output = None
        for node, node_config in zip(self.nodes, self.nodes_config):
            node_input = {v: current_dict.get(v) for v in node_config.invoke_keys}
            node_output = node.invoke(node_input, **kwargs)
            if isinstance(node_output, dict):
                current_dict.update({node.output_keys.get(k, k): v for k, v in node_output.items()})  # TODO: check (structured output especially)
            else:
                current_dict[next(iter(node.output_keys.values()))] = node_output
            # TODO: adjust accordingly
            # if "retrieval_query" in node.output_keys.values():
            #     retrieval_query = current_dict.get("retrieval_query")
            #     retrieval_queries = [retrieval_query] if isinstance(retrieval_query, str) else retrieval_query
            #     documents = []
            #     for current_retrieval_query in retrieval_queries:
            #         current_documents = get_documents_fn(
            #             question=current_retrieval_query,
            #             agent_config=kwargs["agent_config"],
            #             logger=kwargs["logger"],
            #             search_client=kwargs["search_client"],
            #         )
            #         documents += current_documents
            #     current_dict["documents"] = documents
        return node_output


class DspyTestNode(DspyNode, TestNode):

    def extract_output(self, node_output: dspy.Prediction) -> str:
        pred = node_output.with_inputs(*[v for _, v in self.output_keys.items() if v != ""])
        pred_inputs = pred.inputs().toDict()
        return list(pred_inputs.values())[0]


class LangChainTestNode(LangChainNode, TestNode):
    type: NodeType = NodeType.langchain


class TestNodeFactory(NodeFactory):
    def __init__(self,prompt_folder=files("src.agentic_rag.agent.prompts"), node_type: NodeType = NodeType.langchain):
        super().__init__(prompt_folder=prompt_folder, node_type=node_type)

    def build_test_node(self,
                        filename: str,
                        input_keys: Optional[Dict[str, str]] = None,
                        output_keys: Optional[Dict[str, str]] = None,
                        llm: Optional[BaseChatModel] = None,
                        output_schema: Optional[BaseModel | Dict | Any] = None,
                        structured_output: Optional[bool] = None,
                        prompt_version: str | int = "default",
                        register_prompt: bool = False,
                        **kwargs
                        ) -> Node:
        base_node = self.build_node(
            filename=filename, llm=llm, output_schema=output_schema, structured_output=structured_output,
            prompt_version=prompt_version, register_prompt=register_prompt, **kwargs
        )
        return LangChainTestNode(**base_node.model_dump(), input_keys=input_keys, output_keys=output_keys) \
            if self.node_type == NodeType.langchain \
            else DspyTestNode(**base_node.model_dump(), input_keys=input_keys, output_keys=output_keys)

    def build_node_from_config(self, node_config: NodeOptimizationConfig, llm: Optional[BaseChatModel] = None,
                               register_prompt: bool = False, **kwargs) -> Node:
        return self.build_test_node(
            filename=node_config.prompt_filename,
            input_keys=node_config.input_keys,
            output_keys=node_config.output_keys,
            llm=llm,
            prompt_version=node_config.prompt_version,
            output_schema=node_output_schemas.get(node_config.name, None),
            register_prompt=register_prompt,
            **kwargs,
        )

    def build_chain(self, chain_name: str, nodes_config_list: List[NodeOptimizationConfig],
                    llm: Optional[BaseChatModel] = None, register_prompt: bool = False, **kwargs) -> NodeChain:
        central_node_config = next(config for config in nodes_config_list if config.name == chain_name)
        chain_nodes_config = []
        n_upstream_nodes = len(central_node_config.upstream_nodes_name) if central_node_config.upstream_nodes_name else 0
        if central_node_config.upstream_nodes_name:
            for upstream_node_name in central_node_config.upstream_nodes_name:
                upstream_node_config = next(config for config in nodes_config_list if config.name == upstream_node_name)
                chain_nodes_config.append(upstream_node_config)
        chain_nodes_config.append(central_node_config)
        if central_node_config.downstream_nodes_name:
            for downstream_node_name in central_node_config.downstream_nodes_name:
                downstream_node_config = next(config for config in nodes_config_list if config.name == downstream_node_name)
                chain_nodes_config.append(downstream_node_config)
        chain_nodes = []
        for node_config in chain_nodes_config:
            node = self.build_node_from_config(node_config=node_config, llm=llm, register_prompt=register_prompt, **kwargs)
            chain_nodes.append(node)
        return NodeChain(
            name=chain_name,
            type=self.node_type,
            input_keys=chain_nodes[0].input_keys,
            output_keys=chain_nodes[-1].output_keys,
            prompt=chain_nodes[n_upstream_nodes].prompt,
            runnable=chain_nodes[n_upstream_nodes].runnable,
            nodes=chain_nodes,
            nodes_config=chain_nodes_config,
        )
