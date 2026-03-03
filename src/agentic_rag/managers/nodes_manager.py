from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from importlib.resources import files

import dspy
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from pydantic import BaseModel, ConfigDict

from agentic_rag.agent.prompts.output_schemas import node_output_schemas
from agentic_rag.managers.prompt_manager import PromptFactory
from agentic_rag.utils.nodes import _strip_think, _get_final, NodeType


def get_llm_with_structured_output(llm: BaseChatModel, output_schema: BaseModel | Dict | Any, prompt: ChatPromptTemplate):
    llm_with_structured_output = llm.with_structured_output(output_schema, include_raw=True)
    language_detector = prompt | llm_with_structured_output
    return language_detector


def get_chain_with_unstructured_output(llm: BaseChatModel, prompt: ChatPromptTemplate):
    runnable = prompt | llm | StrOutputParser() | RunnableLambda(_strip_think) | RunnableLambda(_get_final)
    return runnable


def get_chain_with_structured_output(llm: BaseChatModel, output_schema: BaseModel | Dict | Any, prompt: ChatPromptTemplate):
    parser = JsonOutputParser(pydantic_object=output_schema)
    to_model = RunnableLambda(lambda d: output_schema.model_validate(d))
    runnable = get_chain_with_unstructured_output(llm, prompt) | parser | to_model
    return runnable


class NodeConfig(BaseModel):
    name: str
    prompt_filename: str = ""
    prompt_version: str = ""

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "NodeConfig":
        node_name = config_dict["name"]
        prompt_filename = config_dict.get("prompt_filename",node_name)
        return cls(
            name=node_name,
            prompt_filename=prompt_filename,
            prompt_version=config_dict.get("prompt_version", "default"),
        )


class Node(BaseModel, ABC):
    name: str
    type: NodeType
    prompt: dspy.SignatureMeta | ChatPromptTemplate
    runnable: dspy.Module | Runnable
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def invoke(self, invoke_dict: Dict, **kwargs):
        pass


class DspyNode(Node):
    name: str
    type: NodeType = NodeType.dspy
    prompt: dspy.SignatureMeta
    runnable: dspy.Module = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any, cot=True) -> None:
        self.runnable = dspy.ChainOfThought(signature=self.prompt) if cot else dspy.Predict(signature=self.prompt)

    def invoke(self, invoke_dict: Dict, cot=True, **kwargs):
        return self.runnable(**invoke_dict)


class LangChainNode(Node):
    name: str
    type: NodeType = NodeType.langchain
    llm: BaseChatModel
    structured_output: Optional[bool]
    output_schema: Optional[BaseModel | Dict | Any]
    prompt: ChatPromptTemplate
    runnable: Runnable = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.output_schema:
            self.runnable = get_llm_with_structured_output(self.llm, self.output_schema, self.prompt) if self.structured_output \
                else get_chain_with_structured_output(self.llm, self.output_schema, self.prompt)
            if not self.structured_output:
                self.structured_output = True
        else:
            self.runnable = get_chain_with_unstructured_output(self.llm, self.prompt)
            if not self.structured_output:
                self.structured_output = False

    def invoke(self, invoke_dict: Dict, config: Optional[RunnableConfig] = None, with_retry: bool = True) -> None:
        response = self.runnable.invoke(invoke_dict, config=config)
        if self.structured_output and with_retry:
            if response["parsing_error"] is None:
                return response["parsed"]
            else:
                retry_node = LangChainNode(name=self.name,llm=self.llm, structured_output=False,
                                           output_schema=self.output_schema, prompt=self.prompt)
                return retry_node.runnable.invoke(invoke_dict, config=config)
        return response["parsed"] if self.structured_output else response


class NodeFactory:
    def __init__(self, prompt_folder=files("agentic_rag.agent.prompts"),
                 node_type: NodeType = NodeType.langchain):
        if isinstance(prompt_folder, str):
            prompt_folder = files(prompt_folder)
        self.prompt_factory = PromptFactory(prompt_folder=prompt_folder)
        self.node_type = node_type

    def build_node(self,
                   filename: str,
                   llm: Optional[BaseChatModel] = None,
                   output_schema: Optional[BaseModel | Dict | Any] = None,
                   structured_output: Optional[bool] = None,
                   prompt_version: str | int = "default",
                   register_prompt: bool = False,
                   **kwargs
                   ) -> Node:
        if output_schema is None:
            output_schema = node_output_schemas.get(filename, None)
        format_instruction = output_schema.format_instruction() if output_schema and not structured_output else ""
        node_prompt = self.prompt_factory.build_prompt(
            name=filename, version=prompt_version, node_type=self.node_type, register_prompt=register_prompt,
            format_instruction=format_instruction,**kwargs
        )
        return LangChainNode(name=filename, llm=llm, output_schema=output_schema, prompt=node_prompt,
                             structured_output=structured_output) \
            if self.node_type == NodeType.langchain else DspyNode(
            name=filename, prompt=node_prompt)

    def build_node_from_config(self, node_config: NodeConfig, llm: Optional[BaseChatModel] = None,
                               register_prompt: bool = False, **kwargs) -> Node:
        structured_output = kwargs.pop("structured_output", None)
        output_schema = kwargs.pop(
            "output_schema",
            node_output_schemas.get(node_config.name, None),
        )
        return self.build_node(
            filename=node_config.prompt_filename,
            llm=llm,
            prompt_version=node_config.prompt_version,
            structured_output=structured_output,
            output_schema=output_schema,
            register_prompt=register_prompt,
            **kwargs,
        )
