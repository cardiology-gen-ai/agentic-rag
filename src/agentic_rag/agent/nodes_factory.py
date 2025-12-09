from typing import Dict, Any, Optional, List
import re
import yaml
import copy
from importlib.resources import files
from importlib.resources.abc import Traversable

from jinja2 import Template, DebugUndefined
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from pydantic import BaseModel


def deep_merge(a: Dict, b: Dict) -> Dict:
    result = copy.deepcopy(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_yaml(filepath: Traversable) -> Dict:
    if not filepath.is_file():
        raise FileNotFoundError(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render(template_str: str, context: Dict) -> str:
    t = Template(template_str, undefined=DebugUndefined)
    return t.render(**context)


def render_template(data: str | List[str] | Dict, context: Dict):
    if isinstance(data, str):
        return render(data, context)

    if isinstance(data, list):
        return [render_template(x, context) for x in data]

    if isinstance(data, dict):
        return {
            k: render_template(v, context)
            for k, v in data.items()
        }


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


class PromptFactory:
    def __init__(self, prompt_folder=files("agentic_rag.agent.prompts")):
        self.prompt_folder = prompt_folder
        self.fragments = load_yaml(self.prompt_folder / "fragments.yaml").get("fragments", {})

    def build_prompt_components(self, name: str, version: str | int):
        prompt_file = load_yaml(self.prompt_folder / f"{name}.yaml")
        prompt_components = prompt_file.get("default", {})
        if version != "default":
            version_prompt_overrides = prompt_file.get("versions", {}).get(version)
            if version_prompt_overrides:
                version_prompt_no_overrides = {k: v for k,v in version_prompt_overrides.items() if k != "overrides"}
                prompt_components = deep_merge(prompt_components, version_prompt_no_overrides)
                overrides = version_prompt_overrides.get("overrides", {})
                if overrides:
                    prompt_components = deep_merge(prompt_components, overrides)
        return prompt_components

    def render_prompt_components(self, prompt_components: Dict, **kwargs):
        context = {
            **prompt_components,
            "fragments": self.fragments,
            **kwargs,
        }
        rendered_text = render_template(prompt_components, context)
        messages = [(role, text) for role, text in rendered_text.items()
                    if role in ["human", "user", "ai", "assistant", "system"]]
        return ChatPromptTemplate.from_messages(messages)

    def build_prompt(self, name: str, version: str | int = "default", **kwargs):
        prompt_components = self.build_prompt_components(name, version)
        return self.render_prompt_components(prompt_components, **kwargs)


class LangChainNode(BaseModel):
    name: str
    llm: BaseChatModel
    structured_output: bool = False
    output_schema: Optional[BaseModel | Dict | Any]
    prompt: ChatPromptTemplate
    runnable: Runnable = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        if self.output_schema:
            self.runnable = get_llm_with_structured_output(self.llm, self.output_schema, self.prompt) if self.structured_output \
                else get_chain_with_structured_output(self.llm, self.output_schema, self.prompt)
        else:
            self.runnable = get_chain_with_unstructured_output(self.llm, self.prompt)

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
    def __init__(self, prompt_folder=files("agentic_rag.agent.prompts")):
        self.prompt_factory = PromptFactory(prompt_folder=prompt_folder)

    def build_node(self, name: str, llm: BaseChatModel, structured_output: bool,
                   output_schema: BaseModel | Dict | Any = None, version: str | int = "default", **kwargs):
        format_instruction = output_schema.format_instruction() if output_schema and not structured_output else ""
        node_prompt = self.prompt_factory.build_prompt(name, version, format_instruction=format_instruction, **kwargs)
        return LangChainNode(name=name, llm=llm, structured_output=structured_output,
                             output_schema=output_schema, prompt=node_prompt)
