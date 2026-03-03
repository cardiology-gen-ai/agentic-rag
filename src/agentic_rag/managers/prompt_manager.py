from importlib.resources import files
from typing import Dict, Optional

import dspy
import mlflow
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agentic_rag.agent.prompts.dspy.dspy_prompts import node_names
from agentic_rag.utils.nodes import NodeType, deep_merge, load_yaml, render_template


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
                version_prompt_no_overrides = {k: v for k, v in version_prompt_overrides.items() if k != "overrides"}
                prompt_components = deep_merge(prompt_components, version_prompt_no_overrides)
                overrides = version_prompt_overrides.get("overrides", {})
                if overrides:
                    prompt_components = deep_merge(prompt_components, overrides)
        return prompt_components

    @staticmethod
    def register_prompt(name: str, component: str, commit_message: Optional[str] = None, tags: Optional[Dict] = None):
        registered_prompt = mlflow.genai.register_prompt(
            name=name,
            template=component,
            commit_message=commit_message,
            tags=tags,
        )
        return registered_prompt

    def render_prompt_components(self, prompt_components: Dict, **kwargs):
        context = {
            **prompt_components,
            "fragments": self.fragments,
            **kwargs,
        }
        rendered_text = render_template(prompt_components, context)
        return rendered_text

    @staticmethod
    def build_langchain_prompt(rendered_components: Dict):
        messages = [(role, text) for role, text in rendered_components.items() if role in ["system"]]
        messages += [MessagesPlaceholder("history", optional=True)]
        messages += [(role, text) for role, text in rendered_components.items()
                     if role in ["human", "user", "ai", "assistant"]]
        return ChatPromptTemplate.from_messages(messages)

    @staticmethod
    def build_dspy_prompt(name: str, rendered_components: Dict):
        prompt_signature: dspy.Signature = eval(node_names[name])
        instructions = rendered_components.get("instructions", "")
        return prompt_signature.with_instructions(instructions=instructions) if instructions != "" \
            else prompt_signature

    def build_prompt(self, name: str, version: str | int = "default", node_type: NodeType = NodeType.langchain,
                     register_prompt: bool = False, **kwargs):
        prompt_components = self.build_prompt_components(name, version)
        rendered_components = self.render_prompt_components(prompt_components, **kwargs)
        if register_prompt:
            registered_component = rendered_components.get("instructions", "") if node_type == NodeType.dspy \
                else rendered_components.get("system", "")
            _ = PromptFactory.register_prompt(name, registered_component)
        return self.build_langchain_prompt(rendered_components) if node_type == NodeType.langchain \
            else self.build_dspy_prompt(name, rendered_components)
