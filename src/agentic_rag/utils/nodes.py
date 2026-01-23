import copy
import re
from enum import Enum
from importlib.abc import Traversable
from pathlib import Path
from typing import Dict, List

import yaml
from jinja2 import Template, DebugUndefined


class NodeType(str, Enum):
    dspy = "dspy"
    langchain = "langchain"


def deep_merge(a: Dict, b: Dict) -> Dict:
    result = copy.deepcopy(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_yaml(filepath: Traversable | Path) -> Dict:
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
