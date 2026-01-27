from abc import ABC
from typing import Literal, Optional, Callable, Dict, Any, List
# import sys
# import subprocess
# subprocess.check_call([
#     sys.executable, "-m", "pip", "install", "fuzzywuzzy"
# ])
from fuzzywuzzy import fuzz
from mlflow.genai import scorer
from pydantic import BaseModel, ConfigDict


def list_fuzzy_match(_expectations: List, _output: List, fuzz_threshold: float) -> int:
    matched_output, intersection = set(), 0
    for inp in _expectations:
        for j, out in enumerate(_output):
            if j in matched_output:
                continue
            if fuzz.partial_ratio(str(inp), str(out)) >= fuzz_threshold:
                intersection += 1
                matched_output.add(j)
                break
    return intersection


def precision_recall_common(_expectation: List, _output: List, fuzz_threshold: Optional[float]) -> int:
    if not fuzz_threshold:
        set_input, set_output = set(_expectation), set(_output)
        intersection = len(set_input & set_output)
    else:
        intersection = list_fuzzy_match(_expectation, _output, fuzz_threshold)
    return intersection

# we assume all metrics are either discrete {0, 1} or range [0, 1]
class Metric(BaseModel, ABC):
    name: str = ""
    type: Literal["binary", "range"] = None
    pass_threshold: float = None
    eval_fn: Callable = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FuzzyMetric(Metric, ABC):
    fuzz_threshold: Optional[float] = None


class Accuracy(FuzzyMetric):
    name: str = "accuracy"
    type: Literal["binary", "range"] = "range"
    pass_threshold: float = 1.

    def model_post_init(self, __context: Any) -> None:
        self.eval_fn = self.make_eval_fn()

    def make_eval_fn(self) -> Callable:
        fuzz_threshold = self.fuzz_threshold
        @scorer
        def accuracy(expectations: Dict, outputs: Dict | str) -> float:
            # expectations are ground truth, outputs is predictions
            _output = outputs.get("_output") if isinstance(outputs, Dict) else outputs
            _expectations = expectations.get("_output")
            if isinstance(_output, list):
                _expectations = [_expectations] if not isinstance(_expectations, list) else _expectations
                # jaccard similarity between lists
                if len(_output) == 0:
                    return 0.
                if not fuzz_threshold:
                    set_input, set_output = set(_expectations), set(_output)
                    return len(set_input & set_output) / len(set_input | set_output)
                else:
                    intersection = list_fuzzy_match(_expectations, _output, fuzz_threshold)
                    union = len(_expectations) + len(_output) - intersection
                    return intersection / union
            return float(_expectations == _output)
        return accuracy


class Precision(FuzzyMetric):
    name: str = "precision"
    type: Literal["binary", "range"] = "range"
    pass_threshold: float = 0.75

    def model_post_init(self, __context: Any) -> None:
        self.eval_fn = self.make_eval_fn()

    def make_eval_fn(self) -> Callable:
        fuzz_threshold = self.fuzz_threshold
        @scorer
        def precision(expectations: Dict, outputs: Dict | str) -> float:
            _expectations = expectations.get("_output")
            _output = outputs.get("_output") if isinstance(outputs, Dict) else outputs
            if len(_output) == 0:
                return 0
            assert (isinstance(_expectations, list) & isinstance(_output, list))
            intersection = precision_recall_common(_expectations, _output, fuzz_threshold)
            return intersection / len(_output)
        return precision


class Recall(FuzzyMetric):
    name: str = "recall"
    type: Literal["binary", "range"] = "range"
    pass_threshold: float = 0.75

    def model_post_init(self, __context: Any) -> None:
        self.eval_fn = self.make_eval_fn()

    def make_eval_fn(self) -> Callable:
        fuzz_threshold = self.fuzz_threshold
        @scorer
        def recall(expectations: Dict, outputs: Dict | str) -> float:
            _output = outputs.get("_output") if isinstance(outputs, Dict) else outputs
            _expectations = expectations.get("_output")
            if len(_output) == 0 or len(_expectations) == 0:
                return 0
            assert (isinstance(_expectations, list) & isinstance(_output, list))
            intersection = precision_recall_common(_expectations, _output, fuzz_threshold)
            return intersection / len(_expectations)
        return recall


def init_metric(
        name: str,
        fuzz_threshold: Optional[float] = None,
) -> Metric:
    metrics_class_map = {
        cls().name: cls for cls in Metric.__subclasses__() + FuzzyMetric.__subclasses__() if cls().name != ""
    }
    cls = metrics_class_map.get(name)
    if cls is None:
        raise ValueError(f"Unknown metric {name}")
    if cls in FuzzyMetric.__subclasses__():
        return cls(name=name, fuzz_threshold=fuzz_threshold)
    return cls(name=name)
