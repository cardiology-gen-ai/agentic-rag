import json
import os
import re
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Callable, Optional, Any

import mlflow
import numpy as np
from pydantic import BaseModel

from tests.accuracy_tests.metrics import init_metric, Metric
from cardiology_gen_ai.utils.logger import get_logger

from tests.accuracy_tests.nodes_config import NodeOptimizationConfig


class EvalTestConfig(BaseModel, ABC):
    test_name: str
    test_id: int
    data_folder: Path
    data_name: str
    results_folder: Path
    fuzzy_threshold: Optional[float] = None


class EvalTest(ABC):
    def __init__(self, test_config_path: pathlib.Path, node_config: NodeOptimizationConfig):
        self.logger = get_logger("Evaluation Test")
        self._test_config_path = test_config_path
        self.test_config: EvalTestConfig = self.get_config()
        self.node_config = node_config
        self._set_test_id()
        self.metrics: List[Metric] = [
            init_metric(metric_name, fuzz_threshold=self.test_config.fuzzy_threshold)
             for metric_name in self.node_config.metrics
        ]

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self._test_config_path, "r") as config_file:
                raw_json = config_file.read()

                def replace_env_var(match):
                    var_name = match.group(1)
                    return os.environ.get(var_name, f"<MISSING:{var_name}>")

                interpolated_json = re.sub(r"\$\{(\w+)\}", replace_env_var, raw_json)
                return json.loads(interpolated_json)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self._test_config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in configuration file at {self._test_config_path}")

    def _set_test_id(self):
        node_name = self.node_config.name
        test_id = self.test_config.test_id
        current_file = self.test_config.results_folder / f"run_{test_id}" / f"{node_name}.json"
        while current_file.is_file():
            test_id += 1
            current_file = self.test_config.results_folder / f"run_{test_id}" / f"{node_name}.json"
        self.test_config = self.test_config.model_copy(update={"test_id": test_id})

    @abstractmethod
    def get_config(self) -> EvalTestConfig:
        pass

    @abstractmethod
    def get_data(self, **kwargs) -> List[Dict]:
        pass

    @abstractmethod
    def get_predict_fn(self, **kwargs) -> Callable:
        pass

    def get_metrics_quantiles(self, save_results: bool = True, **kwargs) -> Dict[str, List[float]]:
        self.logger.info(f"Getting metric quantiles for metrics: {self.node_config.metrics}")
        data = self.get_data(**kwargs)
        predict_fn = self.get_predict_fn(**kwargs)
        with mlflow.start_run(run_name=f"{self.test_config.test_name}_{self.test_config.test_id}", nested=True):
            results = mlflow.genai.evaluate(
                data=data,
                predict_fn=predict_fn,
                scorers=[m.eval_fn for m in self.metrics],
            )
        df_results = results.result_df
        if save_results:
            df_results_filename = (
                    self.test_config.results_folder/ f"run_{self.test_config.test_id}" / f"{self.node_config.name}.csv"
            )
            df_results_filename.parent.mkdir(parents=True, exist_ok=True)
            columns_to_save = ["request", "response", "execution_duration"] + [f"{m.name}/value" for m in self.metrics]
            if "_output/value" in df_results.columns.tolist():
                columns_to_save.append("_output/value")
            df_results.to_csv(
                str(df_results_filename),
                columns=columns_to_save,
                index=False
            )
        scores = [np.asarray(df_results[f"{m.name}/value"].tolist()) for m in self.metrics]
        return {metric.name: np.quantile(scores, q=[0.01, 0.25, 0.5, 0.75, 0.99]).tolist()
                for metric, scores in zip(self.metrics, np.asarray(scores).T)}

    def get_eval_results(self, **kwargs) -> Dict:
        self.logger.info(f"Starting evaluation run for node: {self.node_config.name}")
        start = datetime.now()
        timestamp_start = start.strftime("%Y-%m-%d_%H-%M-%S")
        eval_results = self.get_metrics_quantiles(**kwargs)
        elapsed_time = datetime.now() - start
        return (self.test_config.model_dump(exclude={"data_folder", "data_name", "results_folder"}) |
                {"timestamp_start": timestamp_start, "elapsed_time": elapsed_time.total_seconds(),
                 "data": str(self.test_config.data_folder / self.test_config.data_name), "scores": eval_results})

    @abstractmethod
    def run_eval(self, save: bool = True):
        pass
