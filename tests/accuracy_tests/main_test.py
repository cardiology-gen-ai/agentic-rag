import logging
import os
from pathlib import Path

import mlflow

from src.agentic_rag.persistence.db import ensure_database

from tests.accuracy_tests.nodes_config import NodeOptimizationConfig
from tests.accuracy_tests.test_retrievers import EvalRetriever

if os.getenv("MLFLOW_DB", None):
    ensure_database(db_name=os.getenv("MLFLOW_DB"))
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(
        f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:5432/{os.getenv("MLFLOW_DB")}")
    mlflow.set_experiment(f"{os.getenv('AGENT_ID')}_accuracy_tests")
    mlflow.langchain.autolog()

node_config = NodeOptimizationConfig(
    name="retriever",
    input_keys=["question"],
    output_keys=["sections"],  # ["sections"], ["document"]
    metrics = ["accuracy", "precision", "recall"],
)

retrieval_eval = EvalRetriever(
    test_config_path=Path("tests/test_configs/retriever/test_config.json"),
    node_config=node_config,
)
retrieval_eval.run_eval(save=True)



