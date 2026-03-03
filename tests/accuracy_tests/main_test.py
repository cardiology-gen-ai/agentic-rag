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


for output_key in ["document", "sections"]:
    node_config = NodeOptimizationConfig(
        name="retriever",
        input_keys=["question"],
        output_keys=[output_key],  # ["sections"], ["document"]
        metrics = ["accuracy", "precision", "recall"],
    )


    if os.getenv("MLFLOW_DB", None):
        mlflow.set_experiment(f"{os.getenv('AGENT_ID')}_graph_toc_{output_key}_retrieval_accuracy_tests")
        mlflow.openai.autolog()
        # mlflow.langchain.autolog()


    for i in range(9):
        if i not in [0, 8]:
            continue
        retrieval_eval = EvalRetriever(
            test_config_path=Path(f"tests/test_configs/retriever/dense/test_config_{i}.json"),
            node_config=node_config,
            eval_agent=True,
        )
        retrieval_eval.run_eval(save=True)
        del retrieval_eval



