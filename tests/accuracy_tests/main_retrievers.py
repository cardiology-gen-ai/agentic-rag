import json
import logging
import os
import pathlib

import mlflow
from cardiology_gen_ai import IndexingConfig

from agentic_rag.config.manager import SearchConfig, SearchTypeNames, FusionStrategy
from agentic_rag.persistence.db import ensure_database
from tests.accuracy_tests.nodes_config import NodeOptimizationConfig
from tests.accuracy_tests.test_retrievers import RetrieverTestConfig, EvalRetriever

if __name__ == "__main__":
    TEST_NAME = "prova"
    TEST_ID = 1

    if os.getenv("MLFLOW_DB", None):
        ensure_database(db_name=os.getenv("MLFLOW_DB"))
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        mlflow.set_tracking_uri(
            f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:5432/{os.getenv("MLFLOW_DB")}")
        mlflow.set_experiment(f"{TEST_NAME}_{TEST_ID}")
        mlflow.openai.autolog()
        # in terminal (with correct parameters)
        # mlflow server --backend-store-uri postgresql+psycopg2://postgres:example@localhost:5432/mlflow --host 0.0.0.0 --port 5000 --allowed-hosts "*" --cors-allowed-origins "*"

    DATA_FOLDER = pathlib.Path("/Users/giai/Desktop/repos/cardiology-gen-ai/agentic-rag/tests/data")
    DATA_FILE = "subset_test_en.json"
    RESULTS_FOLDER = pathlib.Path("/Users/giai/Desktop/repos/cardiology-gen-ai/agentic-rag/tests/results")
    index_config_folder = pathlib.Path("/Users/giai/Desktop/repos/cardiology-gen-ai/agentic-rag/tests/cvd/indexes")
    index_config_file = "flat_faiss.json"

    with open(index_config_folder / index_config_file, "r") as f:
        index_dict = json.load(f)
    index_config = IndexingConfig.from_config(index_dict["indexing"])

    k = 15

    search_config = SearchConfig(
        type=SearchTypeNames.similarity,
        k=k,
        top_k=5,
        fusion=FusionStrategy.bm25,
        kwargs={"k": k},
    )

    fuzzy_threshold = 0.9

    test_config = RetrieverTestConfig(
        test_name=TEST_NAME,
        test_id=TEST_ID,
        data_folder=DATA_FOLDER,
        data_name=DATA_FILE,
        results_folder=RESULTS_FOLDER,
        fuzzy_threshold=fuzzy_threshold,
        index_config=index_config,  # this can be a list of indexes
        search_config=search_config,
    )

    output_key = "sections"

    node_config = NodeOptimizationConfig(
        name="retriever",
        input_keys=["question"],
        output_keys=[output_key],  # ["sections"], ["document"]
        metrics=["accuracy", "precision", "recall"],
    )

    retrieval_eval = EvalRetriever(
        test_config_path=test_config,
        node_config=node_config,
        eval_agent=False,
    )
    retrieval_eval.run_eval(save=True)
