"""Batch smoke test for the structured KG retrieval router.

This script routes a JSON test set through KGStructuredRouter and saves only
the structured retrieval plans. It does not query Neo4j.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agentic_rag.agent.services.llm_service import LLMService
from agentic_rag.config.manager import (
    LLMConfig,
    LLMProvider,
    NodePromptConfig,
)
from agentic_rag.kg.router import KGStructuredRouter
from agentic_rag.managers.llm_manager import LLMManager


QUESTION_KEYS = (
    "question",
    "query",
    "text",
)

ID_KEYS = (
    "question_id",
    "id",
    "qid",
    "query_id",
)

GOLD_KEYS = (
    "gold",
    "gold_sections",
    "expected_sections",
    "relevant_sections",
    "positive_sections",
    "source_sections",
    "documents",
    "document_id",
    "document_ids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Route a JSON test set through the structured KG router without "
            "querying Neo4j."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to the environment file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/data/subset_test_en_cm_syn.json"),
        help="JSON dataset containing the questions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kg_retrieval"),
        help="Directory where the router plans JSON file will be saved.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used by the router.",
    )
    parser.add_argument(
        "--nodes-config",
        type=Path,
        default=Path("configs/graphs/rag_agent.yaml"),
        help="YAML file containing the registered LLM nodes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of questions to route.",
    )
    return parser.parse_args()


def load_environment(env_file: Path) -> Path:
    resolved = env_file.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Environment file not found: {resolved}"
        )

    load_dotenv(
        dotenv_path=resolved,
        override=False,
    )

    return resolved


def load_json(path: Path) -> Any:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")

    return json.loads(resolved.read_text(encoding="utf-8"))


def get_question_text(record: dict[str, Any]) -> str | None:
    for key in QUESTION_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def find_question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [
            item for item in payload
            if isinstance(item, dict) and get_question_text(item)
        ]

    if not isinstance(payload, dict):
        return []

    for key in ("questions", "queries", "data", "items", "examples"):
        value = payload.get(key)
        if isinstance(value, list):
            records = [
                item for item in value
                if isinstance(item, dict) and get_question_text(item)
            ]
            if records:
                return records

    records: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if get_question_text(obj):
                records.append(obj)
                return
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(payload)

    return records


def get_question_id(record: dict[str, Any], index: int) -> str:
    for key in ID_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)

    return f"Q_{index:03d}"


def collect_gold(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in GOLD_KEYS
        if key in record
    }


def build_router(
    *,
    model_name: str,
    nodes_config_path: Path,
) -> KGStructuredRouter:
    llm_config = LLMConfig(
        model_name=model_name,
        provider=LLMProvider.openai,
        temperature=0.0,
        router_temperature=0.0,
        generator_temperature=None,
        grader_temperature=None,
        nbits=16,
    )

    llm_manager = LLMManager(config=llm_config)

    nodes_prompt_config = NodePromptConfig(
        config=nodes_config_path,
        prompts="agentic_rag.agent.prompts",
    )

    llm_service = LLMService(
        llm_manager=llm_manager,
        nodes_prompt_config=nodes_prompt_config,
    )

    return KGStructuredRouter(
        llm_service=llm_service,
        router_config=llm_manager.router_config,
    )


def make_run_path(output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"kg_router_plans_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir / "router_plans.json"


def main() -> None:
    args = parse_args()

    env_path = load_environment(args.env_file)
    dataset_path = args.dataset.expanduser().resolve()
    payload = load_json(dataset_path)

    records = find_question_records(payload)

    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        raise SystemExit(
            f"No question records found in dataset: {dataset_path}"
        )

    router = build_router(
        model_name=args.model,
        nodes_config_path=args.nodes_config,
    )

    results: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        question_id = get_question_id(record, index)
        question = get_question_text(record)

        assert question is not None

        print(f"[{index}/{len(records)}] {question_id}: {question}")

        try:
            plan = router.route(question)
            results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "status": "success",
                    "gold": collect_gold(record),
                    "plan": plan.model_dump(mode="json"),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "status": "error",
                    "gold": collect_gold(record),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    output_path = make_run_path(args.output_dir)

    output_payload = {
        "run_type": "kg_router_batch_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "env_file": str(env_path),
        "dataset": str(dataset_path),
        "total_questions": len(records),
        "successes": sum(1 for item in results if item["status"] == "success"),
        "errors": sum(1 for item in results if item["status"] == "error"),
        "results": results,
    }

    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Saved router plans:", output_path)
    print("Successes:", output_payload["successes"])
    print("Errors:", output_payload["errors"])


if __name__ == "__main__":
    main()
