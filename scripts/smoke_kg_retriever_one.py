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
from agentic_rag.kg.client import Neo4jKGClient
from agentic_rag.kg.retriever import KGParameterizedRetriever, KGRetrievalRun
from agentic_rag.kg.router import KGStructuredRouter
from agentic_rag.kg.tools import KGSectionTools
from agentic_rag.managers.llm_manager import LLMManager


DEFAULT_DATASET = Path("tests/data/subset_test_en_cm_syn.json")
DEFAULT_QUESTION_ID = "CM_01"
DEFAULT_OUTPUT_ROOT = Path("artifacts/kg_retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one complete KG retrieval request: structured router, "
            "Neo4j tools, and Reciprocal Rank Fusion."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Environment file containing OpenAI and Neo4j credentials.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Dataset used to resolve --question-id.",
    )
    parser.add_argument(
        "--question-id",
        default=DEFAULT_QUESTION_ID,
        help="Question identifier to load from the dataset.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help=(
            "Optional custom question. When supplied, it overrides "
            "--question-id and does not use gold annotations."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used by the structured router.",
    )
    parser.add_argument(
        "--nodes-config",
        type=Path,
        default=Path("configs/graphs/rag_agent.yaml"),
        help="YAML file containing the registered LLM nodes.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=15,
        help="Maximum results requested from each routed KG tool.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of final RRF-fused results.",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=("concept_match", "weighted_match"),
        default="weighted_match",
        help="Deterministic ranking mode used inside each KG tool.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion constant.",
    )
    parser.add_argument(
        "--include-summary-sections",
        action="store_true",
        help="Do not exclude summary sections such as 'What to do'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for the retrieval trace artifact.",
    )
    return parser.parse_args()


def resolve_existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_environment(env_file: Path) -> Path:
    resolved = resolve_existing_file(env_file, "Environment file")
    load_dotenv(dotenv_path=resolved, override=False)
    return resolved


def load_dataset_question(
    dataset_path: Path,
    question_id: str,
) -> tuple[str, dict[str, Any]]:
    resolved = resolve_existing_file(dataset_path, "Dataset")
    payload = json.loads(resolved.read_text(encoding="utf-8"))

    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(
            f"Dataset must contain a 'questions' list: {resolved}"
        )

    for record in questions:
        if str(record.get("id", "")).strip() == question_id:
            question = str(record.get("question", "")).strip()
            if not question:
                raise ValueError(
                    f"Question {question_id!r} has empty text"
                )
            return question, record

    available_ids = [
        str(record.get("id"))
        for record in questions
        if record.get("id") is not None
    ]
    raise KeyError(
        f"Question id {question_id!r} not found in {resolved}. "
        f"Available ids: {available_ids}"
    )


def build_router(
    *,
    model_name: str,
    nodes_config_path: Path,
) -> KGStructuredRouter:
    resolved_nodes_config = resolve_existing_file(
        nodes_config_path,
        "Nodes configuration",
    )

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
        config=resolved_nodes_config,
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


def make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"kg_retriever_smoke_{timestamp}"


def save_artifact(
    *,
    run: KGRetrievalRun,
    run_id: str,
    question_id: str,
    question_source: str,
    env_path: Path,
    dataset_path: Path | None,
    model: str,
    output_root: Path,
) -> Path:
    output_dir = output_root.expanduser().resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "retrieval_run.json"

    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_type": "kg_retriever_single_question_smoke",
        "question_id": question_id,
        "question_source": question_source,
        "dataset": str(dataset_path) if dataset_path is not None else None,
        "environment_file": str(env_path),
        "configuration": {
            "model": model,
            "candidate_k": run.candidate_k,
            "final_k": run.final_k,
            "ranking_mode": run.ranking_mode,
            "rrf_k": run.rrf_k,
            "exclude_summary_sections": run.exclude_summary_sections,
            "document_filtering": None,
            "gold_annotations_used_for_retrieval": False,
            "exact_section_text_saved": True,
        },
        "retrieval": run.model_dump(mode="json"),
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def print_run(run: KGRetrievalRun) -> None:
    print()
    print("Question:")
    print(run.question)
    print()
    print("Status:", run.status)
    print(f"Total latency: {run.latency_ms:.1f} ms")

    if run.error:
        print("Error:", run.error)

    print()
    print("Structured retrieval plan:")
    if run.plan is None:
        print("<none>")
    else:
        print(run.plan.model_dump_json(indent=2))

    print()
    print("Tool executions:")
    if not run.tool_executions:
        print("<none>")

    for execution in run.tool_executions:
        call = execution.call
        print(
            f"[{execution.call_index}] {call.tool} "
            f"status={execution.status} "
            f"returned={execution.returned_count} "
            f"latency={execution.latency_ms:.1f} ms"
        )
        print(
            "    terms=",
            call.terms,
            "require_all=",
            call.require_all,
        )
        if execution.error:
            print("    error=", execution.error)

        for result in execution.results[:5]:
            print(
                f"    {result.rank}. {result.document_id} | "
                f"{result.printed_section_id or result.section_id} | "
                f"{result.title} | score={result.score}"
            )

    print()
    print("Final RRF ranking:")
    if not run.results:
        print("<no results>")

    for result in run.results:
        source_trace = ", ".join(
            f"call={item.call_index}:rank={item.source_rank}"
            for item in result.contributions
        )
        print(
            f"{result.rank}. {result.document_id} | "
            f"{result.printed_section_id or result.section_id} | "
            f"{result.title} | rrf={result.fusion_score:.8f} | "
            f"best_source_rank={result.best_source_rank} | "
            f"{source_trace}"
        )


def main() -> None:
    args = parse_args()
    env_path = load_environment(args.env_file)

    if args.question is not None:
        question = str(args.question).strip()
        if not question:
            raise ValueError("--question must be non-empty")
        question_id = "custom"
        question_source = "custom"
        dataset_path = None
    else:
        dataset_path = resolve_existing_file(args.dataset, "Dataset")
        question_id = str(args.question_id).strip()
        if not question_id:
            raise ValueError("--question-id must be non-empty")
        question, _ = load_dataset_question(
            dataset_path,
            question_id,
        )
        question_source = "dataset"

    router = build_router(
        model_name=args.model,
        nodes_config_path=args.nodes_config,
    )

    run_id = make_run_id()

    with Neo4jKGClient.from_env(
        env_path=env_path,
        verify_connectivity=True,
    ) as client:
        tools = KGSectionTools(client)
        retriever = KGParameterizedRetriever(
            router=router,
            tools=tools,
            candidate_k=args.candidate_k,
            final_k=args.top_k,
            ranking_mode=args.ranking_mode,
            rrf_k=args.rrf_k,
            exclude_summary_sections=(
                not args.include_summary_sections
            ),
        )
        run = retriever.retrieve(question)

    print("Environment file:", env_path)
    print("Model:", args.model)
    print("Question id:", question_id)
    print_run(run)

    output_path = save_artifact(
        run=run,
        run_id=run_id,
        question_id=question_id,
        question_source=question_source,
        env_path=env_path,
        dataset_path=dataset_path,
        model=args.model,
        output_root=args.output_root,
    )

    print()
    print("Saved:", output_path)

    if run.status in {"router_error", "execution_error"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
