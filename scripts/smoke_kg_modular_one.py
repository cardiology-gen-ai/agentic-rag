from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agentic_rag.agent.services.llm_service import LLMService
from agentic_rag.config.manager import LLMConfig, LLMProvider, NodePromptConfig
from agentic_rag.kg.client import Neo4jKGClient
from agentic_rag.kg.pipeline import (
    ModularKGRetrievalRun,
    build_modular_kg_pipeline,
)
from agentic_rag.kg.router import KGMentionsRouter
from agentic_rag.kg.tools import KGSectionTools
from agentic_rag.managers.llm_manager import LLMManager


DEFAULT_DATASET = Path("tests/data/subset_test_en_cm_syn.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/kg_retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one modular KG retrieval baseline request."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--question-id", default="CM_01")
    parser.add_argument("--question", default=None)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--nodes-config",
        type=Path,
        default=Path("configs/graphs/rag_agent.yaml"),
    )
    parser.add_argument(
        "--mode",
        choices=(
            "mentions_only",
            "mentions_weighted",
            "mentions_descendants",
            "mentions_same_as",
            "mentions_umls_safe",
            "mentions_same_as_rescue",
            "mentions_umls_safe_rescue",
        ),
        default="mentions_only",
    )
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hierarchy-max-depth", type=int, default=3)
    parser.add_argument("--descendants-per-seed", type=int, default=3)
    parser.add_argument("--max-expanded-rows", type=int, default=1000)
    parser.add_argument("--include-summary-sections", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def resolve_existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_dataset_question(
    dataset_path: Path,
    question_id: str,
) -> tuple[str, dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Dataset must contain a 'questions' list")
    for record in questions:
        if str(record.get("id", "")).strip() == question_id:
            question = str(record.get("question", "")).strip()
            if not question:
                raise ValueError(f"Question {question_id!r} is empty")
            return question, record
    raise KeyError(f"Question id {question_id!r} not found")


def build_router(
    *,
    model_name: str,
    nodes_config_path: Path,
) -> KGMentionsRouter:
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
    llm_service = LLMService(
        llm_manager=llm_manager,
        nodes_prompt_config=NodePromptConfig(
            config=resolve_existing_file(
                nodes_config_path,
                "Nodes configuration",
            ),
            prompts="agentic_rag.agent.prompts",
        ),
    )
    return KGMentionsRouter(
        llm_service=llm_service,
        router_config=llm_manager.router_config,
    )


def make_run_id(mode: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"kg_{mode}_smoke_{timestamp}"


def print_run(run: ModularKGRetrievalRun) -> None:
    print()
    print("Question:")
    print(run.question)
    print()
    print("Mode:", run.mode)
    print("Status:", run.status)
    print(f"Latency: {run.latency_ms:.1f} ms")
    print("Retrieval unit:", run.retrieval_unit, run.unit_scope)
    if run.error:
        print("Error:", run.error)

    print()
    print("MENTIONS plan:")
    print(run.plan.model_dump_json(indent=2) if run.plan else "<none>")

    print()
    print("Raw candidates:", len(run.raw_candidates))
    for item in run.raw_candidates[:10]:
        section = item.section
        print(
            f"  {item.source_rank}. {section.document_id} | "
            f"{section.printed_section_id or section.section_id} | "
            f"level={section.level} | {section.title} | score={section.score}"
        )

    print()
    print("Expanded candidates:", len(run.expanded_candidates))
    print("Final ranking:")
    if not run.results:
        print("<no results>")
    for item in run.results:
        section = item.section
        expansion = "direct"
        if not item.direct:
            expansion = (
                f"descendant seed={item.seed_uid} "
                f"distance={item.graph_distance}"
            )
        print(
            f"  {item.final_rank}. {section.document_id} | "
            f"{section.printed_section_id or section.section_id} | "
            f"level={section.level} | {section.title} | {expansion}"
        )


def save_artifact(
    run: ModularKGRetrievalRun,
    *,
    output_root: Path,
    question_id: str,
    dataset_path: Path | None,
    model: str,
) -> Path:
    run_id = make_run_id(run.mode)
    output_dir = output_root.expanduser().resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "retrieval_run.json"
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_type": "kg_modular_single_question_smoke",
        "question_id": question_id,
        "dataset": str(dataset_path) if dataset_path else None,
        "model": model,
        "gold_annotations_used_for_retrieval": False,
        "retrieval": run.model_dump(mode="json"),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()
    env_path = resolve_existing_file(args.env_file, "Environment file")
    load_dotenv(dotenv_path=env_path, override=False)

    if args.question is not None:
        question = str(args.question).strip()
        if not question:
            raise ValueError("--question must be non-empty")
        question_id = "custom"
        dataset_path = None
    else:
        dataset_path = resolve_existing_file(args.dataset, "Dataset")
        question_id = str(args.question_id).strip()
        question, _ = load_dataset_question(dataset_path, question_id)

    router = build_router(
        model_name=args.model,
        nodes_config_path=args.nodes_config,
    )

    with Neo4jKGClient.from_env(env_path=env_path) as client:
        tools = KGSectionTools(client)
        pipeline = build_modular_kg_pipeline(
            args.mode,
            router=router,
            tools=tools,
            client=client,
            candidate_k=args.candidate_k,
            final_k=args.top_k,
            exclude_summary_sections=(not args.include_summary_sections),
            hierarchy_max_depth=args.hierarchy_max_depth,
            descendants_per_seed=args.descendants_per_seed,
            max_expanded_rows=args.max_expanded_rows,
        )
        run = pipeline.retrieve(question)

    print("Environment file:", env_path)
    print("Model:", args.model)
    print("Question id:", question_id)
    print_run(run)

    output_path = save_artifact(
        run,
        output_root=args.output_root,
        question_id=question_id,
        dataset_path=dataset_path,
        model=args.model,
    )
    print()
    print("Saved:", output_path)

    if run.status in {"router_error", "execution_error"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
