"""Batch-evaluate modular and role-aware KG retrieval configurations.

For each question the MENTIONS router is called once. The resulting
``KGMentionsPlan`` is replayed unchanged across ``mentions_only``,
``mentions_weighted``, and ``mentions_descendants`` so those modes form a
controlled ablation. ``planned_role_aware`` uses its own richer router.

Gold annotations are never passed to retrieval. They are used only after a
ranking has been produced, to calculate section- and document-level metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agentic_rag.agent.output import KGMentionsPlan
from agentic_rag.agent.services.llm_service import LLMService
from agentic_rag.config.manager import LLMConfig, LLMProvider, NodePromptConfig
from agentic_rag.evaluation.kg_batch import (
    aggregate_metric_rows,
    build_metric_rows,
    candidate_diagnostics,
    evaluate_rankings,
    gold_section_keys,
    index_dataset_questions,
    load_coverage_evaluation_sets,
    load_retrieval_dataset,
    section_keys_from_results,
)
from agentic_rag.kg.client import Neo4jKGClient
from agentic_rag.kg.pipeline import build_modular_kg_pipeline
from agentic_rag.kg.retriever import KGParameterizedRetriever
from agentic_rag.kg.router import KGMentionsRouter, KGStructuredRouter
from agentic_rag.kg.tools import KGSectionTools
from agentic_rag.managers.llm_manager import LLMManager


DEFAULT_DATASET = Path("tests/data/subset_test_en_cm_syn.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/kg_retrieval")
MODULAR_MODES = (
    "mentions_only",
    "mentions_weighted",
    "mentions_descendants",
)
ALL_MODES = (*MODULAR_MODES, "planned_role_aware")


class StaticMentionsRouter:
    """Return one already-generated MENTIONS plan without invoking an LLM."""

    def __init__(self, plan: KGMentionsPlan) -> None:
        self.plan = plan

    def route(
        self,
        question: str,
        *,
        config: Any | None = None,
    ) -> KGMentionsPlan:
        del question, config
        return self.plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KG retrieval configurations on a gold dataset."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--coverage-artifact",
        type=Path,
        default=None,
        help=(
            "Enriched gold-resolution artifact. If omitted, the newest "
            "gold_resolution_enriched.json under artifacts/kg_retrieval is used."
        ),
    )
    parser.add_argument(
        "--nodes-config",
        type=Path,
        default=Path("configs/graphs/rag_agent.yaml"),
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--mode",
        action="append",
        choices=ALL_MODES,
        dest="modes",
        help="Mode to execute. Repeat the option to select multiple modes.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        dest="question_ids",
        help="Question ID to execute. Repeat to select multiple questions.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-after", default=None)
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hierarchy-max-depth", type=int, default=3)
    parser.add_argument("--descendants-per-seed", type=int, default=5)
    parser.add_argument("--max-expanded-rows", type=int, default=1000)
    parser.add_argument(
        "--advanced-ranking-mode",
        choices=("concept_match", "weighted_match"),
        default="weighted_match",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--advanced-multiple-facets-context-aware-merge",
        action="store_true",
        help=(
            "For planned_role_aware, rerank multiple_facets results using "
            "document-level support from context calls. Disabled by default."
        ),
    )
    parser.add_argument("--include-summary-sections", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def find_latest_coverage_artifact(output_root: Path) -> Path:
    candidates = sorted(
        output_root.expanduser().resolve().glob(
            "kg_gold_validation_*/gold_resolution_enriched.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No gold_resolution_enriched.json found. Pass --coverage-artifact "
            "or run augment_kg_gold_coverage.py first."
        )
    return candidates[0]


def build_llm_service(
    *,
    model_name: str,
    nodes_config_path: Path,
) -> tuple[LLMService, Any]:
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
    service = LLMService(
        llm_manager=llm_manager,
        nodes_prompt_config=NodePromptConfig(
            config=resolve_file(nodes_config_path, "Nodes configuration"),
            prompts="agentic_rag.agent.prompts",
        ),
    )
    return service, llm_manager.router_config


def make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"kg_retrieval_eval_{timestamp}"


def select_question_ids(
    all_ids: Sequence[str],
    *,
    requested: Sequence[str] | None,
    start_after: str | None,
    limit: int | None,
) -> list[str]:
    selected = list(all_ids)
    if requested:
        requested_set = set(requested)
        missing = [item for item in requested if item not in all_ids]
        if missing:
            raise KeyError(f"Unknown question IDs: {missing}")
        selected = [item for item in all_ids if item in requested_set]
    if start_after:
        if start_after not in selected:
            raise KeyError(f"--start-after question not selected: {start_after}")
        selected = selected[selected.index(start_after) + 1 :]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return selected


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL object at {path}:{line_number}")
        rows.append(payload)
    return rows


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def key_results_from_modular(run: Any, attribute: str) -> list[Any]:
    candidates = getattr(run, attribute)
    return [item.section for item in candidates]


def build_query_record(
    *,
    question_id: str,
    question: str,
    mode: str,
    run: Any,
    gold_sections: Sequence[Any],
    raw_results: Sequence[Any],
    expanded_results: Sequence[Any] | None,
    final_results: Sequence[Any],
) -> dict[str, Any]:
    raw_keys = section_keys_from_results(raw_results)
    expanded_keys = (
        section_keys_from_results(expanded_results)
        if expanded_results is not None
        else None
    )
    final_keys = section_keys_from_results(final_results)
    evaluation = evaluate_rankings(
        gold_sections=gold_sections,
        section_ranking=final_keys,
    )

    diagnostics: dict[str, Any] = {
        "raw_candidates": candidate_diagnostics(gold_sections, raw_keys),
        "final_ranking": candidate_diagnostics(gold_sections, final_keys),
    }
    if expanded_keys is not None:
        diagnostics["expanded_candidates"] = candidate_diagnostics(
            gold_sections,
            expanded_keys,
        )

    return {
        "question_id": question_id,
        "question": question,
        "mode": mode,
        "status": run.status,
        "error": run.error,
        "latency_ms": run.latency_ms,
        "returned_count": len(final_results),
        "plan": (
            run.plan.model_dump(mode="json") if run.plan is not None else None
        ),
        "gold_sections": [item.to_dict() for item in gold_sections],
        "raw_candidates": [item.to_dict() for item in raw_keys],
        "expanded_candidates": (
            [item.to_dict() for item in expanded_keys]
            if expanded_keys is not None
            else None
        ),
        "final_ranking": evaluation["section_ranking"],
        "document_ranking": evaluation["document_ranking"],
        "candidate_diagnostics": diagnostics,
        "metrics": {
            "section": evaluation["section"],
            "document": evaluation["document"],
        },
        "retrieval_trace": run.model_dump(mode="json"),
    }


def write_metric_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fixed = [
        "question_id",
        "mode",
        "level",
        "view",
        "status",
        "latency_ms",
        "returned_count",
    ]
    metric_names = sorted(
        {key for row in rows for key in row.keys()} - set(fixed)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fixed, *metric_names])
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_csv(
    path: Path,
    aggregates: Sequence[Mapping[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    metric_names: set[str] = set()
    for aggregate in aggregates:
        means = aggregate.get("means") or {}
        metric_names.update(means)
        row = {
            "mode": aggregate["mode"],
            "level": aggregate["level"],
            "view": aggregate["view"],
            "query_count": aggregate["query_count"],
            "mean_latency_ms": aggregate.get("mean_latency_ms"),
            **means,
        }
        rows.append(row)
    fieldnames = [
        "mode",
        "level",
        "view",
        "query_count",
        "mean_latency_ms",
        *sorted(metric_names),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_query_summary(record: Mapping[str, Any]) -> None:
    section_metrics = record["metrics"]["section"]
    diagnostics = record["candidate_diagnostics"]["final_ranking"]
    print(
        f"{record['question_id']} | {record['mode']} | "
        f"status={record['status']} | returned={record['returned_count']} | "
        f"hit@10={section_metrics['hit@10']:.0f} | "
        f"recall@10={section_metrics['recall@10']:.3f} | "
        f"rr@10={section_metrics['reciprocal_rank@10']:.3f} | "
        f"best_gold_rank={diagnostics['best_gold_rank']}"
    )


def main() -> None:
    args = parse_args()
    env_path = resolve_file(args.env_file, "Environment file")
    load_dotenv(dotenv_path=env_path, override=False)
    dataset_path = resolve_file(args.dataset, "Dataset")
    dataset = load_retrieval_dataset(dataset_path)
    indexed_questions = index_dataset_questions(dataset)
    all_question_ids = list(indexed_questions)
    selected_ids = select_question_ids(
        all_question_ids,
        requested=args.question_ids,
        start_after=args.start_after,
        limit=args.limit,
    )
    modes = list(dict.fromkeys(args.modes or ALL_MODES))

    output_root = args.output_root.expanduser().resolve()
    coverage_path = (
        resolve_file(args.coverage_artifact, "Coverage artifact")
        if args.coverage_artifact is not None
        else find_latest_coverage_artifact(output_root)
    )
    evaluation_sets = load_coverage_evaluation_sets(coverage_path)

    run_id = args.run_id or make_run_id()
    output_dir = output_root / run_id
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    plans_path = output_dir / "mentions_plans.jsonl"
    queries_path = output_dir / "queries.jsonl"
    manifest_path = output_dir / "manifest.json"

    existing_plan_rows = load_jsonl(plans_path) if args.resume else []
    existing_query_rows = load_jsonl(queries_path) if args.resume else []
    plan_by_question = {
        str(row["question_id"]): KGMentionsPlan.model_validate(row["plan"])
        for row in existing_plan_rows
    }
    completed = {
        (str(row["question_id"]), str(row["mode"]))
        for row in existing_query_rows
    }

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_type": "kg_retrieval_batch_evaluation",
        "dataset": str(dataset_path),
        "coverage_artifact": str(coverage_path),
        "environment_file": str(env_path),
        "configuration": {
            "model": args.model,
            "modes": modes,
            "question_ids": selected_ids,
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "hierarchy_max_depth": args.hierarchy_max_depth,
            "descendants_per_seed": args.descendants_per_seed,
            "max_expanded_rows": args.max_expanded_rows,
            "advanced_ranking_mode": args.advanced_ranking_mode,
            "rrf_k": args.rrf_k,
            "advanced_multiple_facets_context_aware_merge": (
                args.advanced_multiple_facets_context_aware_merge
            ),
            "exclude_summary_sections": not args.include_summary_sections,
            "document_filtering": None,
            "gold_annotations_used_for_retrieval": False,
            "mentions_plan_reused_across_modular_modes": True,
        },
        "status": "running",
    }
    write_json(manifest_path, manifest)

    llm_service, router_config = build_llm_service(
        model_name=args.model,
        nodes_config_path=args.nodes_config,
    )
    mentions_router = KGMentionsRouter(
        llm_service=llm_service,
        router_config=router_config,
    )
    advanced_router = KGStructuredRouter(
        llm_service=llm_service,
        router_config=router_config,
    )

    started = time.perf_counter()
    failure_count = 0
    with Neo4jKGClient.from_env(env_path=env_path) as client:
        tools = KGSectionTools(client)
        advanced_retriever = KGParameterizedRetriever(
            router=advanced_router,
            tools=tools,
            candidate_k=args.candidate_k,
            final_k=args.top_k,
            ranking_mode=args.advanced_ranking_mode,
            rrf_k=args.rrf_k,
            hierarchy_max_depth=args.hierarchy_max_depth,
            exclude_summary_sections=not args.include_summary_sections,
            multiple_facets_context_aware_merge=(
                args.advanced_multiple_facets_context_aware_merge
            ),
        )

        for question_id in selected_ids:
            question_record = indexed_questions[question_id]
            question = str(question_record["question"]).strip()
            gold_sections = gold_section_keys(question_record)

            needs_modular = any(
                mode in MODULAR_MODES
                and (question_id, mode) not in completed
                for mode in modes
            )
            if needs_modular and question_id not in plan_by_question:
                try:
                    plan = mentions_router.route(question)
                except Exception as exc:
                    plan = None
                    failure_count += 1
                    error_record = {
                        "question_id": question_id,
                        "question": question,
                        "stage": "mentions_planning",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    append_jsonl(output_dir / "errors.jsonl", error_record)
                    if args.fail_fast:
                        raise
                if plan is not None:
                    plan_by_question[question_id] = plan
                    append_jsonl(
                        plans_path,
                        {
                            "question_id": question_id,
                            "question": question,
                            "plan": plan.model_dump(mode="json"),
                        },
                    )

            for mode in modes:
                if (question_id, mode) in completed:
                    continue
                try:
                    if mode in MODULAR_MODES:
                        plan = plan_by_question.get(question_id)
                        if plan is None:
                            raise RuntimeError(
                                "No reusable MENTIONS plan available"
                            )
                        pipeline = build_modular_kg_pipeline(
                            mode,
                            router=StaticMentionsRouter(plan),
                            tools=tools,
                            client=client,
                            candidate_k=args.candidate_k,
                            final_k=args.top_k,
                            exclude_summary_sections=(
                                not args.include_summary_sections
                            ),
                            hierarchy_max_depth=args.hierarchy_max_depth,
                            descendants_per_seed=args.descendants_per_seed,
                            max_expanded_rows=args.max_expanded_rows,
                        )
                        run = pipeline.retrieve(question)
                        raw_results = key_results_from_modular(
                            run, "raw_candidates"
                        )
                        expanded_results = key_results_from_modular(
                            run, "expanded_candidates"
                        )
                        final_results = key_results_from_modular(run, "results")
                    else:
                        run = advanced_retriever.retrieve(question)
                        raw_results = [
                            result
                            for execution in run.tool_executions
                            for result in execution.results
                        ]
                        expanded_results = None
                        final_results = list(run.results)

                    record = build_query_record(
                        question_id=question_id,
                        question=question,
                        mode=mode,
                        run=run,
                        gold_sections=gold_sections,
                        raw_results=raw_results,
                        expanded_results=expanded_results,
                        final_results=final_results,
                    )
                    append_jsonl(queries_path, record)
                    existing_query_rows.append(record)
                    completed.add((question_id, mode))
                    print_query_summary(record)
                except Exception as exc:
                    failure_count += 1
                    error_record = {
                        "question_id": question_id,
                        "question": question,
                        "mode": mode,
                        "stage": "retrieval_or_evaluation",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    append_jsonl(output_dir / "errors.jsonl", error_record)
                    print(
                        f"{question_id} | {mode} | ERROR | "
                        f"{error_record['error']}"
                    )
                    if args.fail_fast:
                        raise

    all_query_rows = load_jsonl(queries_path)
    selected_set = set(selected_ids)
    selected_mode_set = set(modes)
    current_rows = [
        row
        for row in all_query_rows
        if row.get("question_id") in selected_set
        and row.get("mode") in selected_mode_set
    ]
    metric_rows = build_metric_rows(current_rows, evaluation_sets)
    aggregates = aggregate_metric_rows(metric_rows) if metric_rows else []

    write_metric_csv(output_dir / "per_query_metrics.csv", metric_rows)
    write_aggregate_csv(output_dir / "aggregate_metrics.csv", aggregates)
    status_counts: dict[str, int] = {}
    for row in current_rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "run_id": run_id,
        "question_count_requested": len(selected_ids),
        "mode_count_requested": len(modes),
        "expected_query_mode_runs": len(selected_ids) * len(modes),
        "completed_query_mode_runs": len(current_rows),
        "status_counts": status_counts,
        "failure_count": failure_count,
        "elapsed_seconds": time.perf_counter() - started,
        "aggregates": aggregates,
    }
    write_json(output_dir / "summary.json", summary)

    manifest["status"] = "completed_with_errors" if failure_count else "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_file"] = str(output_dir / "summary.json")
    write_json(manifest_path, manifest)

    print()
    print("Batch evaluation complete")
    print("Output:", output_dir)
    print("Completed query/mode runs:", len(current_rows))
    print("Failures:", failure_count)


if __name__ == "__main__":
    main()
