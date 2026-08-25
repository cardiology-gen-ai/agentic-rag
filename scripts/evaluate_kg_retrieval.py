"""Batch-evaluate modular and role-aware KG retrieval configurations.

For each question the MENTIONS router is called once. The resulting
``KGMentionsPlan`` is replayed unchanged across ``mentions_only``, the
controlled lexical/embedding seeded MENTIONS modes,
``mentions_descendants``, ``mentions_same_as``, ``mentions_umls_safe``, ISA artifact,
and frozen non-hier RAW/SAFE artifact modes so those modes form controlled ablations.
``planned_role_aware`` uses its own richer router.

Gold annotations are never passed to retrieval. They are used only after a
ranking has been produced, to calculate section- and document-level metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    aggregate_metric_rows_by_group,
    build_question_group_membership,
    build_metric_rows,
    candidate_diagnostics,
    evaluate_rankings,
    gold_section_keys,
    index_dataset_questions,
    load_coverage_evaluation_sets,
    load_retrieval_dataset,
    section_keys_from_results,
)
from agentic_rag.kg.experiment_scope import (
    normalize_document_scope,
    validate_result_document_scope,
    validate_selected_gold_document_scope,
)
from agentic_rag.kg.client import Neo4jKGClient
from agentic_rag.kg.concept_seeders import EmbeddingConceptSeeder
from agentic_rag.kg.pipeline import build_modular_kg_pipeline
from agentic_rag.kg.retriever import KGParameterizedRetriever
from agentic_rag.kg.router import KGMentionsRouter, KGStructuredRouter
from agentic_rag.kg.tools import KGSectionTools
from agentic_rag.managers.llm_manager import LLMManager


DEFAULT_DATASET = Path("tests/data/subset_test_en_cm_syn.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/kg_retrieval")
MODULAR_MODES = (
    "mentions_only",
    "mentions_lexical_seeded",
    "mentions_embedding_seeded",
    "mentions_embedding_seeded_similarity_weighted",
    "mentions_lexical_semantic_best_channel",
    "mentions_descendants",
    "mentions_same_as",
    "mentions_umls_safe",
    "mentions_isa_forward_artifact",
    "mentions_isa_forward_artifact_strict_direct_first",
    "mentions_isa_semantic_safe_rerank",
    "mentions_nonhier_artifact_raw",
    "mentions_nonhier_artifact_safe",
    "mentions_nonhier_artifact_raw_strict",
    "mentions_nonhier_artifact_safe_strict",
    "mentions_nonhier_artifact_raw_strict_direct_first",
    "mentions_nonhier_artifact_safe_strict_direct_first",
    "mentions_nonhier_artifact_safe_strict_direct_first_frozen",
    "mentions_nonhier_artifact_safe_strict_direct_first_rescue",
    "mentions_same_as_rescue",
    "mentions_umls_safe_rescue",
    "mentions_direct_balanced",
    "mentions_bridge_sa_top5",
    "mentions_direct_bridge_sa_top3",
    "mentions_direct_bridge_sa_top5",
    "mentions_direct_bridge_sa_top10",
    "semantic_weighted_direct_balanced",
    "semantic_weighted_bridge_sa_top5",
    "semantic_weighted_direct_bridge_sa_top3",
    "semantic_weighted_direct_bridge_sa_top5",
    "semantic_weighted_direct_bridge_sa_top10",
    "semantic_weighted_pool_union_direct_bridge_sa_top3",
    "semantic_weighted_pool_rrf_direct_bridge_sa_top3",
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
    parser.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        help=(
            "Restrict KG retrieval to this Document.doc_id. Repeat the option "
            "for a controlled multi-document corpus. If omitted, legacy "
            "all-document behaviour is preserved."
        ),
    )
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--graph-candidate-k", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concept-embedding-model", default=None)
    parser.add_argument("--concepts-per-term", type=int, default=3)
    parser.add_argument(
        "--concept-embedding-cache",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--mentions-plans-file",
        type=Path,
        default=None,
        help=(
            "JSONL file with existing question_id/question/plan rows. When "
            "supplied, selected modular questions use these plans and do not "
            "invoke the MENTIONS router."
        ),
    )
    parser.add_argument("--hierarchy-max-depth", type=int, default=3)
    parser.add_argument(
        "--isa-connections",
        type=Path,
        default=None,
        help=(
            "Frozen data-etl collapsed-connections JSON used only by "
            "mentions_isa_forward_artifact modes."
        ),
    )
    parser.add_argument(
        "--isa-max-depth",
        type=int,
        default=1,
        help="Forward ISA hops for the artifact ablation (v1 recommendation: 1).",
    )
    parser.add_argument(
        "--nonhier-raw-artifact",
        type=Path,
        default=None,
        help=(
            "Frozen data-etl nonhier_semantic_raw_v1.json used by RAW "
            "non-hier artifact modes (v1 and strict-v2)."
        ),
    )
    parser.add_argument(
        "--nonhier-safe-artifact",
        type=Path,
        default=None,
        help=(
            "Frozen data-etl nonhier_semantic_safe_v1.json used by SAFE "
            "non-hier artifact modes (v1 and strict-v2)."
        ),
    )
    parser.add_argument(
        "--connection-config",
        type=Path,
        default=None,
        help=(
            "JSON configuration for frozen DIRECT/BRIDGE connection ablation "
            "modes."
        ),
    )
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
    parser.add_argument(
        "--advanced-multiple-facets-context-candidate-injection",
        action="store_true",
        help=(
            "For planned_role_aware, when context-aware multiple_facets is "
            "enabled, allow strong context candidates to compete with facet "
            "results. Disabled by default."
        ),
    )
    parser.add_argument(
        "--advanced-same-section-anchor-fallback",
        action="store_true",
        help=(
            "For planned_role_aware, when same_section produces no final "
            "results, use anchor-term lexical evidence to select fallback "
            "results from successful context calls. Disabled by default."
        ),
    )
    parser.add_argument(
        "--advanced-same-section-anchor-rescue",
        action="store_true",
        help=(
            "For planned_role_aware, merge strong anchor-sensitive context "
            "candidates into same_section results even when normal "
            "same_section returns non-empty results. Disabled by default."
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


def load_mentions_plans(path: Path) -> dict[str, KGMentionsPlan]:
    rows = load_jsonl(path)
    plans: dict[str, KGMentionsPlan] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(
                f"Mentions plans file row is missing question_id: {path}"
            )
        if question_id in plans:
            raise ValueError(
                f"Duplicate question_id in mentions plans file: {question_id}"
            )

        plan_payload = row.get("plan")
        if plan_payload is None:
            plan_payload = {
                "terms": row.get("terms"),
                "require_all": row.get("require_all", False),
            }
        plans[question_id] = KGMentionsPlan.model_validate(plan_payload)

    return plans


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
        section_ranking=final_results,
    )

    diagnostics: dict[str, Any] = {
        "raw_candidates": candidate_diagnostics(
            gold_sections,
            raw_results,
        ),
        "final_ranking": candidate_diagnostics(
            gold_sections,
            final_results,
        ),
    }
    if expanded_keys is not None:
        diagnostics["expanded_candidates"] = candidate_diagnostics(
            gold_sections,
            expanded_results,
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
        "concept_seeds": [
            seed.model_dump(mode="json")
            for seed in getattr(run, "concept_seeds", [])
        ],
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



def write_group_aggregate_csv(
    path: Path,
    aggregates: Sequence[Mapping[str, Any]],
) -> None:
    """Write evaluation-group aggregate metrics without altering base reports."""
    rows: list[dict[str, Any]] = []
    metric_names: set[str] = set()

    for aggregate in aggregates:
        means = aggregate.get("means") or {}
        metric_names.update(means)

        rows.append(
            {
                "group": aggregate["group"],
                "mode": aggregate["mode"],
                "level": aggregate["level"],
                "view": aggregate["view"],
                "query_count": aggregate["query_count"],
                "mean_latency_ms": aggregate.get("mean_latency_ms"),
                **means,
            }
        )

    fieldnames = [
        "group",
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


def build_concept_seed_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_question: dict[str, dict[str, Any]] = {}

    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            continue
        entry = by_question.setdefault(
            question_id,
            {
                "question_id": question_id,
                "question": row.get("question"),
                "lexical_seeds": [],
                "embedding_seeds": [],
            },
        )

        seeds = row.get("concept_seeds")
        if seeds is None:
            trace = row.get("retrieval_trace") or {}
            if isinstance(trace, Mapping):
                seeds = trace.get("concept_seeds")
        if not isinstance(seeds, list):
            continue

        for seed in seeds:
            if not isinstance(seed, Mapping):
                continue
            method = str(seed.get("method") or "").strip()
            if method == "lexical":
                entry["lexical_seeds"].append(dict(seed))
            elif method == "embedding":
                entry["embedding_seeds"].append(dict(seed))

    diagnostics: list[dict[str, Any]] = []
    for entry in by_question.values():
        lexical = _deduplicate_seed_dicts(entry["lexical_seeds"])
        embedding = _deduplicate_seed_dicts(entry["embedding_seeds"])
        lexical_keys = {_seed_key(seed) for seed in lexical}
        embedding_keys = {_seed_key(seed) for seed in embedding}

        diagnostics.append(
            {
                "question_id": entry["question_id"],
                "question": entry["question"],
                "lexical_seeds": lexical,
                "embedding_seeds": embedding,
                "shared_concept_seeds": _seed_key_dicts(
                    lexical_keys & embedding_keys
                ),
                "lexical_only_seeds": _seed_key_dicts(
                    lexical_keys - embedding_keys
                ),
                "embedding_only_seeds": _seed_key_dicts(
                    embedding_keys - lexical_keys
                ),
            }
        )

    diagnostics.sort(key=lambda item: item["question_id"])
    return diagnostics


def _deduplicate_seed_dicts(
    seeds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for seed in seeds:
        key = _seed_key(seed)
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(seed))
    return output


def _seed_key(seed: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(seed.get("query_term") or "").casefold(),
        str(seed.get("concept_name") or "").casefold(),
    )


def _seed_key_dicts(keys: set[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "query_term": query_term,
            "concept_name": concept_name,
        }
        for query_term, concept_name in sorted(keys)
        if query_term and concept_name
    ]


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
    document_ids = normalize_document_scope(args.document_ids)
    validate_selected_gold_document_scope(
        indexed_questions,
        selected_ids,
        document_ids,
    )
    modes = list(dict.fromkeys(args.modes or ALL_MODES))
    if any(
        mode in modes
        for mode in {
            "mentions_isa_forward_artifact",
            "mentions_isa_forward_artifact_strict_direct_first",
            "mentions_isa_semantic_safe_rerank",
        }
    ):
        if args.isa_connections is None:
            raise ValueError(
                "--isa-connections is required when an ISA artifact mode "
                "is selected"
            )
        args.isa_connections = resolve_file(
            args.isa_connections,
            "ISA collapsed-connections artifact",
        )
        if args.isa_max_depth < 1:
            raise ValueError("--isa-max-depth must be >= 1")
    if any(
        mode in modes
        for mode in {
            "mentions_nonhier_artifact_raw",
            "mentions_nonhier_artifact_raw_strict",
            "mentions_nonhier_artifact_raw_strict_direct_first",
        }
    ):
        if args.nonhier_raw_artifact is None:
            raise ValueError(
                "--nonhier-raw-artifact is required when a RAW non-hier "
                "artifact mode is selected"
            )
        args.nonhier_raw_artifact = resolve_file(
            args.nonhier_raw_artifact,
            "Non-hier RAW retrieval artifact",
        )
    if any(
        mode in modes
        for mode in {
            "mentions_nonhier_artifact_safe",
            "mentions_nonhier_artifact_safe_strict",
            "mentions_nonhier_artifact_safe_strict_direct_first",
            "mentions_nonhier_artifact_safe_strict_direct_first_frozen",
            "mentions_nonhier_artifact_safe_strict_direct_first_rescue",
        }
    ):
        if args.nonhier_safe_artifact is None:
            raise ValueError(
                "--nonhier-safe-artifact is required when a SAFE non-hier "
                "artifact mode is selected"
            )
        args.nonhier_safe_artifact = resolve_file(
            args.nonhier_safe_artifact,
            "Non-hier SAFE retrieval artifact",
        )
    connection_modes = {
        "mentions_direct_balanced",
        "mentions_bridge_sa_top5",
        "mentions_direct_bridge_sa_top3",
        "mentions_direct_bridge_sa_top5",
        "mentions_direct_bridge_sa_top10",
        "semantic_weighted_direct_balanced",
        "semantic_weighted_bridge_sa_top5",
        "semantic_weighted_direct_bridge_sa_top3",
        "semantic_weighted_direct_bridge_sa_top5",
        "semantic_weighted_direct_bridge_sa_top10",
        "semantic_weighted_pool_union_direct_bridge_sa_top3",
        "semantic_weighted_pool_rrf_direct_bridge_sa_top3",
    }
    if any(mode in modes for mode in connection_modes):
        if args.connection_config is None:
            raise ValueError(
                "--connection-config is required when a frozen connection "
                "ablation mode is selected"
            )
        args.connection_config = resolve_file(
            args.connection_config,
            "Connection ablation configuration",
        )

    embedding_seed_modes = {
        "mentions_embedding_seeded",
        "mentions_embedding_seeded_similarity_weighted",
        "mentions_lexical_semantic_best_channel",
        "semantic_weighted_direct_balanced",
        "semantic_weighted_bridge_sa_top5",
        "semantic_weighted_direct_bridge_sa_top3",
        "semantic_weighted_direct_bridge_sa_top5",
        "semantic_weighted_direct_bridge_sa_top10",
        "semantic_weighted_pool_union_direct_bridge_sa_top3",
        "semantic_weighted_pool_rrf_direct_bridge_sa_top3",
    }
    if (
        any(mode in modes for mode in embedding_seed_modes)
        and not args.concept_embedding_model
    ):
        raise ValueError(
            "--concept-embedding-model is required when "
            "mentions_embedding_seeded is selected"
        )

    output_root = args.output_root.expanduser().resolve()
    coverage_path = (
        resolve_file(args.coverage_artifact, "Coverage artifact")
        if args.coverage_artifact is not None
        else find_latest_coverage_artifact(output_root)
    )
    evaluation_sets = load_coverage_evaluation_sets(coverage_path)

    supplied_plans_path = (
        resolve_file(args.mentions_plans_file, "Mentions plans file")
        if args.mentions_plans_file is not None
        else None
    )
    supplied_plan_by_question = (
        load_mentions_plans(supplied_plans_path)
        if supplied_plans_path is not None
        else {}
    )
    if supplied_plans_path is not None:
        missing_plan_ids = [
            question_id
            for question_id in selected_ids
            if question_id not in supplied_plan_by_question
        ]
        if missing_plan_ids:
            raise KeyError(
                "Selected question IDs missing from --mentions-plans-file: "
                f"{missing_plan_ids}"
            )

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
    if supplied_plans_path is not None:
        plan_by_question.update(supplied_plan_by_question)
        if not args.resume:
            for question_id in selected_ids:
                question = str(
                    indexed_questions[question_id]["question"]
                ).strip()
                append_jsonl(
                    plans_path,
                    {
                        "question_id": question_id,
                        "question": question,
                        "plan": plan_by_question[
                            question_id
                        ].model_dump(mode="json"),
                    },
                )
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
            "concept_embedding_model": args.concept_embedding_model,
            "concepts_per_term": args.concepts_per_term,
            "concept_embedding_cache": (
                str(args.concept_embedding_cache)
                if args.concept_embedding_cache is not None
                else None
            ),
            "concept_catalogue_size": None,
            "concept_catalogue_build_load_seconds": None,
            "concept_embedding_model_load_seconds": None,
            "concept_embedding_cache_loaded": None,
            "concept_embedding_cache_file": None,
            "hierarchy_max_depth": args.hierarchy_max_depth,
            "isa_connections": (
                str(args.isa_connections)
                if args.isa_connections is not None
                else None
            ),
            "isa_max_depth": args.isa_max_depth,
            "isa_direction": "forward_specific_to_general",
            "isa_uses_neo4j_umls_edges": False,
            "isa_includes_same_as": False,
            "isa_includes_nonhier_relations": False,
            "isa_v1_seed_match_policy": "permissive",
            "isa_strict_seed_match_policy": "exact_name_only",
            "isa_strict_ranking_policy": "direct_first_graph_second",
            "nonhier_raw_artifact": (
                str(args.nonhier_raw_artifact)
                if args.nonhier_raw_artifact is not None
                else None
            ),
            "nonhier_raw_artifact_sha256": (
                hashlib.sha256(args.nonhier_raw_artifact.read_bytes()).hexdigest()
                if args.nonhier_raw_artifact is not None
                else None
            ),
            "nonhier_safe_artifact": (
                str(args.nonhier_safe_artifact)
                if args.nonhier_safe_artifact is not None
                else None
            ),
            "nonhier_safe_artifact_sha256": (
                hashlib.sha256(args.nonhier_safe_artifact.read_bytes()).hexdigest()
                if args.nonhier_safe_artifact is not None
                else None
            ),
            "nonhier_direction": "forward_source_to_target",
            "nonhier_max_depth": 1,
            "nonhier_uses_neo4j_umls_edges": False,
            "nonhier_includes_same_as": False,
            "nonhier_includes_isa": False,
            "nonhier_includes_external_cuis": False,
            "nonhier_benchmark_tuned": False,
            "nonhier_v1_seed_match_policy": "permissive",
            "nonhier_strict_seed_match_policy": "exact_name_only",
            "nonhier_v1_support_only_ranking_active": True,
            "nonhier_strict_support_only_ranking_active": False,
            "nonhier_v3_ranking_policy": "direct_first_graph_second",
            "nonhier_frozen_direct_candidate_pool_mode": (
                "mentions_nonhier_artifact_safe_strict_direct_first_frozen"
                in modes
            ),
            "nonhier_controlled_candidate_rescue_mode": (
                "mentions_nonhier_artifact_safe_strict_direct_first_rescue" in modes
            ),
            "connection_config": (
                str(args.connection_config)
                if args.connection_config is not None
                else None
            ),
            "connection_config_sha256": (
                hashlib.sha256(args.connection_config.read_bytes()).hexdigest()
                if args.connection_config is not None
                else None
            ),
            "connection_modes": sorted(
                mode for mode in modes if mode in connection_modes
            ),
            "connection_candidate_ranking": (
                "concept_match_total_preserve_baseline_when_no_graph"
            ),
            "pool_preserving_local_candidate_k": args.candidate_k,
            "pool_preserving_graph_candidate_k": (args.graph_candidate_k if args.graph_candidate_k is not None else args.candidate_k),
            "pool_preserving_candidate_k_semantics": (
                "candidate_k is the local-channel budget; graph-channel budget is independent; union is not globally truncated before final top_k"
            ),
            "connection_max_depth": 1,
            "connection_second_hop": False,
            "connection_gold_used_for_retrieval": False,
            "descendants_per_seed": args.descendants_per_seed,
            "max_expanded_rows": args.max_expanded_rows,
            "advanced_ranking_mode": args.advanced_ranking_mode,
            "rrf_k": args.rrf_k,
            "advanced_multiple_facets_context_aware_merge": (
                args.advanced_multiple_facets_context_aware_merge
            ),
            "advanced_multiple_facets_context_candidate_injection": (
                args.advanced_multiple_facets_context_candidate_injection
            ),
            "advanced_same_section_anchor_fallback": (
                args.advanced_same_section_anchor_fallback
            ),
            "advanced_same_section_anchor_rescue": (
                args.advanced_same_section_anchor_rescue
            ),
            "exclude_summary_sections": not args.include_summary_sections,
            "document_filtering": (document_ids or None),
            "gold_annotations_used_for_retrieval": False,
            "mentions_plan_reused_across_modular_modes": True,
            "mentions_plans_file": (
                str(supplied_plans_path)
                if supplied_plans_path is not None
                else None
            ),
            "mentions_plans_source": (
                "loaded_from_file"
                if supplied_plans_path is not None
                else "generated_or_resumed"
            ),
        },
        "status": "running",
    }
    write_json(manifest_path, manifest)

    needs_llm_service = (
        "planned_role_aware" in modes
        or (
            any(mode in MODULAR_MODES for mode in modes)
            and supplied_plans_path is None
        )
    )
    if needs_llm_service:
        llm_service, router_config = build_llm_service(
            model_name=args.model,
            nodes_config_path=args.nodes_config,
        )
        mentions_router: KGMentionsRouter | None = KGMentionsRouter(
            llm_service=llm_service,
            router_config=router_config,
        )
        advanced_router: KGStructuredRouter | None = KGStructuredRouter(
            llm_service=llm_service,
            router_config=router_config,
        )
    else:
        mentions_router = None
        advanced_router = None

    started = time.perf_counter()
    failure_count = 0
    generated_plan_count = 0
    with Neo4jKGClient.from_env(env_path=env_path) as client:
        tools = KGSectionTools(client)
        advanced_retriever = None
        if "planned_role_aware" in modes:
            if advanced_router is None:
                raise RuntimeError("planned_role_aware requires LLM routing")
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
                multiple_facets_context_candidate_injection=(
                    args.advanced_multiple_facets_context_candidate_injection
                ),
                same_section_anchor_fallback=(
                    args.advanced_same_section_anchor_fallback
                ),
                same_section_anchor_rescue=(
                    args.advanced_same_section_anchor_rescue
                ),
                document_ids=(document_ids or None),
            )

        embedding_seeder = None
        if any(mode in modes for mode in embedding_seed_modes):
            embedding_seeder = EmbeddingConceptSeeder(
                tools,
                embedding_model=args.concept_embedding_model,
                concepts_per_term=args.concepts_per_term,
                cache_path=args.concept_embedding_cache,
            )
            embedding_seeder.prepare(document_ids=(document_ids or None))
            manifest["configuration"].update(
                {
                    "concept_catalogue_size": (
                        embedding_seeder.catalogue_size
                    ),
                    "concept_catalogue_build_load_seconds": (
                        embedding_seeder.catalogue_build_load_seconds
                    ),
                    "concept_embedding_model_load_seconds": (
                        embedding_seeder.model_load_seconds
                    ),
                    "concept_embedding_cache_loaded": (
                        embedding_seeder.loaded_from_cache
                    ),
                    "concept_embedding_cache_file": (
                        embedding_seeder.resolved_cache_file
                    ),
                }
            )
            write_json(manifest_path, manifest)

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
                if mentions_router is None:
                    raise RuntimeError(
                        "No reusable MENTIONS plan available and the "
                        "MENTIONS router was not initialized"
                    )
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
                    generated_plan_count += 1
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
                            document_ids=(document_ids or None),
                            exclude_summary_sections=(
                                not args.include_summary_sections
                            ),
                            hierarchy_max_depth=args.hierarchy_max_depth,
                            descendants_per_seed=args.descendants_per_seed,
                            max_expanded_rows=args.max_expanded_rows,
                            concepts_per_term=args.concepts_per_term,
                            concept_embedding_model=(
                                args.concept_embedding_model
                            ),
                            concept_embedding_cache=(
                                str(args.concept_embedding_cache)
                                if args.concept_embedding_cache is not None
                                else None
                            ),
                            concept_seeder=(
                                embedding_seeder
                                if mode in embedding_seed_modes
                                else None
                            ),
                            isa_connections_path=(
                                str(args.isa_connections)
                                if args.isa_connections is not None
                                else None
                            ),
                            isa_max_depth=args.isa_max_depth,
                            nonhier_artifact_path=(
                                str(args.nonhier_raw_artifact)
                                if mode in {
                                    "mentions_nonhier_artifact_raw",
                                    "mentions_nonhier_artifact_raw_strict",
                                    "mentions_nonhier_artifact_raw_strict_direct_first",
                                }
                                and args.nonhier_raw_artifact is not None
                                else (
                                    str(args.nonhier_safe_artifact)
                                    if mode in {
                                        "mentions_nonhier_artifact_safe",
                                        "mentions_nonhier_artifact_safe_strict",
                                        "mentions_nonhier_artifact_safe_strict_direct_first",
                                        "mentions_nonhier_artifact_safe_strict_direct_first_frozen",
                                        "mentions_nonhier_artifact_safe_strict_direct_first_rescue",
                                    }
                                    and args.nonhier_safe_artifact is not None
                                    else None
                                )
                            ),
                            connection_config_path=(
                                str(args.connection_config)
                                if mode in connection_modes
                                and args.connection_config is not None
                                else None
                            ),
                            graph_candidate_k=args.graph_candidate_k,
                            rrf_k=args.rrf_k,
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
                        if advanced_retriever is None:
                            raise RuntimeError(
                                "planned_role_aware retriever is not "
                                "initialized"
                            )
                        run = advanced_retriever.retrieve(question)
                        raw_results = [
                            result
                            for execution in run.tool_executions
                            for result in execution.results
                        ]
                        expanded_results = None
                        final_results = list(run.results)

                    validate_result_document_scope(
                        raw_results,
                        document_ids,
                        stage=f"{mode}:raw_candidates",
                    )
                    if expanded_results is not None:
                        validate_result_document_scope(
                            expanded_results,
                            document_ids,
                            stage=f"{mode}:expanded_candidates",
                        )
                    validate_result_document_scope(
                        final_results,
                        document_ids,
                        stage=f"{mode}:final_results",
                    )

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
    question_groups = build_question_group_membership(dataset)
    group_aggregates = aggregate_metric_rows_by_group(
        metric_rows,
        question_groups,
    )
    concept_seed_diagnostics = build_concept_seed_diagnostics(current_rows)

    write_metric_csv(output_dir / "per_query_metrics.csv", metric_rows)
    write_aggregate_csv(output_dir / "aggregate_metrics.csv", aggregates)
    group_aggregate_path = (
        output_dir / "aggregate_metrics_by_group.csv"
    )
    write_group_aggregate_csv(
        group_aggregate_path,
        group_aggregates,
    )
    concept_seed_diagnostics_path = output_dir / "concept_seed_diagnostics.json"
    write_json(concept_seed_diagnostics_path, concept_seed_diagnostics)
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
        "mentions_plans_generated_count": generated_plan_count,
        "concept_seed_diagnostics_file": str(concept_seed_diagnostics_path),
        "elapsed_seconds": time.perf_counter() - started,
        "configuration": manifest["configuration"],
        "aggregates": aggregates,
        "group_aggregates_file": str(group_aggregate_path),
        "group_aggregates": group_aggregates,
    }
    write_json(output_dir / "summary.json", summary)

    manifest["status"] = "completed_with_errors" if failure_count else "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_file"] = str(output_dir / "summary.json")
    manifest["concept_seed_diagnostics_file"] = str(
        concept_seed_diagnostics_path
    )
    manifest["group_aggregates_file"] = str(
        group_aggregate_path
    )
    manifest["mentions_plans_generated_count"] = generated_plan_count
    write_json(manifest_path, manifest)

    print()
    print("Batch evaluation complete")
    print("Output:", output_dir)
    print("Completed query/mode runs:", len(current_rows))
    print("Failures:", failure_count)


if __name__ == "__main__":
    main()
