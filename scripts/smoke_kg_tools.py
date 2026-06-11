from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_rag.kg.client import Neo4jKGClient
from agentic_rag.kg.tools import KGSectionTools


DATASET_PATH = Path("tests/data/subset_test_en_cm_syn.json")
OUTPUT_ROOT = Path("artifacts/kg_retrieval")
TOP_K = 10

CASES = {
    "CM_01": {
        "tool": "concepts",
        "terms": ["hypertrophic cardiomyopathy"],
        "document_ids": ["Cardiomyopathies_2023"],
        "require_all": False,
    },
    "CM_06": {
        "tool": "concepts",
        "terms": ["atrial fibrillation", "anticoagulation"],
        "document_ids": ["Cardiomyopathies_2023"],
        "require_all": False,
    },
    "CM_07": {
        "tool": "title",
        "terms": ["cardiac magnetic resonance"],
        "document_ids": ["Cardiomyopathies_2023"],
        "require_all": False,
    },
    "SYN_04": {
        "tool": "title",
        "terms": ["implantable loop recorders", "diagnostic criteria"],
        "document_ids": ["Syncope_2018"],
        "require_all": False,
    },
    "SYN_07": {
        "tool": "concepts",
        "terms": ["orthostatic hypotension"],
        "document_ids": ["Syncope_2018"],
        "require_all": False,
    },
    "SYN_09": {
        "tool": "title",
        "terms": ["falls", "cognitive assessment"],
        "document_ids": ["Syncope_2018"],
        "require_all": False,
    },
}

RANKING_MODES = (
    "concept_match",
    "weighted_match",
)


def load_questions() -> dict[str, dict[str, Any]]:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return {
        question["id"]: question
        for question in payload["questions"]
    }


def serialize_result(result) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "unit_type": "section",
        "section_uid": result.section_uid,
        "document_id": result.document_id,
        "section_id": result.section_id,
        "title": result.title,
        "text": result.text,
        "page_content": result.page_content,
        "matched_concepts": result.matched_concepts,
        "matched_terms": result.matched_terms,
        "score": result.score,
        "score_type": result.score_type,
        "scores": (
            result.scores.model_dump(mode="json")
            if result.scores is not None
            else None
        ),
        "match_diagnostics": [
            diagnostic.model_dump(mode="json")
            for diagnostic in result.match_diagnostics
        ],
    }


def run_case(
    tools: KGSectionTools,
    question: dict[str, Any],
    case: dict[str, Any],
    ranking_mode: str,
    run_id: str,
) -> dict[str, Any]:
    question_id = question["id"]
    query_run_id = f"{run_id}::{question_id}::{ranking_mode}"

    started = time.perf_counter()

    try:
        if case["tool"] == "concepts":
            results = tools.search_sections_by_concepts(
                case["terms"],
                document_ids=case["document_ids"],
                top_k=TOP_K,
                require_all=case["require_all"],
                ranking_mode=ranking_mode,
            )
            selected_tool = "search_sections_by_concepts"

        elif case["tool"] == "title":
            results = tools.search_sections_by_title(
                case["terms"],
                document_ids=case["document_ids"],
                top_k=TOP_K,
                require_all=case["require_all"],
                ranking_mode=ranking_mode,
            )
            selected_tool = "search_sections_by_title"

        else:
            raise ValueError(f"Unknown tool: {case['tool']}")

        status = "success" if results else "no_results"
        error = None

    except Exception as exc:
        results = []
        selected_tool = case["tool"]
        status = "execution_error"
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "question_id": question_id,
        "query_run_id": query_run_id,
        "question": question["question"],
        "status": status,
        "selected_tool": selected_tool,
        "tool_parameters": {
            "terms": case["terms"],
            "document_ids": case["document_ids"],
            "require_all": case["require_all"],
            "top_k": TOP_K,
            "ranking_mode": ranking_mode,
        },
        "requested_k": TOP_K,
        "returned_count": len(results),
        "latency_ms": latency_ms,
        "retrieved_units": [
            serialize_result(result)
            for result in results
        ],
        "error": error,
    }


def main() -> None:
    questions = load_questions()

    missing = sorted(set(CASES) - set(questions))
    if missing:
        raise ValueError(
            f"Questions missing from dataset: {missing}"
        )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_id = f"kg_tools_smoke_{timestamp}"

    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "retrieval_run.json"

    queries: list[dict[str, Any]] = []

    with Neo4jKGClient.from_env() as client:
        tools = KGSectionTools(client)

        for question_id, case in CASES.items():
            question = questions[question_id]

            for ranking_mode in RANKING_MODES:
                record = run_case(
                    tools=tools,
                    question=question,
                    case=case,
                    ranking_mode=ranking_mode,
                    run_id=run_id,
                )
                queries.append(record)

                print()
                print(
                    question_id,
                    ranking_mode,
                    record["status"],
                    f"returned={record['returned_count']}",
                )

                for unit in record["retrieved_units"][:5]:
                    print(
                        f"  {unit['rank']}. "
                        f"{unit['document_id']} "
                        f"{unit['section_id']} "
                        f"{unit['title']} "
                        f"score={unit['score']}"
                    )

    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": str(DATASET_PATH),
        "configuration": {
            "retriever": "kg_parameterized_tools",
            "top_k": TOP_K,
            "ranking_modes": list(RANKING_MODES),
            "manual_parameter_mapping": True,
            "exact_section_text_saved": True,
        },
        "queries": queries,
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()