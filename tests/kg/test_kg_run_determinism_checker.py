from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_checker():
    path = Path("scripts/check_kg_run_determinism_v1.py")
    spec = importlib.util.spec_from_file_location("kg_determinism_checker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_row():
    return {
        "question_id": "Q1",
        "mode": "mentions_embedding_seeded_similarity_weighted",
        "status": "success",
        "plan": {"terms": ["DMD"]},
        "concept_seeds": [{"query_term": "DMD", "concept_name": "DMD", "similarity": 0.9}],
        "raw_candidates": [
            {
                "document_id": "Cardiomyopathies_2023",
                "printed_section_id": "7.1",
                "title": "Example",
            }
        ],
        "expanded_candidates": [],
        "final_ranking": [
            {
                "document_id": "Cardiomyopathies_2023",
                "retrieval_unit_id": "Cardiomyopathies_2023::7.1",
                "covered_section_ids": ["7.1"],
                "raw_rank": 1,
                "source_record_ids": ["Cardiomyopathies_2023::7.1"],
                "source_type": "kg_section",
                "raw_score": 1.0,
            }
        ],
        "candidate_diagnostics": {"final_ranking": {"best_gold_rank": 1}},
        "metrics": {"section": {"recall@20": 1.0}},
        "retrieval_trace": {
            "latency_ms": 1.0,
            "raw_candidates": [
                {
                    "section": {
                        "section_uid": "Cardiomyopathies_2023::7.1",
                        "document_id": "Cardiomyopathies_2023",
                        "printed_section_id": "7.1",
                        "text": "x",
                        "score": 1.0,
                    },
                    "source": "mentions",
                    "source_rank": 1,
                    "direct": True,
                    "metadata": {},
                }
            ],
            "expanded_candidates": [],
            "results": [],
            "router_term_normalization": {"mode": "none"},
        },
    }


def test_checker_preserves_native_candidate_identity():
    checker = _load_checker()
    left = _base_row()
    right = _base_row()
    right["raw_candidates"][0]["printed_section_id"] = "7.2"

    assert (
        checker.stable_row(left)["raw_candidate_keys"]
        != checker.stable_row(right)["raw_candidate_keys"]
    )


def test_checker_preserves_final_evidence_coverage():
    checker = _load_checker()
    left = _base_row()
    right = _base_row()
    right["final_ranking"][0]["covered_section_ids"] = ["7.1", "7.1.1"]

    assert (
        checker.stable_row(left)["final_evidence"]
        != checker.stable_row(right)["final_evidence"]
    )


def test_checker_compares_full_trace_candidate_scores():
    checker = _load_checker()
    left = _base_row()
    right = _base_row()
    right["retrieval_trace"]["raw_candidates"][0]["section"]["score"] = 0.99

    assert (
        checker.stable_row(left)["trace_raw_candidates"]
        != checker.stable_row(right)["trace_raw_candidates"]
    )
