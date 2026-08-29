from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_matrix_runner():
    path = Path("scripts/evaluate_kg_matrix.py")
    spec = importlib.util.spec_from_file_location("kg_matrix_runner_norm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(mode: str):
    return {
        "retrieval_trace": {
            "router_term_normalization": mode,
        }
    }


def test_matrix_accepts_matching_safe_v1_semantic_runs():
    runner = _load_matrix_runner()
    runner.validate_router_normalization_mode(
        {30: {"Q1": _row("safe_v1")}, 50: {"Q1": _row("safe_v1")}},
        expected_mode="safe_v1",
    )


def test_matrix_rejects_cross_treatment_semantic_pool_reuse():
    runner = _load_matrix_runner()
    with pytest.raises(RuntimeError, match="router normalization mismatch"):
        runner.validate_router_normalization_mode(
            {30: {"Q1": _row("none")}, 50: {"Q1": _row("safe_v1")}},
            expected_mode="safe_v1",
        )


def test_matrix_rejects_unknown_normalization_mode():
    runner = _load_matrix_runner()
    with pytest.raises(ValueError, match="none, safe_v1"):
        runner.validate_router_normalization_mode(
            {30: {"Q1": _row("none")}},
            expected_mode="semantic_rewrite_v99",
        )
