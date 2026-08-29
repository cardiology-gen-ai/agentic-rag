from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_matrix_runner():
    path = Path("scripts/evaluate_kg_matrix.py")
    spec = importlib.util.spec_from_file_location("kg_matrix_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(uid: str, original_rank: int):
    return {
        "section": {"section_uid": uid},
        "metadata": {"original_seeded_rank": original_rank},
    }


def test_semantic_order_need_not_be_prefix_when_membership_is_nested():
    runner = _load_matrix_runner()

    # K30 semantic reranking can order retained candidates A,C,B.
    small = [
        _candidate("A", 1),
        _candidate("C", 3),
        _candidate("B", 2),
    ]
    # K50 admits D at coverage rank 4; semantic scoring can move D to the top.
    large = [
        _candidate("D", 4),
        _candidate("A", 1),
        _candidate("C", 3),
        _candidate("B", 2),
    ]

    runner.validate_nested_semantic_pools(
        small,
        large,
        smaller_k=3,
        larger_k=4,
        question_id="Q1",
    )


def test_nested_semantic_pool_rejects_membership_drift():
    runner = _load_matrix_runner()
    small = [_candidate("A", 1), _candidate("X", 2)]
    large = [_candidate("A", 1), _candidate("B", 2), _candidate("C", 3)]

    with pytest.raises(RuntimeError, match="not contained"):
        runner.validate_nested_semantic_pools(
            small,
            large,
            smaller_k=2,
            larger_k=3,
            question_id="Q1",
        )


def test_nested_semantic_pool_rejects_changed_original_rank():
    runner = _load_matrix_runner()
    small = [_candidate("A", 1), _candidate("B", 2)]
    large = [_candidate("A", 2), _candidate("B", 1), _candidate("C", 3)]

    with pytest.raises(RuntimeError, match="original seeded rank changed"):
        runner.validate_nested_semantic_pools(
            small,
            large,
            smaller_k=2,
            larger_k=3,
            question_id="Q1",
        )
