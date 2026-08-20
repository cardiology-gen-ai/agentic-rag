from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_seed_channel_replay.py"
)
spec = importlib.util.spec_from_file_location("seed_replay", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_evaluate_counts_covered_sections():
    gold = {("d", "1"), ("d", "2")}
    ranked = [
        {"covered": {("d", "1")}},
        {"covered": {("d", "x")}},
        {"covered": {("d", "2")}},
    ]
    out = mod.evaluate(gold, ranked, 3)
    assert out["hit@3"] == 1.0
    assert out["recall@3"] == 1.0
    assert out["complete_recall@3"] == 1.0
    assert out["mrr@3"] == 1.0


def test_semantic_similarity_sums_best_per_term():
    unit = {
        "match_diagnostics": [
            {"query_term": "a", "similarity": 0.5},
            {"query_term": "a", "similarity": 0.8},
            {"query_term": "b", "similarity": 0.6},
        ]
    }
    score, terms = mod.semantic_similarity_score(unit)
    assert abs(score - 1.4) < 1e-9
    assert terms == 2


def test_policies_are_frozen():
    assert mod.POLICIES == (
        "lexical_existing",
        "semantic_unweighted_existing",
        "semantic_similarity_weighted",
        "hybrid_best_channel",
    )
