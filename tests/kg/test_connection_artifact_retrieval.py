from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.kg.connection_artifact_retrieval import (
    FrozenConnectionAblationConfig,
    _build_bridge_expansions,
    _build_direct_expansions,
)
from agentic_rag.kg.direct_local_artifact import DirectNeighbor
from agentic_rag.kg.ontology_bridge_artifact import BridgeNeighbor


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "kg_connection_ablation_v1",
                "experiment": {},
                "artifacts": {
                    "direct_artifact_dir": "direct",
                    "bridge_artifact_dir": "bridge",
                },
                "connection_modes": {
                    "mentions_direct_balanced": {
                        "direct": {
                            "profile": "balanced",
                            "sources": ["SNOMEDCT_US", "NCI", "OMIM"],
                        },
                        "bridge": None,
                    },
                    "mentions_direct_bridge_sa_top5": {
                        "direct": {
                            "profile": "balanced",
                            "sources": ["SNOMEDCT_US", "NCI", "OMIM"],
                        },
                        "bridge": {
                            "source_profiles": {
                                "SNOMEDCT_US": "strong",
                                "NCI": "balanced",
                                "OMIM": "balanced",
                            },
                            "top_n": 5,
                            "ranking_policy": "tier_first",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def seed() -> list[dict[str, object]]:
    return [
        {
            "query_term": "fabry",
            "seed_cui": "A",
            "seed_concept_name": "Fabry disease",
            "match_type": "exact_name",
            "matched_value": "fabry",
            "lexical_weight": 3.0,
        }
    ]


def test_config_keeps_final_source_aware_top5(tmp_path: Path):
    config = FrozenConnectionAblationConfig(write_config(tmp_path))
    mode = config.mode("mentions_direct_bridge_sa_top5")
    assert mode["direct"]["profile"] == "balanced"
    assert mode["bridge"]["top_n"] == 5
    assert mode["bridge"]["source_profiles"] == {
        "SNOMEDCT_US": "strong",
        "NCI": "balanced",
        "OMIM": "balanced",
    }


def test_direct_expansion_keeps_distinct_semantic_pairs():
    adjacency = {
        "A": (
            DirectNeighbor(
                seed_cui="A",
                neighbor_cui="B",
                neighbor_preferred_name="B",
                neighbor_types=("disease",),
                direct_id="D1",
                tier="MEDIUM",
                sources=("OMIM",),
                relation_families=("manifestation",),
                projection_ambiguity=False,
            ),
            DirectNeighbor(
                seed_cui="A",
                neighbor_cui="C",
                neighbor_preferred_name="C",
                neighbor_types=("disease",),
                direct_id="D2",
                tier="MEDIUM",
                sources=("OMIM",),
                relation_families=("manifestation",),
                projection_ambiguity=False,
            ),
        )
    }
    expansions = _build_direct_expansions(seed(), adjacency)
    assert {(row["target_cui"], row["artifact_edge_id"]) for row in expansions} == {
        ("B", "D1"),
        ("C", "D2"),
    }


def test_bridge_top_n_is_applied_per_source():
    def neighbor(seed_cui: str, target: str, bridge_id: str, source: str) -> BridgeNeighbor:
        return BridgeNeighbor(
            seed_cui=seed_cui,
            neighbor_cui=target,
            neighbor_preferred_name=target,
            neighbor_types=("disease",),
            bridge_id=bridge_id,
            tier="MEDIUM",
            score=0.5,
            sources=(source,),
            external_hub_count=1,
            top_external_cuis=("X",),
            top_external_names=("External",),
        )

    source_adjacencies = {
        "SNOMEDCT_US": {
            "A": (
                neighbor("A", "S1", "BS1", "SNOMEDCT_US"),
                neighbor("A", "S2", "BS2", "SNOMEDCT_US"),
            )
        },
        "OMIM": {
            "A": (
                neighbor("A", "O1", "BO1", "OMIM"),
                neighbor("A", "O2", "BO2", "OMIM"),
            )
        },
    }
    expansions = _build_bridge_expansions(
        seed(),
        source_adjacencies,
        top_n=1,
    )
    assert {(row["target_cui"], row["traversal_policy"]) for row in expansions} == {
        ("S1", "source_aware_SNOMEDCT_US_top1_depth1"),
        ("O1", "source_aware_OMIM_top1_depth1"),
    }


def test_bridge_contextual_target_ranking_can_replace_frozen_top1():
    def neighbor(target: str, bridge_id: str) -> BridgeNeighbor:
        return BridgeNeighbor(
            seed_cui="A",
            neighbor_cui=target,
            neighbor_preferred_name=target,
            neighbor_types=("disease",),
            bridge_id=bridge_id,
            tier="MEDIUM",
            score=0.5,
            sources=("SNOMEDCT_US",),
            external_hub_count=1,
            top_external_cuis=("X",),
            top_external_names=("External",),
        )

    trace: list[dict[str, object]] = []
    expansions = _build_bridge_expansions(
        seed(),
        {
            "SNOMEDCT_US": {
                "A": (
                    neighbor("CURRENT_FIRST", "B1"),
                    neighbor("CONTEXT_FIRST", "B2"),
                )
            }
        },
        top_n=1,
        ranking_policy="contextual_target",
        question_context="full clinical question",
        contextual_scorer=lambda _query, labels: [
            0.1 if label == "CURRENT_FIRST" else 0.9
            for label in labels
        ],
        ranking_trace=trace,
    )
    assert [row["target_cui"] for row in expansions] == ["CONTEXT_FIRST"]
    by_target = {row["target_cui"]: row for row in trace}
    assert by_target["CURRENT_FIRST"]["current_rank"] == 1
    assert by_target["CURRENT_FIRST"]["contextual_rank"] == 2
    assert by_target["CURRENT_FIRST"]["selected"] is False
    assert by_target["CONTEXT_FIRST"]["current_rank"] == 2
    assert by_target["CONTEXT_FIRST"]["contextual_rank"] == 1
    assert by_target["CONTEXT_FIRST"]["selected"] is True


def test_bridge_contextual_rrf_rewards_complementary_middle_candidate():
    def neighbor(target: str, bridge_id: str) -> BridgeNeighbor:
        return BridgeNeighbor(
            seed_cui="A",
            neighbor_cui=target,
            neighbor_preferred_name=target,
            neighbor_types=("disease",),
            bridge_id=bridge_id,
            tier="MEDIUM",
            score=0.5,
            sources=("SNOMEDCT_US",),
            external_hub_count=1,
            top_external_cuis=("X",),
            top_external_names=("External",),
        )

    expansions = _build_bridge_expansions(
        seed(),
        {
            "SNOMEDCT_US": {
                "A": (
                    neighbor("A1", "B1"),
                    neighbor("A2", "B2"),
                    neighbor("A3", "B3"),
                    neighbor("A4", "B4"),
                )
            }
        },
        top_n=1,
        ranking_policy="tier_first_contextual_rrf",
        question_context="full clinical question",
        # Current order:     A1, A2, A3, A4
        # Contextual order:  A4, A2, A3, A1
        #
        # A2 therefore has rank 2 in both channels and beats the two
        # complementary extremes (1,4) and (4,1) under standard RRF.
        contextual_scorer=lambda _query, labels: [
            {
                "A1": 0.1,
                "A2": 0.7,
                "A3": 0.5,
                "A4": 0.9,
            }[label]
            for label in labels
        ],
        contextual_rrf_k=60,
    )
    assert [row["target_cui"] for row in expansions] == ["A2"]


def test_frozen_tier_first_expansion_payload_is_unchanged():
    neighbor = BridgeNeighbor(
        seed_cui="A",
        neighbor_cui="B",
        neighbor_preferred_name="B",
        neighbor_types=("disease",),
        bridge_id="B1",
        tier="MEDIUM",
        score=0.5,
        sources=("SNOMEDCT_US",),
        external_hub_count=1,
        top_external_cuis=("X",),
        top_external_names=("External",),
    )
    expansion = _build_bridge_expansions(
        seed(),
        {"SNOMEDCT_US": {"A": (neighbor,)}},
        top_n=1,
        ranking_policy="tier_first",
    )[0]
    assert "relation_contextual_score" not in expansion
    assert "relation_contextual_rank" not in expansion
    assert expansion["target_cui"] == "B"


def test_config_accepts_contextual_bridge_policies(tmp_path: Path):
    path = write_config(tmp_path)
    raw = json.loads(path.read_text())
    base = raw["connection_modes"]["mentions_direct_bridge_sa_top5"]
    raw["connection_modes"]["contextual"] = {
        **base,
        "bridge": {
            **base["bridge"],
            "ranking_policy": "contextual_target",
        },
    }
    raw["connection_modes"]["contextual_rrf"] = {
        **base,
        "bridge": {
            **base["bridge"],
            "ranking_policy": "tier_first_contextual_rrf",
            "contextual_rrf_k": 60,
            "contextual_embedding_cache_path": "context-cache",
            "contextual_embedding_cache_read_only": True,
        },
    }
    path.write_text(json.dumps(raw))
    config = FrozenConnectionAblationConfig(path)
    assert (
        config.mode("contextual")["bridge"]["ranking_policy"]
        == "contextual_target"
    )
    assert (
        config.mode("contextual_rrf")["bridge"]["ranking_policy"]
        == "tier_first_contextual_rrf"
    )


def test_mode_without_connections_is_rejected(tmp_path: Path):
    path = write_config(tmp_path)
    raw = json.loads(path.read_text())
    raw["connection_modes"]["bad"] = {"direct": None, "bridge": None}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        FrozenConnectionAblationConfig(path)
