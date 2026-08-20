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


def test_mode_without_connections_is_rejected(tmp_path: Path):
    path = write_config(tmp_path)
    raw = json.loads(path.read_text())
    raw["connection_modes"]["bad"] = {"direct": None, "bridge": None}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        FrozenConnectionAblationConfig(path)
