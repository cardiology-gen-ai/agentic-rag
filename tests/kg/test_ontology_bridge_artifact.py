from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.kg.ontology_bridge_artifact import (
    FrozenOntologyBridgeArtifact,
)


def _write_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "bridge"
    artifact.mkdir()
    manifest = {
        "schema_version": "ontology_bridge_artifact_v1_1",
        "sources_requested": ["SNOMEDCT_US", "NCI"],
        "files": {"pair_evidence": "pair_evidence.jsonl"},
        "safety": {
            "umls_api_calls": False,
            "neo4j_writes": False,
            "second_hop_requests": False,
            "retrieval_metrics_used": False,
            "benchmark_tuned": False,
        },
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))

    rows = [
        {
            "bridge_id": "A__B",
            "local_cui_a": "A",
            "local_cui_b": "B",
            "local_a": {
                "preferred_name": "A",
                "canonical_types": ["disease"],
            },
            "local_b": {
                "preferred_name": "B",
                "canonical_types": ["disease"],
            },
            "score_top_k": 5,
            "external_path_evidence": [
                {
                    "source_vocabulary": "SNOMEDCT_US",
                    "best_tier_for_external_path": "STRONG",
                    "bridge_score_contribution_v1_1": 0.5,
                    "external_cui": "X1",
                    "external_preferred_name": "Strong SNOMED X",
                },
                {
                    "source_vocabulary": "NCI",
                    "best_tier_for_external_path": "MEDIUM",
                    "bridge_score_contribution_v1_1": 0.9,
                    "external_cui": "X2",
                    "external_preferred_name": "Medium NCI X",
                },
            ],
        },
        {
            "bridge_id": "A__C",
            "local_cui_a": "A",
            "local_cui_b": "C",
            "local_a": {
                "preferred_name": "A",
                "canonical_types": ["disease"],
            },
            "local_b": {
                "preferred_name": "C",
                "canonical_types": ["clinical_finding"],
            },
            "score_top_k": 5,
            "external_path_evidence": [
                {
                    "source_vocabulary": "SNOMEDCT_US",
                    "best_tier_for_external_path": "MEDIUM",
                    "bridge_score_contribution_v1_1": 1.2,
                    "external_cui": "X3",
                    "external_preferred_name": "High-score medium X",
                }
            ],
        },
    ]
    (artifact / "pair_evidence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return artifact


def test_source_specific_scores_do_not_leak_between_sources(tmp_path: Path):
    artifact = FrozenOntologyBridgeArtifact(_write_artifact(tmp_path))

    snomed = artifact.build_adjacency(
        profile="balanced",
        sources=["SNOMEDCT_US"],
        ranking_policy="tier_first",
    )
    nci = artifact.build_adjacency(
        profile="balanced",
        sources=["NCI"],
        ranking_policy="tier_first",
    )

    b_snomed = next(row for row in snomed["A"] if row.neighbor_cui == "B")
    b_nci = next(row for row in nci["A"] if row.neighbor_cui == "B")

    assert b_snomed.sources == ("SNOMEDCT_US",)
    assert b_nci.sources == ("NCI",)
    assert b_snomed.score != b_nci.score
    assert b_snomed.tier == "STRONG"
    assert b_nci.tier == "MEDIUM"


def test_tier_first_keeps_strong_before_higher_score_medium(tmp_path: Path):
    artifact = FrozenOntologyBridgeArtifact(_write_artifact(tmp_path))
    adjacency = artifact.build_adjacency(
        profile="balanced",
        sources=["SNOMEDCT_US"],
        ranking_policy="tier_first",
    )
    assert [row.neighbor_cui for row in adjacency["A"]] == ["B", "C"]
    assert adjacency["A"][0].tier == "STRONG"
    assert adjacency["A"][1].tier == "MEDIUM"
    assert adjacency["A"][1].score > adjacency["A"][0].score


def test_strong_profile_excludes_medium_only_pair(tmp_path: Path):
    artifact = FrozenOntologyBridgeArtifact(_write_artifact(tmp_path))
    adjacency = artifact.build_adjacency(
        profile="strong",
        sources=["SNOMEDCT_US"],
    )
    assert [row.neighbor_cui for row in adjacency["A"]] == ["B"]
