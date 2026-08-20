from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.kg.direct_local_artifact import FrozenDirectLocalArtifact


def make_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    (root / "adjacency" / "by_source").mkdir(parents=True)
    manifest = {
        "schema_version": "direct_local_artifact_v1",
        "inverse_relations_grouped": True,
        "retrieval_adjacency_is_bidirectional_per_semantic_pair": True,
        "source_summary": {
            "SNOMEDCT_US": {"pair_count": 1},
            "NCI": {"pair_count": 1},
            "OMIM": {"pair_count": 0},
        },
        "safety": {
            "umls_api_calls": False,
            "neo4j_writes": False,
            "benchmark_data_used": False,
            "retrieval_metrics_used": False,
            "benchmark_tuned": False,
            "second_hop_traversal": False,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    pair = {
        "direct_id": "DLV1-A__B",
        "local_cui_a": "A",
        "local_cui_b": "B",
        "local_a": {"names": ["A name"], "canonical_types": ["disease"]},
        "local_b": {"names": ["B name"], "canonical_types": ["clinical_finding"]},
    }
    (root / "pair_evidence.jsonl").write_text(json.dumps(pair) + "\n")

    edge_ab = {
        "direct_id": "DLV1-A__B",
        "tier": "MEDIUM",
        "sources": ["SNOMEDCT_US"],
        "relation_families": ["manifestation"],
        "projection_ambiguity": False,
        "neighbor_cui": "B",
    }
    edge_ba = dict(edge_ab, neighbor_cui="A")
    adj = {"A": [edge_ab], "B": [edge_ba]}
    for profile in ("strong", "balanced", "broad"):
        value = {} if profile == "strong" else adj
        (root / "adjacency" / f"{profile}.json").write_text(json.dumps(value))
        (root / "adjacency" / "by_source" / f"SNOMEDCT_US__{profile}.json").write_text(
            json.dumps(value)
        )
        (root / "adjacency" / "by_source" / f"NCI__{profile}.json").write_text(
            json.dumps({})
        )
    return root


def test_loader_validates_and_reads_bidirectional_pair(tmp_path: Path):
    artifact = FrozenDirectLocalArtifact(make_artifact(tmp_path))
    adj = artifact.build_adjacency(profile="balanced", sources=["SNOMEDCT_US"])
    assert adj["A"][0].neighbor_cui == "B"
    assert adj["B"][0].neighbor_cui == "A"
    assert adj["A"][0].neighbor_preferred_name == "B name"


def test_source_union_deduplicates_same_direct_id(tmp_path: Path):
    root = make_artifact(tmp_path)
    # Copy the same semantic pair into NCI with NCI provenance.
    src = json.loads(
        (root / "adjacency/by_source/SNOMEDCT_US__balanced.json").read_text()
    )
    for entries in src.values():
        for entry in entries:
            entry["sources"] = ["NCI"]
    (root / "adjacency/by_source/NCI__balanced.json").write_text(json.dumps(src))
    artifact = FrozenDirectLocalArtifact(root)
    adj = artifact.build_adjacency(
        profile="balanced", sources=["SNOMEDCT_US", "NCI"]
    )
    assert len(adj["A"]) == 1
    assert adj["A"][0].sources == ("NCI", "SNOMEDCT_US")


def test_strong_profile_can_be_empty(tmp_path: Path):
    artifact = FrozenDirectLocalArtifact(make_artifact(tmp_path))
    assert artifact.build_adjacency(
        profile="strong", sources=["SNOMEDCT_US"]
    ) == {}


def test_unknown_source_fails(tmp_path: Path):
    artifact = FrozenDirectLocalArtifact(make_artifact(tmp_path))
    with pytest.raises(ValueError):
        artifact.build_adjacency(profile="balanced", sources=["OMIM"])


def test_safety_flag_is_enforced(tmp_path: Path):
    root = make_artifact(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["safety"]["benchmark_tuned"] = True
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        FrozenDirectLocalArtifact(root)
