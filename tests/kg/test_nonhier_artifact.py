from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.kg.nonhier_artifact import (
    FrozenNonHierArtifact,
    _build_expansions,
    _filter_support_only_rows,
    _nonhier_seed_query,
    _seed_row_allowed,
)


def write_artifact(
    path: Path,
    *,
    artifact_name: str = "nonhier_semantic_safe_v1",
    support_only: bool = True,
) -> None:
    edges = [
        {
            "edge_id": "NH52-001",
            "source_cui": "C1",
            "target_cui": "C2",
            "relation_name": "has_associated_morphology",
            "semantic_status": "valid",
            "direction": "forward_source_to_target",
            "max_depth": 1,
            "expansion_mode": "expand",
        },
        {
            "edge_id": "NH52-002",
            "source_cui": "C1",
            "target_cui": "C3",
            "relation_name": "has_focus",
            "semantic_status": "valid_but_broad",
            "direction": "forward_source_to_target",
            "max_depth": 1,
            "expansion_mode": "support_only" if support_only else "expand",
        },
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": "umls_nonhier_retrieval_artifact_v1",
                "semantic_freeze_version": "umls_nonhier_semantic_freeze_v1",
                "traversal_version": "umls_nonhier_traversal_policy_v1",
                "edge_count": len(edges),
                "direction": "forward_source_to_target",
                "max_depth": 1,
                "same_as_included": False,
                "isa_included": False,
                "external_cuis_included": False,
                "benchmark_tuned": False,
                "artifact_name": artifact_name,
                "policy": "test policy",
                "support_only_edge_ids": (
                    ["NH52-002"] if support_only else []
                ),
                "edges": edges,
            }
        ),
        encoding="utf-8",
    )


def test_safe_artifact_loads_frozen_forward_edges(tmp_path: Path):
    path = tmp_path / "safe.json"
    write_artifact(path)
    artifact = FrozenNonHierArtifact(
        path,
        expected_artifact_name="nonhier_semantic_safe_v1",
    )

    assert artifact.edge_count == 2
    assert artifact.expand_edge_count == 1
    assert artifact.support_only_edge_count == 1
    assert artifact.support_only_edge_ids == ("NH52-002",)
    assert artifact.max_depth == 1
    assert artifact.direction == "forward_source_to_target"
    assert artifact.traversal_policy == "nonhier_artifact_safe_forward"
    assert [edge.edge_id for edge in artifact.forward_edges("C1")] == [
        "NH52-001",
        "NH52-002",
    ]
    assert artifact.forward_edges("C2") == ()


def test_raw_artifact_has_no_support_only_edges(tmp_path: Path):
    path = tmp_path / "raw.json"
    write_artifact(
        path,
        artifact_name="nonhier_semantic_raw_v1",
        support_only=False,
    )
    artifact = FrozenNonHierArtifact(
        path,
        expected_artifact_name="nonhier_semantic_raw_v1",
    )
    assert artifact.expand_edge_count == 2
    assert artifact.support_only_edge_count == 0
    assert artifact.traversal_policy == "nonhier_artifact_raw_forward"


def test_artifact_name_must_match_mode(tmp_path: Path):
    path = tmp_path / "safe.json"
    write_artifact(path)
    with pytest.raises(ValueError, match="artifact_name mismatch"):
        FrozenNonHierArtifact(
            path,
            expected_artifact_name="nonhier_semantic_raw_v1",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("same_as_included", True, "same_as_included"),
        ("isa_included", True, "isa_included"),
        ("external_cuis_included", True, "external_cuis_included"),
        ("benchmark_tuned", True, "benchmark_tuned"),
        ("direction", "reverse_target_to_source", "direction"),
        ("max_depth", 2, "max_depth"),
    ],
)
def test_artifact_rejects_out_of_scope_features(
    tmp_path: Path,
    field: str,
    value,
    message: str,
):
    path = tmp_path / "artifact.json"
    write_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        FrozenNonHierArtifact(path)


def test_expansions_preserve_edge_relation_semantics_and_policy(tmp_path: Path):
    path = tmp_path / "safe.json"
    write_artifact(path)
    artifact = FrozenNonHierArtifact(path)
    seed_rows = [
        {
            "query_term": "hypertrophy",
            "seed_cui": "C1",
            "seed_concept_name": "Hypertrophy",
            "matched_value": "Hypertrophy",
            "match_type": "exact_name",
            "lexical_weight": 3.0,
        }
    ]
    expansions = _build_expansions(seed_rows, artifact)
    assert [(x["edge_id"], x["target_cui"], x["expansion_mode"]) for x in expansions] == [
        ("NH52-001", "C2", "expand"),
        ("NH52-002", "C3", "support_only"),
    ]
    assert expansions[0]["relation_name"] == "has_associated_morphology"
    assert expansions[1]["semantic_status"] == "valid_but_broad"


def test_support_only_rows_cannot_introduce_new_sections():
    rows = [
        {"section_uid": "DIRECT", "query_term": "term"},
        {"section_uid": "EXPANDED", "query_term": "term"},
        {"section_uid": "NEW_FROM_HUB", "query_term": "term"},
    ]
    kept = _filter_support_only_rows(
        rows,
        allowed_section_uids={"DIRECT", "EXPANDED"},
    )
    assert [row["section_uid"] for row in kept] == ["DIRECT", "EXPANDED"]



def test_exact_seed_query_requires_exact_name():
    query = _nonhier_seed_query("exact_name_only")
    assert "AND match_type = 'exact_name'" in query


def test_permissive_seed_query_keeps_prefix_and_partial_available():
    query = _nonhier_seed_query("permissive")
    assert "AND match_type = 'exact_name'" not in query


def test_exact_seed_policy_rejects_prefix_and_partial_rows():
    assert _seed_row_allowed(
        {"match_type": "exact_name"},
        seed_match_policy="exact_name_only",
    )
    assert not _seed_row_allowed(
        {"match_type": "prefix"},
        seed_match_policy="exact_name_only",
    )
    assert not _seed_row_allowed(
        {"match_type": "partial"},
        seed_match_policy="exact_name_only",
    )


def test_build_expansions_exact_only_drops_weak_seed_rows(tmp_path: Path):
    path = tmp_path / "raw.json"
    write_artifact(
        path,
        artifact_name="nonhier_semantic_raw_v1",
        support_only=False,
    )
    artifact = FrozenNonHierArtifact(path)
    seeds = [
        {
            "query_term": "exact",
            "seed_cui": "C1",
            "seed_concept_name": "exact",
            "matched_value": "exact",
            "match_type": "exact_name",
            "lexical_weight": 3.0,
        },
        {
            "query_term": "weak",
            "seed_cui": "C1",
            "seed_concept_name": "weak concept",
            "matched_value": "weak concept",
            "match_type": "partial",
            "lexical_weight": 1.0,
        },
    ]
    expansions = _build_expansions(
        seeds,
        artifact,
        seed_match_policy="exact_name_only",
    )
    assert {row["query_term"] for row in expansions} == {"exact"}
