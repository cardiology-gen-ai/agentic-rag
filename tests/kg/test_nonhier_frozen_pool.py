import json
from pathlib import Path

import pytest

from agentic_rag.kg.nonhier_artifact import (
    FrozenNonHierArtifact,
    NonHierArtifactCandidateGenerator,
)
from agentic_rag.kg.pipeline import _validate_mode


def _artifact(path: Path, *, expansion_mode: str):
    path.write_text(json.dumps({
        "schema_version":"umls_nonhier_retrieval_artifact_v1",
        "semantic_freeze_version":"test",
        "traversal_version":"test",
        "edge_count":1,
        "direction":"forward_source_to_target",
        "max_depth":1,
        "same_as_included":False,
        "isa_included":False,
        "external_cuis_included":False,
        "benchmark_tuned":False,
        "artifact_name":"nonhier_semantic_safe_v1",
        "policy":"test",
        "edges":[{
            "edge_id":"E1",
            "source_cui":"C1",
            "target_cui":"C2",
            "relation_name":"r",
            "semantic_status":"valid",
            "direction":"forward_source_to_target",
            "max_depth":1,
            "expansion_mode":expansion_mode,
        }],
    }))


def test_frozen_mode_is_registered():
    assert _validate_mode(
        "mentions_nonhier_artifact_safe_strict_direct_first_frozen"
    ) == "mentions_nonhier_artifact_safe_strict_direct_first_frozen"


def test_support_only_artifact_is_compatible_with_frozen_pool(tmp_path: Path):
    path=tmp_path/"a.json"
    _artifact(path, expansion_mode="support_only")
    artifact=FrozenNonHierArtifact(path, expected_artifact_name="nonhier_semantic_safe_v1")
    assert artifact.expand_edge_count == 0
    assert artifact.support_only_edge_count == 1


class _Dummy:
    pass


def test_frozen_generator_rejects_expand_edges(tmp_path: Path):
    path=tmp_path/"expand.json"
    _artifact(path, expansion_mode="expand")
    with pytest.raises(ValueError, match="support_only"):
        NonHierArtifactCandidateGenerator(
            _Dummy(),
            artifact_path=path,
            expected_artifact_name="nonhier_semantic_safe_v1",
            seed_match_policy="exact_name_only",
            support_only_ranking_active=True,
            direct_first_graph_second=True,
            freeze_direct_candidate_pool=True,
            baseline_tools=_Dummy(),
        )


def test_frozen_generator_accepts_support_only_edges(tmp_path: Path):
    path=tmp_path/"support.json"
    _artifact(path, expansion_mode="support_only")
    generator=NonHierArtifactCandidateGenerator(
        _Dummy(),
        artifact_path=path,
        expected_artifact_name="nonhier_semantic_safe_v1",
        seed_match_policy="exact_name_only",
        support_only_ranking_active=True,
        direct_first_graph_second=True,
        freeze_direct_candidate_pool=True,
        baseline_tools=_Dummy(),
    )
    assert generator.freeze_direct_candidate_pool is True
    assert generator.support_only_ranking_active is True
