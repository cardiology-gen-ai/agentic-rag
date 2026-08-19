from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.kg.isa_artifact import FrozenISAGraph, ISAArtifactCandidateGenerator


def write_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"source_cui": "C1", "relation_name": "isa", "target_cui": "C2"},
                {"source_cui": "C2", "relation_name": "isa", "target_cui": "C3"},
                {"source_cui": "C1", "relation_name": "has_focus", "target_cui": "C9"},
                {"source_cui": "C1", "relation_name": "isa", "target_cui": "C2"},
            ]
        ),
        encoding="utf-8",
    )


def test_frozen_isa_graph_is_isa_only_and_forward(tmp_path: Path):
    path = tmp_path / "isa.json"
    write_artifact(path)
    graph = FrozenISAGraph(path)
    assert graph.edge_count == 2
    assert graph.ignored_non_isa_count == 1
    assert graph.forward_targets("C1", max_depth=1) == [("C2", 1)]
    assert graph.forward_targets("C2", max_depth=1) == [("C3", 1)]
    assert graph.forward_targets("C3", max_depth=1) == []


def test_frozen_isa_graph_depth_is_explicit(tmp_path: Path):
    path = tmp_path / "isa.json"
    write_artifact(path)
    graph = FrozenISAGraph(path)
    assert graph.forward_targets("C1", max_depth=2) == [("C2", 1), ("C3", 2)]


class FakeClient:
    def __init__(self):
        self.calls = []

    def run_read(self, query, parameters=None):
        parameters = dict(parameters or {})
        self.calls.append((query, parameters))
        if "KG_CONCEPT_GRAPH_SEED_MATCH_LOCAL_ONLY" in query and "RETURN DISTINCT" in query and "seed_cui" in query:
            return [
                {
                    "query_term": "pompe disease",
                    "seed_cui": "C1",
                    "seed_concept_name": "Pompe disease",
                    "matched_value": "Pompe disease",
                    "match_type": "exact_name",
                    "lexical_weight": 3.0,
                }
            ]
        # Returning no Section rows keeps this test independent of KGSectionResult shape.
        return []


def test_generator_passes_only_frozen_forward_isa_targets(tmp_path: Path):
    path = tmp_path / "isa.json"
    write_artifact(path)
    client = FakeClient()
    generator = ISAArtifactCandidateGenerator(client, connections_path=path, max_depth=1)
    assert generator.generate("Pompe disease", top_k=10) == []
    isa_calls = [(q, p) for q, p in client.calls if "KG_ISA_ARTIFACT_FORWARD_SECTION_EVIDENCE" in q]
    assert len(isa_calls) == 1
    expansions = isa_calls[0][1]["isa_expansions"]
    assert [(x["seed_cui"], x["target_cui"], x["graph_distance"]) for x in expansions] == [("C1", "C2", 1)]
    assert all(x["target_cui"] != "C9" for x in expansions)


def test_generator_rejects_zero_depth(tmp_path: Path):
    path = tmp_path / "isa.json"
    write_artifact(path)
    with pytest.raises(ValueError, match="max_depth"):
        ISAArtifactCandidateGenerator(FakeClient(), connections_path=path, max_depth=0)
