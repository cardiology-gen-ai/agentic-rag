from __future__ import annotations

import json

from agentic_rag.kg.isa_artifact import FrozenISASafeGraph, ISASafeRerankCandidateGenerator
from agentic_rag.kg.pipeline import _validate_mode


def _row(uid: str, term: str, *, evidence_source="direct", target_cui="CSEED"):
    return {
        "section_uid": uid,
        "document_id": "Cardiomyopathies_2023",
        "section_id": uid,
        "printed_section_id": uid,
        "title": f"Section {uid}",
        "level": 3,
        "text": "non-empty section text",
        "page_start": 1,
        "page_end": 1,
        "part_index": 0,
        "part_count": 1,
        "retrieval_unit_id": f"{uid}::retrieval",
        "section_view_schema_version": "1",
        "section_view_role": "retrieval",
        "retrieval_strategy": "max_level_4",
        "aggregation_mode": "merge_below_level",
        "is_aggregated": False,
        "content_owner_section_id": uid,
        "source_section_ids": [uid],
        "source_chunk_ids": [uid],
        "represented_section_ids": [uid],
        "structural_context_section_ids": [],
        "absorbed_section_ids": [],
        "absorbed_source_section_ids": [],
        "query_term": term,
        "concept_name": term,
        "matched_value": term,
        "match_type": "exact_name",
        "lexical_weight": 3.0,
        "evidence_source": evidence_source,
        "relation_type": "MENTIONS" if evidence_source == "direct" else "UMLS_ISA_ARTIFACT",
        "traversal_policy": None if evidence_source == "direct" else "hierarchy_artifact_forward",
        "review_needed": evidence_source != "direct",
        "evidence_weight": 1.0 if evidence_source == "direct" else 0.5,
        "seed_concept_name": term,
        "seed_cui": "CSEED",
        "target_cui": target_cui,
    }


def _artifact(tmp_path, action="expand", *, legacy=False):
    p = tmp_path / "isa.json"
    if legacy:
        schema = "umls_isa_retrieval_artifact_v1"
        name = "isa_semantic_safe_v1"
    else:
        schema = "umls_isa_retrieval_artifact_v1_1"
        name = "isa_semantic_safe_v1_1"
    p.write_text(json.dumps({
        "schema_version": schema,
        "artifact_name": name,
        "document_id": "Cardiomyopathies_2023",
        "direction": "forward_specific_to_general",
        "max_depth": 1,
        "benchmark_data_used": False,
        "retrieval_metrics_used": False,
        "edges": [{
            "edge_id": "ISA149-001",
            "source_cui": "CSEED",
            "source_names": ["specific"],
            "source_type": "disease",
            "target_cui": "CTARGET",
            "target_names": ["general"],
            "target_type": "disease",
            "relation_name": "isa",
            "semantic_status": "valid" if action != "block" else "invalid",
            "traversal_action": action,
        }],
    }))
    return p


class FakeClient:
    def run_read(self, query, params):
        if "KG_ISA_ARTIFACT_FORWARD_SECTION_EVIDENCE" in query:
            # C is graph-only and must never enter the safe rerank candidate pool.
            return [
                _row("B", "term two", evidence_source="umls_neighbor", target_cui="CTARGET"),
                _row("C", "term two", evidence_source="umls_neighbor", target_cui="CTARGET"),
            ]
        if "AND trim(coalesce(seed.umls_cui" in query:
            return [{
                "query_term": "term two",
                "seed_cui": "CSEED",
                "seed_concept_name": "specific",
                "matched_value": "specific",
                "match_type": "exact_name",
                "lexical_weight": 3.0,
            }]
        if "WHERE mentioned = seed" in query:
            return [_row("A", "term one"), _row("B", "term one")]
        raise AssertionError("unexpected query")


def test_safe_artifact_mode_registered():
    assert _validate_mode("mentions_isa_semantic_safe_rerank") == "mentions_isa_semantic_safe_rerank"


def test_safe_graph_accepts_v11_and_legacy_v1(tmp_path):
    assert FrozenISASafeGraph(_artifact(tmp_path)).artifact_name == "isa_semantic_safe_v1_1"
    assert FrozenISASafeGraph(_artifact(tmp_path, action="support_only", legacy=True)).artifact_name == "isa_semantic_safe_v1"


def test_safe_graph_rejects_benchmark_tuned_artifact(tmp_path):
    p = _artifact(tmp_path)
    x = json.loads(p.read_text())
    x["benchmark_data_used"] = True
    p.write_text(json.dumps(x))
    try:
        FrozenISASafeGraph(p)
    except ValueError as exc:
        assert "benchmark" in str(exc).lower()
    else:
        raise AssertionError("benchmark-dependent artifact should be rejected")


def test_expand_rerank_preserves_exact_baseline_candidate_pool(tmp_path):
    g = ISASafeRerankCandidateGenerator(FakeClient(), artifact_path=_artifact(tmp_path, action="expand"))
    out = g.generate(["term one", "term two"], top_k=2)
    assert {c.section_uid for c in out} == {"A", "B"}
    assert "C" not in {c.section_uid for c in out}
    assert [c.section_uid for c in out] == ["B", "A"]
    assert out[0].section.scores.graph_only_concept_match == 1.0


def test_rerank_only_contributes_graph_facet_without_candidate_injection(tmp_path):
    g = ISASafeRerankCandidateGenerator(
        FakeClient(), artifact_path=_artifact(tmp_path, action="rerank_only")
    )
    out = g.generate(["term one", "term two"], top_k=2)
    assert [c.section_uid for c in out] == ["B", "A"]
    assert "C" not in {c.section_uid for c in out}
    assert out[0].section.scores.graph_only_concept_match == 1.0
    assert any(d.expansion_mode == "rerank_only" for d in out[0].section.match_diagnostics)


def test_provenance_only_is_ranking_neutral(tmp_path):
    g = ISASafeRerankCandidateGenerator(
        FakeClient(), artifact_path=_artifact(tmp_path, action="provenance_only")
    )
    out = g.generate(["term one", "term two"], top_k=2)
    assert [c.section_uid for c in out] == ["A", "B"]
    assert out[1].section.scores.graph_only_concept_match == 0.0
    assert any(d.expansion_mode == "provenance_only" for d in out[1].section.match_diagnostics)


def test_legacy_support_only_remains_ranking_neutral(tmp_path):
    g = ISASafeRerankCandidateGenerator(
        FakeClient(), artifact_path=_artifact(tmp_path, action="support_only", legacy=True)
    )
    out = g.generate(["term one", "term two"], top_k=2)
    assert [c.section_uid for c in out] == ["A", "B"]
    assert out[1].section.scores.graph_only_concept_match == 0.0
