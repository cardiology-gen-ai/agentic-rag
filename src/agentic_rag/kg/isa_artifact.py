"""Read-only ISA expansion backed by a frozen data-etl discovery artifact.

This module intentionally does not read UMLS relationships from Neo4j. It is
for a controlled retrieval ablation in which:
  * direct Concept.name -> MENTIONS evidence is identical to the local baseline;
  * only canonical ``isa`` edges from a supplied collapsed-connections JSON are
    added;
  * traversal is forward only (specific -> more general) in v1;
  * SAME_AS and non-hierarchical UMLS relations are never used;
  * no graph writes occur.

The artifact remains provenance/audit data; using it for retrieval does not
imply that ISA edges have been approved for production materialization.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentic_rag.kg.candidate_generators import (
    GraphReadClientProtocol,
    KGCandidate,
    _CONCEPT_GRAPH_DIRECT_WEIGHT,
    _CONCEPT_GRAPH_EVIDENCE_RETURN,
    _CONCEPT_GRAPH_EXACT_WEIGHT,
    _CONCEPT_GRAPH_PARTIAL_WEIGHT,
    _CONCEPT_GRAPH_PREFIX_WEIGHT,
    _CONCEPT_GRAPH_SECTION_FILTER,
    _CONCEPT_GRAPH_SEED_MATCH,
    _CONCEPT_GRAPH_UMLS_WEIGHT,
    _DIRECT_CONCEPT_GRAPH_EVIDENCE,
    _EXCLUDED_TITLE_PREFIXES,
    _evidence_rows_to_results,
    _normalize_optional_values,
    _normalize_terms,
    _validate_ranking_mode,
    _validate_top_k,
    _wrap_concept_graph_results,
)
from agentic_rag.kg.models import KGRankingMode


_SUPPORTED_ISA_SEED_MATCH_POLICIES = frozenset({
    "permissive",
    "exact_name_only",
})


def _isa_seed_query(seed_match_policy: str) -> str:
    normalized = str(seed_match_policy or "").strip()
    if normalized not in _SUPPORTED_ISA_SEED_MATCH_POLICIES:
        raise ValueError(
            "Unsupported ISA seed_match_policy: "
            f"{normalized!r}"
        )
    exact_clause = (
        "\n  AND match_type = 'exact_name'"
        if normalized == "exact_name_only"
        else ""
    )
    return (
        _CONCEPT_GRAPH_SEED_MATCH
        + exact_clause
        + """
  AND trim(coalesce(seed.umls_cui, '')) <> ''
RETURN DISTINCT
    term AS query_term,
    seed.umls_cui AS seed_cui,
    coalesce(
        seed.name,
        seed.normalized_name,
        seed.umls_canonical_name,
        seed.umls_cui
    ) AS seed_concept_name,
    matched_value,
    match_type,
    toFloat(lexical_weight) AS lexical_weight
ORDER BY query_term, lexical_weight DESC, seed_concept_name, seed_cui
"""
    )


def _isa_seed_row_allowed(
    seed: dict[str, Any],
    *,
    seed_match_policy: str,
) -> bool:
    if seed_match_policy == "permissive":
        return True
    if seed_match_policy == "exact_name_only":
        return str(seed.get("match_type") or "").strip() == "exact_name"
    raise ValueError(
        "Unsupported ISA seed_match_policy: "
        f"{seed_match_policy!r}"
    )

_ISA_SECTION_EVIDENCE = (
    """
// KG_ISA_ARTIFACT_FORWARD_SECTION_EVIDENCE
UNWIND $isa_expansions AS expansion
WITH
    expansion.query_term AS term,
    expansion.seed_cui AS seed_cui,
    expansion.seed_concept_name AS seed_concept_name,
    expansion.target_cui AS target_cui,
    expansion.match_type AS match_type,
    expansion.matched_value AS matched_value,
    toFloat(expansion.lexical_weight) AS lexical_weight,
    toInteger(expansion.graph_distance) AS graph_distance
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(mentioned:Concept)
WHERE mentioned.umls_cui = target_cui
"""
    + _CONCEPT_GRAPH_SECTION_FILTER
    + """
WITH
    d,
    s,
    mentioned,
    term,
    match_type,
    matched_value,
    lexical_weight,
    'umls_neighbor' AS evidence_source,
    'UMLS_ISA_ARTIFACT' AS relation_type,
    'hierarchy_artifact_forward' AS traversal_policy,
    true AS review_needed,
    $umls_neighbor_weight AS evidence_weight,
    seed_concept_name,
    seed_cui,
    target_cui
"""
    + _CONCEPT_GRAPH_EVIDENCE_RETURN
)


class FrozenISAGraph:
    """Canonical local-CUI ISA graph loaded from a collapsed JSON artifact."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"ISA artifact not found: {self.path}")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("ISA collapsed-connections artifact must be a JSON list")

        adjacency: dict[str, set[str]] = defaultdict(set)
        retained = 0
        ignored_non_isa = 0
        ignored_invalid = 0
        for edge in raw:
            if not isinstance(edge, dict):
                ignored_invalid += 1
                continue
            relation = str(edge.get("relation_name") or "").strip().lower()
            if relation != "isa":
                ignored_non_isa += 1
                continue
            source = str(edge.get("source_cui") or "").strip()
            target = str(edge.get("target_cui") or "").strip()
            if not source or not target or source == target:
                ignored_invalid += 1
                continue
            before = len(adjacency[source])
            adjacency[source].add(target)
            if len(adjacency[source]) > before:
                retained += 1

        if retained == 0:
            raise ValueError(f"ISA artifact contains no usable canonical ISA edges: {self.path}")

        self._adjacency = {source: tuple(sorted(targets)) for source, targets in adjacency.items()}
        self.edge_count = retained
        self.ignored_non_isa_count = ignored_non_isa
        self.ignored_invalid_count = ignored_invalid
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def forward_targets(self, source_cui: str, *, max_depth: int = 1) -> list[tuple[str, int]]:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        source = str(source_cui or "").strip()
        if not source:
            return []
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        best_distance: dict[str, int] = {}
        visited = {source}
        while queue:
            current, distance = queue.popleft()
            if distance >= max_depth:
                continue
            for target in self._adjacency.get(current, ()):
                next_distance = distance + 1
                if target not in best_distance or next_distance < best_distance[target]:
                    best_distance[target] = next_distance
                if target not in visited:
                    visited.add(target)
                    queue.append((target, next_distance))
        best_distance.pop(source, None)
        return sorted(best_distance.items(), key=lambda item: (item[1], item[0]))


class ISAArtifactCandidateGenerator:
    """Direct MENTIONS + frozen forward ISA expansion for controlled ablation."""

    name = "isa_artifact_forward"

    def __init__(
        self,
        client: GraphReadClientProtocol,
        *,
        connections_path: str | Path,
        max_depth: int = 1,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
        seed_match_policy: str = "permissive",
        direct_first_graph_second: bool = False,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.client = client
        self.graph = FrozenISAGraph(connections_path)
        self.max_depth = int(max_depth)
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)
        normalized_seed_policy = str(seed_match_policy or "").strip()
        if normalized_seed_policy not in _SUPPORTED_ISA_SEED_MATCH_POLICIES:
            raise ValueError(
                "Unsupported ISA seed_match_policy: "
                f"{normalized_seed_policy!r}"
            )
        self.seed_match_policy = normalized_seed_policy
        self.direct_first_graph_second = bool(direct_first_graph_second)

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]:
        normalized_terms = _normalize_terms(terms)
        validated_top_k = _validate_top_k(top_k)
        params = {
            "terms": [term.casefold() for term in normalized_terms],
            "document_ids": _normalize_optional_values(document_ids),
            "exclude_summary_sections": self.exclude_summary_sections,
            "excluded_title_prefixes": _EXCLUDED_TITLE_PREFIXES,
            "direct_weight": _CONCEPT_GRAPH_DIRECT_WEIGHT,
            "umls_neighbor_weight": _CONCEPT_GRAPH_UMLS_WEIGHT,
            "exact_weight": _CONCEPT_GRAPH_EXACT_WEIGHT,
            "prefix_weight": _CONCEPT_GRAPH_PREFIX_WEIGHT,
            "partial_weight": _CONCEPT_GRAPH_PARTIAL_WEIGHT,
        }

        evidence_rows = list(self.client.run_read(_DIRECT_CONCEPT_GRAPH_EVIDENCE, params))
        seed_rows = list(
            self.client.run_read(
                _isa_seed_query(self.seed_match_policy),
                params,
            )
        )
        expansions: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for seed in seed_rows:
            if not _isa_seed_row_allowed(
                seed,
                seed_match_policy=self.seed_match_policy,
            ):
                continue
            seed_cui = str(seed.get("seed_cui") or "").strip()
            query_term = str(seed.get("query_term") or "").strip()
            if not seed_cui or not query_term:
                continue
            for target_cui, distance in self.graph.forward_targets(seed_cui, max_depth=self.max_depth):
                key = (query_term, seed_cui, target_cui)
                if key in seen:
                    continue
                seen.add(key)
                expansions.append({
                    "query_term": query_term,
                    "seed_cui": seed_cui,
                    "seed_concept_name": seed.get("seed_concept_name"),
                    "target_cui": target_cui,
                    "match_type": seed.get("match_type"),
                    "matched_value": seed.get("matched_value"),
                    "lexical_weight": seed.get("lexical_weight"),
                    "graph_distance": distance,
                })

        if expansions:
            isa_params = dict(params)
            isa_params["isa_expansions"] = expansions
            evidence_rows.extend(self.client.run_read(_ISA_SECTION_EVIDENCE, isa_params))

        results = _evidence_rows_to_results(
            evidence_rows,
            terms=params["terms"],
            require_all=bool(require_all),
            top_k=validated_top_k,
            ranking_mode=self.ranking_mode,
            preserve_concept_match_baseline_order=True,
            direct_first_graph_second=self.direct_first_graph_second,
        )
        candidates = _wrap_concept_graph_results(results)
        artifact_meta = {
            "isa_artifact": str(self.graph.path),
            "isa_artifact_sha256": self.graph.sha256,
            "isa_artifact_edge_count": self.graph.edge_count,
            "isa_direction": "forward_specific_to_general",
            "isa_max_depth": self.max_depth,
            "isa_nonhier_included": False,
            "same_as_included": False,
            "isa_seed_match_policy": self.seed_match_policy,
            "isa_ranking_policy": (
                "direct_first_graph_second"
                if self.direct_first_graph_second
                else "concept_match_total"
            ),
        }
        return [
            candidate.model_copy(
                update={
                    "metadata": {
                        **candidate.metadata,
                        **artifact_meta,
                    }
                }
            )
            for candidate in candidates
        ]

# ---------------------------------------------------------------------------
# ISA semantic-safe rerank-only artifact (v1)
# ---------------------------------------------------------------------------

_ISA_SAFE_SUPPORTED_ARTIFACTS = {
    ("umls_isa_retrieval_artifact_v1", "isa_semantic_safe_v1"),
    ("umls_isa_retrieval_artifact_v1_1", "isa_semantic_safe_v1_1"),
}


class FrozenISASafeGraph:
    """Policy-aware ISA graph built by data-etl semantic/traversal freeze.

    ``expand`` and ``rerank_only`` edges may contribute graph-only query
    facets, but the safe rerank-only generator never lets either class create
    new candidate Sections. ``provenance_only`` (and legacy ``support_only``)
    edges are retained for diagnostics and are ranking-neutral. ``block`` edges
    are never traversed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"ISA safe artifact not found: {self.path}")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("ISA safe artifact must be a JSON object")
        artifact_key = (
            str(raw.get("schema_version") or ""),
            str(raw.get("artifact_name") or ""),
        )
        if artifact_key not in _ISA_SAFE_SUPPORTED_ARTIFACTS:
            raise ValueError(
                "Unsupported ISA safe artifact schema/name: "
                f"{artifact_key!r}"
            )
        self.schema_version, self.artifact_name = artifact_key
        if str(raw.get("direction") or "") != "forward_specific_to_general":
            raise ValueError("ISA safe artifact requires forward_specific_to_general")
        if int(raw.get("max_depth") or 0) != 1:
            raise ValueError("ISA safe artifact requires max_depth=1")
        if raw.get("benchmark_data_used") is not False:
            raise ValueError("ISA safe artifact must be benchmark-independent")
        if raw.get("retrieval_metrics_used") is not False:
            raise ValueError("ISA safe artifact must not use retrieval metrics")

        edges = raw.get("edges")
        if not isinstance(edges, list):
            raise ValueError("ISA safe artifact edges must be a list")
        adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
        action_counts: dict[str, int] = defaultdict(int)
        seen_ids: set[str] = set()
        for edge in edges:
            if not isinstance(edge, dict):
                raise ValueError("ISA safe artifact edge must be an object")
            edge_id = str(edge.get("edge_id") or "").strip()
            source = str(edge.get("source_cui") or "").strip()
            target = str(edge.get("target_cui") or "").strip()
            action = str(edge.get("traversal_action") or "").strip()
            semantic = str(edge.get("semantic_status") or "").strip()
            if not edge_id or edge_id in seen_ids:
                raise ValueError(f"Invalid/duplicate ISA edge_id: {edge_id!r}")
            seen_ids.add(edge_id)
            if not source or not target or source == target:
                raise ValueError(f"Invalid ISA safe edge endpoints: {edge_id}")
            if action not in {"expand", "rerank_only", "provenance_only", "support_only", "block"}:
                raise ValueError(f"Invalid ISA traversal_action on {edge_id}: {action!r}")
            if semantic not in {"valid", "valid_but_broad", "invalid", "uncertain"}:
                raise ValueError(f"Invalid ISA semantic_status on {edge_id}: {semantic!r}")
            action_counts[action] += 1
            if action == "block":
                continue
            adjacency[source].append({
                "edge_id": edge_id,
                "source_cui": source,
                "target_cui": target,
                "semantic_status": semantic,
                "expansion_mode": action,
            })

        self._adjacency = {
            source: tuple(sorted(items, key=lambda x: (x["target_cui"], x["edge_id"])))
            for source, items in adjacency.items()
        }
        self.edge_count = len(edges)
        self.expand_edge_count = action_counts["expand"]
        self.rerank_only_edge_count = action_counts["rerank_only"]
        self.provenance_only_edge_count = action_counts["provenance_only"]
        self.support_only_edge_count = action_counts["support_only"]  # legacy v1
        self.block_edge_count = action_counts["block"]
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def forward_edges(self, source_cui: str) -> tuple[dict[str, Any], ...]:
        return self._adjacency.get(str(source_cui or "").strip(), ())


def _enrich_isa_safe_rows(
    rows: Sequence[dict[str, Any]],
    expansions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for expansion in expansions:
        key = (
            str(expansion.get("query_term") or ""),
            str(expansion.get("seed_cui") or ""),
            str(expansion.get("target_cui") or ""),
        )
        current = by_key.get(key)
        if current is None or str(expansion["edge_id"]) < str(current["edge_id"]):
            by_key[key] = expansion

    output: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        key = (
            str(row.get("query_term") or ""),
            str(row.get("seed_cui") or ""),
            str(row.get("target_cui") or ""),
        )
        expansion = by_key.get(key)
        if expansion is not None:
            row["artifact_edge_id"] = expansion["edge_id"]
            row["semantic_status"] = expansion["semantic_status"]
            row["expansion_mode"] = expansion["expansion_mode"]
        output.append(row)
    return output


class ISASafeRerankCandidateGenerator:
    """Exact-seed ISA reranking over the *unchanged* baseline candidate pool.

    Candidate-pool purity invariant:
        Section UIDs returned by direct MENTIONS at ``top_k`` are preserved
        exactly. ISA may only add diagnostics/support to those Sections; it can
        never inject a graph-only Section or displace a baseline candidate.
    """

    name = "isa_semantic_safe_rerank"

    def __init__(
        self,
        client: GraphReadClientProtocol,
        *,
        artifact_path: str | Path,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> None:
        self.client = client
        self.graph = FrozenISASafeGraph(artifact_path)
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]:
        normalized_terms = _normalize_terms(terms)
        validated_top_k = _validate_top_k(top_k)
        params = {
            "terms": [term.casefold() for term in normalized_terms],
            "document_ids": _normalize_optional_values(document_ids),
            "exclude_summary_sections": self.exclude_summary_sections,
            "excluded_title_prefixes": _EXCLUDED_TITLE_PREFIXES,
            "direct_weight": _CONCEPT_GRAPH_DIRECT_WEIGHT,
            "umls_neighbor_weight": _CONCEPT_GRAPH_UMLS_WEIGHT,
            "exact_weight": _CONCEPT_GRAPH_EXACT_WEIGHT,
            "prefix_weight": _CONCEPT_GRAPH_PREFIX_WEIGHT,
            "partial_weight": _CONCEPT_GRAPH_PARTIAL_WEIGHT,
        }

        direct_rows = [dict(row) for row in self.client.run_read(_DIRECT_CONCEPT_GRAPH_EVIDENCE, params)]
        direct_results = _evidence_rows_to_results(
            direct_rows,
            terms=params["terms"],
            require_all=bool(require_all),
            top_k=validated_top_k,
            ranking_mode=self.ranking_mode,
            preserve_concept_match_baseline_order=True,
        )
        allowed_uids = {result.section_uid for result in direct_results}
        if not allowed_uids:
            return []

        # Keep only rows belonging to the frozen direct top-k pool. This is the
        # source of truth for candidate-pool purity.
        evidence_rows = [
            row for row in direct_rows
            if str(row.get("section_uid") or "").strip() in allowed_uids
        ]

        seed_rows = list(self.client.run_read(_isa_seed_query("exact_name_only"), params))
        expansions: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for seed in seed_rows:
            if not _isa_seed_row_allowed(seed, seed_match_policy="exact_name_only"):
                continue
            seed_cui = str(seed.get("seed_cui") or "").strip()
            query_term = str(seed.get("query_term") or "").strip()
            if not seed_cui or not query_term:
                continue
            for edge in self.graph.forward_edges(seed_cui):
                key = (query_term, seed_cui, edge["target_cui"], edge["edge_id"])
                if key in seen:
                    continue
                seen.add(key)
                expansions.append({
                    "query_term": query_term,
                    "seed_cui": seed_cui,
                    "seed_concept_name": seed.get("seed_concept_name"),
                    "target_cui": edge["target_cui"],
                    "match_type": seed.get("match_type"),
                    "matched_value": seed.get("matched_value"),
                    "lexical_weight": seed.get("lexical_weight"),
                    "graph_distance": 1,
                    "edge_id": edge["edge_id"],
                    "semantic_status": edge["semantic_status"],
                    "expansion_mode": edge["expansion_mode"],
                })

        if expansions:
            isa_params = dict(params)
            isa_params["isa_expansions"] = expansions
            isa_rows = list(self.client.run_read(_ISA_SECTION_EVIDENCE, isa_params))
            isa_rows = _enrich_isa_safe_rows(isa_rows, expansions)
            evidence_rows.extend(
                row for row in isa_rows
                if str(row.get("section_uid") or "").strip() in allowed_uids
            )

        results = _evidence_rows_to_results(
            evidence_rows,
            terms=params["terms"],
            require_all=bool(require_all),
            top_k=validated_top_k,
            ranking_mode=self.ranking_mode,
            preserve_concept_match_baseline_order=True,
            ranking_neutral_expansion_modes=frozenset({"support_only", "provenance_only"}),
            direct_first_graph_second=True,
        )
        # Final tie-breaking must preserve baseline order, not Section uid.
        baseline_rank = {result.section_uid: rank for rank, result in enumerate(direct_results, 1)}
        results.sort(key=lambda result: (
            -(result.scores.direct_concept_match if result.scores is not None else 0.0),
            -(result.scores.graph_only_concept_match if result.scores is not None else 0.0),
            baseline_rank.get(result.section_uid, 10**9),
        ))
        results = [result.model_copy(update={"rank": rank}) for rank, result in enumerate(results, 1)]
        candidates = _wrap_concept_graph_results(results)
        meta = {
            "isa_safe_artifact": str(self.graph.path),
            "isa_safe_artifact_sha256": self.graph.sha256,
            "isa_safe_artifact_schema_version": self.graph.schema_version,
            "isa_safe_artifact_name": self.graph.artifact_name,
            "isa_safe_artifact_edge_count": self.graph.edge_count,
            "isa_safe_expand_edge_count": self.graph.expand_edge_count,
            "isa_safe_rerank_only_edge_count": self.graph.rerank_only_edge_count,
            "isa_safe_provenance_only_edge_count": self.graph.provenance_only_edge_count,
            "isa_safe_support_only_edge_count": self.graph.support_only_edge_count,
            "isa_safe_block_edge_count": self.graph.block_edge_count,
            "isa_seed_match_policy": "exact_name_only",
            "isa_ranking_policy": "direct_first_graph_second_baseline_ties",
            "isa_candidate_pool_policy": "baseline_top_k_rerank_only",
            "isa_direction": "forward_specific_to_general",
            "isa_max_depth": 1,
            "isa_uses_neo4j_umls_edges": False,
        }
        return [
            candidate.model_copy(update={"metadata": {**candidate.metadata, **meta}})
            for candidate in candidates
        ]
