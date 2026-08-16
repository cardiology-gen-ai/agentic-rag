"""Read-only non-hierarchical UMLS/SNOMED expansion from frozen artifacts.

Controlled ablation contract:
  * direct Concept.name -> MENTIONS evidence is always included;
  * only local-local edges present in a supplied frozen retrieval artifact are used;
  * traversal is canonical forward source_cui -> target_cui, exactly one hop;
  * SAME_AS, ISA, external CUIs, reverse traversal, multi-hop traversal, and Neo4j
    UMLS relationship reads are not used;
  * RAW artifacts allow every frozen edge to introduce Section candidates;
  * SAFE artifacts may mark selected hub edges as ``support_only``. In the v1
    mode those edges may reinforce an existing candidate; in strict v2 they are
    provenance-only and therefore ranking-neutral;
  * strict v2 graph traversal requires an exact ``Concept.name`` seed match,
    while leaving the direct MENTIONS baseline matcher unchanged;
  * no graph writes occur.

The generator preserves the pure ``mentions_only`` ordering when graph evidence
adds neither a candidate nor support for an additional query term.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
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


_NONHIER_EVIDENCE_WEIGHT = 0.5
_SUPPORTED_SCHEMA_VERSION = "umls_nonhier_retrieval_artifact_v1"
_SUPPORTED_DIRECTION = "forward_source_to_target"
_SUPPORTED_MAX_DEPTH = 1
_SUPPORTED_EXPANSION_MODES = {"expand", "support_only"}
_SUPPORTED_SEED_MATCH_POLICIES = {"permissive", "exact_name_only"}


def _nonhier_seed_query(seed_match_policy: str) -> str:
    exact_clause = (
        "\n  AND match_type = 'exact_name'"
        if seed_match_policy == "exact_name_only"
        else ""
    )
    return (
        _CONCEPT_GRAPH_SEED_MATCH
        + """
  AND trim(coalesce(seed.umls_cui, '')) <> ''
"""
        + exact_clause
        + """
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


_NONHIER_SECTION_EVIDENCE = (
    """
// KG_NONHIER_ARTIFACT_FORWARD_SECTION_EVIDENCE
UNWIND $nonhier_expansions AS expansion
WITH
    expansion.query_term AS term,
    expansion.seed_cui AS seed_cui,
    expansion.seed_concept_name AS seed_concept_name,
    expansion.target_cui AS target_cui,
    expansion.match_type AS match_type,
    expansion.matched_value AS matched_value,
    toFloat(expansion.lexical_weight) AS lexical_weight,
    expansion.relation_name AS relation_name,
    expansion.semantic_status AS semantic_status,
    expansion.edge_id AS artifact_edge_id,
    expansion.expansion_mode AS expansion_mode
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
    'nonhier_artifact' AS evidence_source,
    relation_name AS relation_type,
    $nonhier_traversal_policy AS traversal_policy,
    semantic_status = 'valid_but_broad' AS review_needed,
    $nonhier_evidence_weight AS evidence_weight,
    seed_concept_name,
    seed_cui,
    target_cui,
    artifact_edge_id,
    semantic_status,
    expansion_mode
"""
    + _CONCEPT_GRAPH_EVIDENCE_RETURN
)


@dataclass(frozen=True)
class FrozenNonHierEdge:
    edge_id: str
    source_cui: str
    target_cui: str
    relation_name: str
    semantic_status: str
    expansion_mode: str


class FrozenNonHierArtifact:
    """Validated forward-only, one-hop, local-local retrieval artifact."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_artifact_name: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Non-hier artifact not found: {self.path}")

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Non-hier retrieval artifact must be a JSON object")

        schema_version = str(raw.get("schema_version") or "").strip()
        if schema_version != _SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported non-hier artifact schema_version: "
                f"{schema_version!r}"
            )

        artifact_name = str(raw.get("artifact_name") or "").strip()
        if not artifact_name:
            raise ValueError("Non-hier artifact is missing artifact_name")
        if expected_artifact_name and artifact_name != expected_artifact_name:
            raise ValueError(
                "Non-hier artifact_name mismatch: expected "
                f"{expected_artifact_name!r}, got {artifact_name!r}"
            )

        direction = str(raw.get("direction") or "").strip()
        if direction != _SUPPORTED_DIRECTION:
            raise ValueError(
                "Non-hier v1 requires canonical forward direction "
                f"{_SUPPORTED_DIRECTION!r}; got {direction!r}"
            )

        try:
            max_depth = int(raw.get("max_depth"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Non-hier artifact max_depth must be an integer") from exc
        if max_depth != _SUPPORTED_MAX_DEPTH:
            raise ValueError(
                "Non-hier v1 requires max_depth=1; "
                f"got {max_depth}"
            )

        for flag in ("same_as_included", "isa_included", "external_cuis_included"):
            if raw.get(flag) is not False:
                raise ValueError(f"Non-hier v1 requires {flag}=false")
        if raw.get("benchmark_tuned") is not False:
            raise ValueError("Non-hier v1 requires benchmark_tuned=false")

        raw_edges = raw.get("edges")
        if not isinstance(raw_edges, list):
            raise ValueError("Non-hier artifact edges must be a JSON list")

        edges: list[FrozenNonHierEdge] = []
        seen_edge_ids: set[str] = set()
        adjacency: dict[str, list[FrozenNonHierEdge]] = defaultdict(list)

        for item in raw_edges:
            if not isinstance(item, dict):
                raise ValueError("Every non-hier artifact edge must be an object")
            edge = _parse_edge(item)
            if edge.edge_id in seen_edge_ids:
                raise ValueError(f"Duplicate non-hier edge_id: {edge.edge_id}")
            seen_edge_ids.add(edge.edge_id)
            edges.append(edge)
            adjacency[edge.source_cui].append(edge)

        declared_edge_count = raw.get("edge_count")
        if int(declared_edge_count) != len(edges):
            raise ValueError(
                "Non-hier artifact edge_count mismatch: "
                f"declared={declared_edge_count!r} actual={len(edges)}"
            )
        if not edges:
            raise ValueError("Non-hier artifact contains no usable edges")

        self.schema_version = schema_version
        self.artifact_name = artifact_name
        self.semantic_freeze_version = str(
            raw.get("semantic_freeze_version") or ""
        ).strip()
        self.traversal_version = str(raw.get("traversal_version") or "").strip()
        self.direction = direction
        self.max_depth = max_depth
        self.policy = str(raw.get("policy") or "").strip()
        self.forward_new_section_threshold = raw.get("forward_new_section_threshold")
        self.edges = tuple(edges)
        self._adjacency = {
            source: tuple(sorted(values, key=lambda edge: edge.edge_id))
            for source, values in adjacency.items()
        }
        self.edge_count = len(edges)
        self.expand_edge_count = sum(
            edge.expansion_mode == "expand" for edge in edges
        )
        self.support_only_edge_count = sum(
            edge.expansion_mode == "support_only" for edge in edges
        )
        self.support_only_edge_ids = tuple(
            edge.edge_id for edge in edges if edge.expansion_mode == "support_only"
        )
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def traversal_policy(self) -> str:
        return (
            "nonhier_artifact_safe_forward"
            if self.support_only_edge_count
            else "nonhier_artifact_raw_forward"
        )

    def forward_edges(self, source_cui: str) -> tuple[FrozenNonHierEdge, ...]:
        source = str(source_cui or "").strip()
        if not source:
            return ()
        return self._adjacency.get(source, ())


def _parse_edge(item: dict[str, Any]) -> FrozenNonHierEdge:
    edge_id = str(item.get("edge_id") or "").strip()
    source_cui = str(item.get("source_cui") or "").strip()
    target_cui = str(item.get("target_cui") or "").strip()
    relation_name = str(item.get("relation_name") or "").strip()
    semantic_status = str(item.get("semantic_status") or "").strip()
    direction = str(item.get("direction") or "").strip()
    expansion_mode = str(item.get("expansion_mode") or "").strip()

    if not edge_id or not source_cui or not target_cui or not relation_name:
        raise ValueError(f"Invalid non-hier edge fields: {item!r}")
    if source_cui == target_cui:
        raise ValueError(f"Self-loop is not allowed in non-hier artifact: {edge_id}")
    if semantic_status not in {"valid", "valid_but_broad"}:
        raise ValueError(
            f"Unsupported semantic_status for {edge_id}: {semantic_status!r}"
        )
    if direction != _SUPPORTED_DIRECTION:
        raise ValueError(
            f"Edge {edge_id} has unsupported direction {direction!r}"
        )
    if int(item.get("max_depth")) != _SUPPORTED_MAX_DEPTH:
        raise ValueError(f"Edge {edge_id} must have max_depth=1")
    if expansion_mode not in _SUPPORTED_EXPANSION_MODES:
        raise ValueError(
            f"Edge {edge_id} has unsupported expansion_mode {expansion_mode!r}"
        )

    return FrozenNonHierEdge(
        edge_id=edge_id,
        source_cui=source_cui,
        target_cui=target_cui,
        relation_name=relation_name,
        semantic_status=semantic_status,
        expansion_mode=expansion_mode,
    )


def _seed_row_allowed(
    seed: dict[str, Any],
    *,
    seed_match_policy: str,
) -> bool:
    if seed_match_policy == "permissive":
        return True
    if seed_match_policy == "exact_name_only":
        return str(seed.get("match_type") or "").strip() == "exact_name"
    raise ValueError(
        "Unsupported non-hier seed_match_policy: "
        f"{seed_match_policy!r}"
    )


def _build_expansions(
    seed_rows: Sequence[dict[str, Any]],
    artifact: FrozenNonHierArtifact,
    *,
    seed_match_policy: str = "permissive",
) -> list[dict[str, Any]]:
    expansions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for seed in seed_rows:
        if not _seed_row_allowed(seed, seed_match_policy=seed_match_policy):
            continue
        seed_cui = str(seed.get("seed_cui") or "").strip()
        query_term = str(seed.get("query_term") or "").strip()
        if not seed_cui or not query_term:
            continue
        for edge in artifact.forward_edges(seed_cui):
            key = (query_term, seed_cui, edge.edge_id)
            if key in seen:
                continue
            seen.add(key)
            expansions.append(
                {
                    "query_term": query_term,
                    "seed_cui": seed_cui,
                    "seed_concept_name": seed.get("seed_concept_name"),
                    "target_cui": edge.target_cui,
                    "match_type": seed.get("match_type"),
                    "matched_value": seed.get("matched_value"),
                    "lexical_weight": seed.get("lexical_weight"),
                    "edge_id": edge.edge_id,
                    "relation_name": edge.relation_name,
                    "semantic_status": edge.semantic_status,
                    "expansion_mode": edge.expansion_mode,
                }
            )
    return expansions


def _enrich_rows_with_expansion_metadata(
    rows: Sequence[dict[str, Any]],
    expansions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for expansion in expansions:
        key = (
            str(expansion.get("query_term") or ""),
            str(expansion.get("seed_cui") or ""),
            str(expansion.get("target_cui") or ""),
            str(expansion.get("relation_name") or ""),
        )
        # Multiple artifact edges may share endpoints and relation. Preserve a
        # deterministic representative for diagnostics; concept_match scoring
        # remains term-based and therefore does not double-count duplicates.
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
            str(row.get("relation_type") or ""),
        )
        expansion = by_key.get(key)
        if expansion is not None:
            row["artifact_edge_id"] = expansion["edge_id"]
            row["semantic_status"] = expansion["semantic_status"]
            row["expansion_mode"] = expansion["expansion_mode"]
        output.append(row)
    return output


def _filter_support_only_rows(
    rows: Sequence[dict[str, Any]],
    *,
    allowed_section_uids: set[str],
) -> list[dict[str, Any]]:
    """Keep support-only evidence only for already-existing candidates."""
    return [
        dict(row)
        for row in rows
        if str(row.get("section_uid") or "").strip() in allowed_section_uids
    ]


class NonHierArtifactCandidateGenerator:
    """Direct MENTIONS + frozen one-hop non-hier expansion."""

    name = "nonhier_artifact_forward"

    def __init__(
        self,
        client: GraphReadClientProtocol,
        *,
        artifact_path: str | Path,
        expected_artifact_name: str,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
        seed_match_policy: str = "permissive",
        support_only_ranking_active: bool = True,
        direct_first_graph_second: bool = False,
    ) -> None:
        self.client = client
        self.artifact = FrozenNonHierArtifact(
            artifact_path,
            expected_artifact_name=expected_artifact_name,
        )
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)
        normalized_seed_policy = str(seed_match_policy or "").strip()
        if normalized_seed_policy not in _SUPPORTED_SEED_MATCH_POLICIES:
            raise ValueError(
                "Unsupported non-hier seed_match_policy: "
                f"{normalized_seed_policy!r}"
            )
        self.seed_match_policy = normalized_seed_policy
        self.support_only_ranking_active = bool(support_only_ranking_active)
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
            "nonhier_evidence_weight": _NONHIER_EVIDENCE_WEIGHT,
            "exact_weight": _CONCEPT_GRAPH_EXACT_WEIGHT,
            "prefix_weight": _CONCEPT_GRAPH_PREFIX_WEIGHT,
            "partial_weight": _CONCEPT_GRAPH_PARTIAL_WEIGHT,
            "nonhier_traversal_policy": self.artifact.traversal_policy,
        }

        direct_rows = list(
            self.client.run_read(_DIRECT_CONCEPT_GRAPH_EVIDENCE, params)
        )
        evidence_rows: list[dict[str, Any]] = [dict(row) for row in direct_rows]
        direct_section_uids = {
            str(row.get("section_uid") or "").strip()
            for row in direct_rows
            if str(row.get("section_uid") or "").strip()
        }

        seed_rows = list(
            self.client.run_read(
                _nonhier_seed_query(self.seed_match_policy),
                params,
            )
        )
        expansions = _build_expansions(
            seed_rows,
            self.artifact,
            seed_match_policy=self.seed_match_policy,
        )
        expand_expansions = [
            item for item in expansions if item["expansion_mode"] == "expand"
        ]
        support_expansions = [
            item for item in expansions if item["expansion_mode"] == "support_only"
        ]

        if expand_expansions:
            expand_params = dict(params)
            expand_params["nonhier_expansions"] = expand_expansions
            expand_rows = list(
                self.client.run_read(_NONHIER_SECTION_EVIDENCE, expand_params)
            )
            expand_rows = _enrich_rows_with_expansion_metadata(
                expand_rows,
                expand_expansions,
            )
            evidence_rows.extend(expand_rows)
        else:
            expand_rows = []

        # SAFE support-only edges may reinforce direct/expandable candidates,
        # but cannot create a Section candidate by themselves.
        allowed_support_section_uids = set(direct_section_uids)
        allowed_support_section_uids.update(
            str(row.get("section_uid") or "").strip()
            for row in expand_rows
            if str(row.get("section_uid") or "").strip()
        )

        if support_expansions and allowed_support_section_uids:
            support_params = dict(params)
            support_params["nonhier_expansions"] = support_expansions
            support_rows = list(
                self.client.run_read(_NONHIER_SECTION_EVIDENCE, support_params)
            )
            support_rows = _enrich_rows_with_expansion_metadata(
                support_rows,
                support_expansions,
            )
            evidence_rows.extend(
                _filter_support_only_rows(
                    support_rows,
                    allowed_section_uids=allowed_support_section_uids,
                )
            )

        results = _evidence_rows_to_results(
            evidence_rows,
            terms=params["terms"],
            require_all=bool(require_all),
            top_k=validated_top_k,
            ranking_mode=self.ranking_mode,
            preserve_concept_match_baseline_order=True,
            ranking_neutral_expansion_modes=(
                frozenset({"support_only"})
                if not self.support_only_ranking_active
                else None
            ),
            direct_first_graph_second=self.direct_first_graph_second,
        )
        candidates = _wrap_concept_graph_results(results)

        artifact_meta = {
            "nonhier_artifact": str(self.artifact.path),
            "nonhier_artifact_sha256": self.artifact.sha256,
            "nonhier_artifact_name": self.artifact.artifact_name,
            "nonhier_artifact_edge_count": self.artifact.edge_count,
            "nonhier_expand_edge_count": self.artifact.expand_edge_count,
            "nonhier_support_only_edge_count": self.artifact.support_only_edge_count,
            "nonhier_support_only_edge_ids": list(
                self.artifact.support_only_edge_ids
            ),
            "nonhier_direction": self.artifact.direction,
            "nonhier_max_depth": self.artifact.max_depth,
            "nonhier_traversal_policy": self.artifact.traversal_policy,
            "nonhier_seed_match_policy": self.seed_match_policy,
            "nonhier_support_only_ranking_active": (
                self.support_only_ranking_active
            ),
            "nonhier_ranking_policy": (
                "direct_first_graph_second"
                if self.direct_first_graph_second
                else "concept_match_total"
            ),
            "nonhier_same_as_included": False,
            "nonhier_isa_included": False,
            "nonhier_external_cuis_included": False,
            "nonhier_uses_neo4j_umls_edges": False,
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
