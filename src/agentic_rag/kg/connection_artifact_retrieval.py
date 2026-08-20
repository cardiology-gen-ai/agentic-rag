"""Candidate generation from frozen DIRECT and ontology-BRIDGE artifacts.

This module implements the final controlled connection ablation without adding
new ontology calls or graph writes.  It deliberately reuses the existing
MENTIONS seed matcher and concept-match ranking used by the modular KG pipeline.

The configuration file determines which frozen connection artifacts are active
for a named mode.  DIRECT neighbours are taken exactly from the selected direct
profile.  BRIDGE neighbours use source-specific profiles and an independent
top-N quota per source.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from agentic_rag.kg.candidate_generators import (
    GraphReadClientProtocol,
    KGCandidate,
    KGSectionSearchProtocol,
    SeededMentionsCandidateGenerator,
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
from agentic_rag.kg.concept_seeders import (
    ConceptSeed,
    ConceptSeederProtocol,
)
from agentic_rag.kg.direct_local_artifact import (
    DirectNeighbor,
    FrozenDirectLocalArtifact,
)
from agentic_rag.kg.models import KGRankingMode, KGRetrievalScores
from agentic_rag.kg.ontology_bridge_artifact import (
    BridgeNeighbor,
    FrozenOntologyBridgeArtifact,
)

_SUPPORTED_SCHEMA = "kg_connection_ablation_v1"
_CONNECTION_EVIDENCE_WEIGHT = 0.5
_SUPPORTED_CONNECTION_SOURCES = {
    "direct_local_artifact",
    "ontology_bridge_artifact",
}

_CONNECTION_SEED_QUERY = (
    _CONCEPT_GRAPH_SEED_MATCH
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

_CONNECTION_SECTION_EVIDENCE = (
    """
// KG_FROZEN_CONNECTION_ARTIFACT_SECTION_EVIDENCE
UNWIND $connection_expansions AS expansion
WITH
    expansion.query_term AS term,
    expansion.seed_cui AS seed_cui,
    expansion.seed_concept_name AS seed_concept_name,
    expansion.target_cui AS target_cui,
    expansion.match_type AS match_type,
    expansion.matched_value AS matched_value,
    toFloat(expansion.lexical_weight) AS lexical_weight,
    expansion.evidence_source AS evidence_source,
    expansion.relation_type AS relation_type,
    expansion.traversal_policy AS traversal_policy,
    expansion.artifact_edge_id AS artifact_edge_id,
    expansion.semantic_status AS semantic_status,
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
    evidence_source,
    relation_type,
    traversal_policy,
    false AS review_needed,
    $connection_evidence_weight AS evidence_weight,
    seed_concept_name,
    seed_cui,
    target_cui,
    artifact_edge_id,
    semantic_status,
    expansion_mode
"""
    + _CONCEPT_GRAPH_EVIDENCE_RETURN
)


class FrozenConnectionAblationConfig:
    """Validated experiment/configuration file for connection modes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Connection ablation config not found: {self.path}")

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Connection ablation config must be a JSON object")
        if str(raw.get("schema_version") or "").strip() != _SUPPORTED_SCHEMA:
            raise ValueError(
                "Unsupported connection ablation schema_version: "
                f"{raw.get('schema_version')!r}"
            )

        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("Connection ablation config requires an artifacts object")
        direct_path = str(artifacts.get("direct_artifact_dir") or "").strip()
        bridge_path = str(artifacts.get("bridge_artifact_dir") or "").strip()
        if not direct_path or not bridge_path:
            raise ValueError(
                "artifacts.direct_artifact_dir and bridge_artifact_dir are required"
            )

        self.direct_artifact_dir = Path(direct_path).expanduser().resolve()
        self.bridge_artifact_dir = Path(bridge_path).expanduser().resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.raw = raw

        modes = raw.get("connection_modes")
        if not isinstance(modes, dict) or not modes:
            raise ValueError(
                "Connection ablation config requires a non-empty connection_modes object"
            )
        self.connection_modes = {
            str(name).strip(): dict(value)
            for name, value in modes.items()
            if str(name).strip() and isinstance(value, dict)
        }
        if len(self.connection_modes) != len(modes):
            raise ValueError("Invalid connection mode name or payload")

        for mode_name, payload in self.connection_modes.items():
            self._validate_mode(mode_name, payload)

    def _validate_mode(self, name: str, payload: Mapping[str, Any]) -> None:
        direct_cfg = payload.get("direct")
        bridge_cfg = payload.get("bridge")
        if direct_cfg is None and bridge_cfg is None:
            raise ValueError(
                f"Connection mode {name!r} must enable direct and/or bridge"
            )

        if direct_cfg is not None:
            if not isinstance(direct_cfg, dict):
                raise ValueError(f"{name}.direct must be an object or null")
            profile = str(direct_cfg.get("profile") or "").strip()
            sources = direct_cfg.get("sources")
            if profile not in {"strong", "balanced", "broad"}:
                raise ValueError(f"Invalid direct profile in {name}: {profile!r}")
            if not isinstance(sources, list) or not all(
                str(source).strip() for source in sources
            ):
                raise ValueError(f"{name}.direct.sources must be a non-empty list")

        if bridge_cfg is not None:
            if not isinstance(bridge_cfg, dict):
                raise ValueError(f"{name}.bridge must be an object or null")
            profiles = bridge_cfg.get("source_profiles")
            if not isinstance(profiles, dict) or not profiles:
                raise ValueError(
                    f"{name}.bridge.source_profiles must be a non-empty object"
                )
            for source, profile in profiles.items():
                if not str(source).strip():
                    raise ValueError(f"Empty bridge source in {name}")
                if str(profile).strip() not in {"strong", "balanced", "broad"}:
                    raise ValueError(
                        f"Invalid bridge profile in {name}: {source}={profile!r}"
                    )
            try:
                top_n = int(bridge_cfg.get("top_n"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}.bridge.top_n must be an integer") from exc
            if top_n < 1:
                raise ValueError(f"{name}.bridge.top_n must be >= 1")
            ranking_policy = str(
                bridge_cfg.get("ranking_policy") or "tier_first"
            ).strip()
            if ranking_policy not in {"tier_first", "score_only"}:
                raise ValueError(
                    f"{name}.bridge.ranking_policy must be tier_first or score_only"
                )

    def mode(self, name: str) -> dict[str, Any]:
        normalized = str(name or "").strip()
        try:
            return dict(self.connection_modes[normalized])
        except KeyError as exc:
            raise KeyError(
                f"Connection mode {normalized!r} is not defined in {self.path}"
            ) from exc


def _build_direct_expansions(
    seed_rows: Sequence[dict[str, Any]],
    adjacency: dict[str, tuple[DirectNeighbor, ...]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for seed in seed_rows:
        query_term = str(seed.get("query_term") or "").strip()
        seed_cui = str(seed.get("seed_cui") or "").strip()
        if not query_term or not seed_cui:
            continue
        for neighbor in adjacency.get(seed_cui, ()):
            key = (
                query_term.casefold(),
                seed_cui,
                neighbor.neighbor_cui,
                neighbor.direct_id,
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "query_term": query_term,
                    "seed_cui": seed_cui,
                    "seed_concept_name": seed.get("seed_concept_name"),
                    "target_cui": neighbor.neighbor_cui,
                    "match_type": seed.get("match_type"),
                    "matched_value": seed.get("matched_value"),
                    "lexical_weight": seed.get("lexical_weight"),
                    "evidence_source": "direct_local_artifact",
                    "relation_type": "|".join(neighbor.relation_families)
                    or "direct_semantic_pair",
                    "traversal_policy": "bidirectional_semantic_pair_depth1",
                    "artifact_edge_id": neighbor.direct_id,
                    "semantic_status": "valid",
                    "expansion_mode": "expand",
                }
            )
    return output


def _build_bridge_expansions(
    seed_rows: Sequence[dict[str, Any]],
    source_adjacencies: Mapping[
        str, dict[str, tuple[BridgeNeighbor, ...]]
    ],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Apply an independent top-N quota within each bridge source."""
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for source, adjacency in source_adjacencies.items():
        for seed in seed_rows:
            query_term = str(seed.get("query_term") or "").strip()
            seed_cui = str(seed.get("seed_cui") or "").strip()
            if not query_term or not seed_cui:
                continue

            for neighbor in adjacency.get(seed_cui, ())[:top_n]:
                key = (
                    query_term.casefold(),
                    seed_cui,
                    neighbor.neighbor_cui,
                    neighbor.bridge_id,
                )
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    {
                        "query_term": query_term,
                        "seed_cui": seed_cui,
                        "seed_concept_name": seed.get("seed_concept_name"),
                        "target_cui": neighbor.neighbor_cui,
                        "match_type": seed.get("match_type"),
                        "matched_value": seed.get("matched_value"),
                        "lexical_weight": seed.get("lexical_weight"),
                        "evidence_source": "ontology_bridge_artifact",
                        "relation_type": "ontology_bridge",
                        "traversal_policy": (
                            f"source_aware_{source}_top{top_n}_depth1"
                        ),
                        "artifact_edge_id": neighbor.bridge_id,
                        "semantic_status": "valid",
                        "expansion_mode": "expand",
                    }
                )
    return output


def _enrich_rows_with_expansion_metadata(
    rows: Sequence[dict[str, Any]],
    expansions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore diagnostic fields omitted by the shared Cypher return template."""
    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for expansion in expansions:
        key = (
            str(expansion.get("query_term") or "").casefold(),
            str(expansion.get("seed_cui") or ""),
            str(expansion.get("target_cui") or ""),
            str(expansion.get("evidence_source") or ""),
        )
        by_key.setdefault(key, []).append(expansion)

    output: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("query_term") or "").casefold(),
            str(row.get("seed_cui") or ""),
            str(row.get("target_cui") or ""),
            str(row.get("evidence_source") or ""),
        )
        candidates = by_key.get(key, [])
        if not candidates:
            output.append(dict(row))
            continue

        # Multiple paths to the same semantic pair are intentionally not given
        # multiple ranking votes.  Preserve one deterministic provenance record.
        chosen = sorted(
            candidates,
            key=lambda item: (
                str(item.get("artifact_edge_id") or ""),
                str(item.get("relation_type") or ""),
            ),
        )[0]
        enriched = dict(row)
        enriched.update(
            {
                "artifact_edge_id": chosen.get("artifact_edge_id"),
                "semantic_status": chosen.get("semantic_status"),
                "expansion_mode": chosen.get("expansion_mode"),
            }
        )
        output.append(enriched)
    return output


class ConnectionArtifactCandidateGenerator:
    """MENTIONS plus frozen DIRECT/BRIDGE local-CUI expansion."""

    name = "connection_artifact"

    def __init__(
        self,
        client: GraphReadClientProtocol,
        *,
        config_path: str | Path,
        mode_name: str,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> None:
        self.client = client
        self.config = FrozenConnectionAblationConfig(config_path)
        self.mode_name = str(mode_name).strip()
        self.mode_config = self.config.mode(self.mode_name)
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)

        direct_cfg = self.mode_config.get("direct")
        self.direct_artifact: FrozenDirectLocalArtifact | None = None
        self.direct_adjacency: dict[str, tuple[DirectNeighbor, ...]] = {}
        if direct_cfg is not None:
            self.direct_artifact = FrozenDirectLocalArtifact(
                self.config.direct_artifact_dir
            )
            self.direct_adjacency = self.direct_artifact.build_adjacency(
                profile=str(direct_cfg["profile"]),
                sources=[str(v) for v in direct_cfg["sources"]],
            )

        bridge_cfg = self.mode_config.get("bridge")
        self.bridge_artifact: FrozenOntologyBridgeArtifact | None = None
        self.bridge_source_adjacencies: dict[
            str, dict[str, tuple[BridgeNeighbor, ...]]
        ] = {}
        self.bridge_top_n: int | None = None
        if bridge_cfg is not None:
            self.bridge_artifact = FrozenOntologyBridgeArtifact(
                self.config.bridge_artifact_dir
            )
            ranking_policy = str(
                bridge_cfg.get("ranking_policy") or "tier_first"
            )
            self.bridge_top_n = int(bridge_cfg["top_n"])
            self.bridge_source_adjacencies = {
                str(source): self.bridge_artifact.build_adjacency(
                    profile=str(profile),
                    sources=[str(source)],
                    ranking_policy=ranking_policy,
                )
                for source, profile in bridge_cfg["source_profiles"].items()
            }

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
            "exact_weight": _CONCEPT_GRAPH_EXACT_WEIGHT,
            "prefix_weight": _CONCEPT_GRAPH_PREFIX_WEIGHT,
            "partial_weight": _CONCEPT_GRAPH_PARTIAL_WEIGHT,
        }

        direct_rows = list(
            self.client.run_read(_DIRECT_CONCEPT_GRAPH_EVIDENCE, params)
        )
        evidence_rows: list[dict[str, Any]] = [dict(row) for row in direct_rows]

        seed_rows = list(self.client.run_read(_CONNECTION_SEED_QUERY, params))
        expansions: list[dict[str, Any]] = []
        if self.direct_artifact is not None:
            expansions.extend(
                _build_direct_expansions(seed_rows, self.direct_adjacency)
            )
        if self.bridge_artifact is not None:
            assert self.bridge_top_n is not None
            expansions.extend(
                _build_bridge_expansions(
                    seed_rows,
                    self.bridge_source_adjacencies,
                    top_n=self.bridge_top_n,
                )
            )

        if expansions:
            expansion_params = dict(params)
            expansion_params.update(
                {
                    "connection_expansions": expansions,
                    "connection_evidence_weight": _CONNECTION_EVIDENCE_WEIGHT,
                }
            )
            expanded_rows = list(
                self.client.run_read(
                    _CONNECTION_SECTION_EVIDENCE,
                    expansion_params,
                )
            )
            evidence_rows.extend(
                _enrich_rows_with_expansion_metadata(
                    expanded_rows,
                    expansions,
                )
            )

        results = _evidence_rows_to_results(
            evidence_rows,
            terms=params["terms"],
            require_all=bool(require_all),
            top_k=validated_top_k,
            ranking_mode=self.ranking_mode,
            preserve_concept_match_baseline_order=True,
        )
        candidates = _wrap_concept_graph_results(results)

        metadata = {
            "connection_config": str(self.config.path),
            "connection_config_sha256": self.config.sha256,
            "connection_mode": self.mode_name,
            "connection_mode_config": self.mode_config,
            "direct_artifact_dir": (
                str(self.direct_artifact.artifact_dir)
                if self.direct_artifact is not None
                else None
            ),
            "direct_artifact_sha256": (
                self.direct_artifact.sha256
                if self.direct_artifact is not None
                else None
            ),
            "bridge_artifact_dir": (
                str(self.bridge_artifact.artifact_dir)
                if self.bridge_artifact is not None
                else None
            ),
            "bridge_artifact_sha256": (
                self.bridge_artifact.sha256
                if self.bridge_artifact is not None
                else None
            ),
            "connection_max_depth": 1,
            "connection_second_hop": False,
            "connection_external_cuis_materialized": False,
        }
        return [
            candidate.model_copy(
                update={
                    "metadata": {
                        **candidate.metadata,
                        **metadata,
                    }
                }
            )
            for candidate in candidates
        ]


def _seed_rows_from_concept_seeds(
    seeds: Sequence[ConceptSeed],
    *,
    cui_by_concept_name: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    """Resolve trusted CUIs *after* semantic Concept selection.

    ``ConceptSeed`` intentionally remains UMLS-agnostic so that embedding
    seeding is identical to the validated baseline.  Once a local Concept has
    been selected, this helper resolves its CUI from the current local Concept
    catalogue.  The CUI can therefore activate DIRECT/BRIDGE expansion without
    participating in seed selection itself.
    """
    output: list[dict[str, Any]] = []
    for seed in seeds:
        cui = str(cui_by_concept_name.get(seed.concept_name) or "").strip()
        if not cui:
            continue
        similarity = (
            float(seed.similarity)
            if seed.similarity is not None
            else 0.0
        )
        output.append(
            {
                "query_term": seed.query_term,
                "seed_cui": cui,
                "seed_concept_name": seed.concept_name,
                "match_type": seed.match_type,
                "matched_value": seed.matched_value or seed.concept_name,
                # Reuse the shared evidence field as semantic seed confidence.
                # No lexical matching is performed in this generator.
                "lexical_weight": similarity,
            }
        )
    return output


def _semantic_value_for_diagnostic(diagnostic: Any) -> float:
    similarity = getattr(diagnostic, "similarity", None)
    if similarity is not None:
        return float(similarity)
    source = str(getattr(diagnostic, "evidence_source", None) or "")
    if source in {"direct_local_artifact", "ontology_bridge_artifact"}:
        value = getattr(diagnostic, "lexical_weight", None)
        return float(value) if value is not None else 0.0
    return 0.0


def _merge_semantic_connection_candidates(
    local_candidates: Sequence[KGCandidate],
    graph_candidates: Sequence[KGCandidate],
    *,
    top_k: int | None,
    metadata: Mapping[str, Any],
) -> list[KGCandidate]:
    """Rank local and graph-supported Sections with one coefficient-free score.

    For each query term, keep the strongest semantic seed similarity that
    supports the Section, whether through direct MENTIONS or through a frozen
    DIRECT/BRIDGE path. Multiple graph paths for the same term do not stack.
    Ties prefer Sections also supported by local MENTIONS, then the original
    semantic-local rank.
    """
    grouped: dict[str, dict[str, Any]] = {}

    def add(candidate: KGCandidate, *, is_local: bool) -> None:
        record = grouped.setdefault(
            candidate.section_uid,
            {
                "base": candidate,
                "diagnostics": [],
                "matched_concepts": [],
                "matched_terms": [],
                "local_rank": 10**9,
                "has_local": False,
                "has_graph": False,
            },
        )
        if is_local:
            if not record["has_local"]:
                record["base"] = candidate
            record["has_local"] = True
            record["local_rank"] = min(
                record["local_rank"],
                int(candidate.source_rank),
            )
        else:
            record["has_graph"] = True

        record["diagnostics"].extend(candidate.section.match_diagnostics)
        record["matched_concepts"].extend(candidate.section.matched_concepts)
        record["matched_terms"].extend(candidate.section.matched_terms)

    for candidate in local_candidates:
        add(candidate, is_local=True)
    for candidate in graph_candidates:
        add(candidate, is_local=False)

    scored: list[tuple[KGCandidate, float, int, int, int]] = []

    for record in grouped.values():
        diagnostics = []
        seen_diagnostics: set[str] = set()
        for diagnostic in record["diagnostics"]:
            payload = (
                diagnostic.model_dump(mode="json")
                if hasattr(diagnostic, "model_dump")
                else diagnostic
            )
            key = json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            if key in seen_diagnostics:
                continue
            seen_diagnostics.add(key)
            diagnostics.append(diagnostic)

        per_term: dict[str, float] = {}
        local_terms: set[str] = set()
        graph_terms: set[str] = set()

        for diagnostic in diagnostics:
            term = str(
                getattr(diagnostic, "query_term", "") or ""
            ).strip()
            if not term:
                continue
            term_key = term.casefold()
            value = _semantic_value_for_diagnostic(diagnostic)
            per_term[term_key] = max(per_term.get(term_key, 0.0), value)

            source = str(
                getattr(diagnostic, "evidence_source", None) or ""
            )
            if source in {"direct_local_artifact", "ontology_bridge_artifact"}:
                graph_terms.add(term_key)
            else:
                # Explicit semantic Concept -> MENTIONS evidence.
                local_terms.add(term_key)

        score = float(sum(per_term.values()))
        graph_only_terms = graph_terms - local_terms
        base: KGCandidate = record["base"]
        base_scores = base.section.scores or KGRetrievalScores()
        updated_section = base.section.model_copy(
            update={
                "matched_concepts": list(
                    dict.fromkeys(record["matched_concepts"])
                ),
                "matched_terms": list(dict.fromkeys(record["matched_terms"])),
                "match_diagnostics": diagnostics,
                "score": score,
                "score_type": "weighted_match",
                "scores": base_scores.model_copy(
                    update={
                        "concept_match": float(len(per_term)),
                        "weighted_match": score,
                        "direct_concept_match": float(len(local_terms)),
                        "graph_only_concept_match": float(
                            len(graph_only_terms)
                        ),
                    }
                ),
            }
        )

        updated = base.model_copy(
            update={
                "section": updated_section,
                "direct": bool(record["has_local"]),
                "graph_distance": 0 if record["has_local"] else 2,
                "metadata": {
                    **base.metadata,
                    **dict(metadata),
                    "semantic_local_support": bool(record["has_local"]),
                    "semantic_graph_support": bool(record["has_graph"]),
                },
            }
        )
        local_rank = int(record["local_rank"])
        scored.append(
            (
                updated,
                score,
                len(local_terms),
                local_rank,
                len(graph_only_terms),
            )
        )

    scored.sort(
        key=lambda item: (
            -item[1],       # semantic evidence score
            -item[2],       # prefer local MENTIONS support on ties
            item[3],        # preserve original semantic-local tie order
            -item[4],       # then graph-only coverage
            item[0].section_uid,
        )
    )

    output: list[KGCandidate] = []
    ranked_items = scored if top_k is None else scored[:top_k]
    for rank, (candidate, *_rest) in enumerate(
        ranked_items,
        start=1,
    ):
        output.append(
            candidate.model_copy(
                update={
                    "source_rank": rank,
                    "seed_rank": rank if candidate.direct else None,
                }
            )
        )
    return output


class SemanticWeightedConnectionCandidateGenerator:
    """Semantic Concept seeding + MENTIONS + frozen DIRECT/BRIDGE expansion.

    Seed selection is performed exclusively by the configured Concept seeder.
    UMLS CUIs are read only from the selected local Concepts and are used only
    to activate frozen direct/bridge adjacency. Section ranking then uses the
    same semantic-similarity score as the validated local weighted baseline:
    maximum seed cosine per query term, summed across distinct terms.
    """

    name = "semantic_weighted_connection_artifact"

    def __init__(
        self,
        tools: KGSectionSearchProtocol,
        client: GraphReadClientProtocol,
        seeder: ConceptSeederProtocol,
        *,
        config_path: str | Path,
        mode_name: str,
        exclude_summary_sections: bool = True,
    ) -> None:
        self.tools = tools
        self.client = client
        self.seeder = seeder
        self.exclude_summary_sections = bool(exclude_summary_sections)
        self.local_generator = SeededMentionsCandidateGenerator(
            tools,
            seeder,
            exclude_summary_sections=exclude_summary_sections,
        )

        # Reuse the already validated artifact/config loader and adjacencies.
        self.connection = ConnectionArtifactCandidateGenerator(
            client,
            config_path=config_path,
            mode_name=mode_name,
            ranking_mode="weighted_match",
            exclude_summary_sections=exclude_summary_sections,
        )
        self.ranking_mode: KGRankingMode = "weighted_match"

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]:
        candidates, _ = self.generate_with_seeds(
            terms,
            top_k=top_k,
            require_all=require_all,
            document_ids=document_ids,
        )
        return candidates

    def _generate_channels_with_seeds(
        self,
        terms: Sequence[str] | str,
        *,
        local_top_k: int,
        graph_top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> tuple[list[KGCandidate], list[KGCandidate], list[ConceptSeed], dict[str, Any]]:
        """Generate local and graph channels independently before fusion."""
        normalized_terms = _normalize_terms(terms)
        validated_local_top_k = _validate_top_k(local_top_k)
        validated_graph_top_k = _validate_top_k(graph_top_k)

        local_candidates, seeds = self.local_generator.generate_with_seeds(
            normalized_terms,
            top_k=validated_local_top_k,
            require_all=require_all,
            document_ids=document_ids,
        )

        catalogue_rows = self.tools.list_concept_catalogue(
            document_ids=document_ids,
        )
        cui_by_concept_name = {
            str(row.get("concept_name") or "").strip(): (
                str(row.get("umls_cui") or "").strip() or None
            )
            for row in catalogue_rows
            if str(row.get("concept_name") or "").strip()
        }
        seed_rows = _seed_rows_from_concept_seeds(
            seeds,
            cui_by_concept_name=cui_by_concept_name,
        )
        expansions: list[dict[str, Any]] = []

        if self.connection.direct_artifact is not None:
            expansions.extend(
                _build_direct_expansions(
                    seed_rows,
                    self.connection.direct_adjacency,
                )
            )

        if self.connection.bridge_artifact is not None:
            assert self.connection.bridge_top_n is not None
            expansions.extend(
                _build_bridge_expansions(
                    seed_rows,
                    self.connection.bridge_source_adjacencies,
                    top_n=self.connection.bridge_top_n,
                )
            )

        graph_candidates: list[KGCandidate] = []
        if expansions:
            params = {
                "document_ids": _normalize_optional_values(document_ids),
                "exclude_summary_sections": self.exclude_summary_sections,
                "excluded_title_prefixes": _EXCLUDED_TITLE_PREFIXES,
                "connection_expansions": expansions,
                "connection_evidence_weight": 1.0,
            }
            expanded_rows = list(
                self.client.run_read(_CONNECTION_SECTION_EVIDENCE, params)
            )
            expanded_rows = _enrich_rows_with_expansion_metadata(
                expanded_rows, expansions
            )
            graph_results = _evidence_rows_to_results(
                expanded_rows,
                terms=[term.casefold() for term in normalized_terms],
                require_all=bool(require_all),
                top_k=validated_graph_top_k,
                ranking_mode="weighted_match",
            )
            graph_candidates = _wrap_concept_graph_results(graph_results)

        conn = self.connection
        metadata = {
            "generator": self.name,
            "seed_policy": "embedding_similarity_weighted",
            "connection_config": str(conn.config.path),
            "connection_config_sha256": conn.config.sha256,
            "connection_mode": conn.mode_name,
            "connection_mode_config": conn.mode_config,
            "semantic_seed_count": len(seeds),
            "semantic_seed_with_cui_count": len(seed_rows),
            "connection_expansion_count": len(expansions),
            "connection_max_depth": 1,
            "connection_second_hop": False,
            "connection_external_cuis_materialized": False,
            "gold_used_for_retrieval": False,
            "local_channel_k": validated_local_top_k,
            "graph_channel_k": validated_graph_top_k,
            "local_channel_candidate_count": len(local_candidates),
            "graph_channel_candidate_count": len(graph_candidates),
        }
        return list(local_candidates), list(graph_candidates), list(seeds), metadata

    def generate_with_seeds(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> tuple[list[KGCandidate], list[ConceptSeed]]:
        validated_top_k = _validate_top_k(top_k)
        local_candidates, graph_candidates, seeds, metadata = self._generate_channels_with_seeds(
            terms,
            local_top_k=validated_top_k,
            graph_top_k=validated_top_k,
            require_all=require_all,
            document_ids=document_ids,
        )
        output = _merge_semantic_connection_candidates(
            local_candidates,
            graph_candidates,
            top_k=validated_top_k,
            metadata=metadata,
        )
        return output, list(seeds)



def _rank_rrf_pool(
    local_candidates: Sequence[KGCandidate],
    graph_candidates: Sequence[KGCandidate],
    *,
    rrf_k: int,
    metadata: Mapping[str, Any],
) -> list[KGCandidate]:
    """Fuse independent local/graph ranks without tuned mixing weights."""
    if int(rrf_k) < 1:
        raise ValueError("rrf_k must be >= 1")
    merged = _merge_semantic_connection_candidates(
        local_candidates, graph_candidates, top_k=None, metadata=metadata
    )
    local_rank = {c.section_uid: int(c.source_rank) for c in local_candidates}
    graph_rank = {c.section_uid: int(c.source_rank) for c in graph_candidates}
    ranked = []
    for candidate in merged:
        lr = local_rank.get(candidate.section_uid)
        gr = graph_rank.get(candidate.section_uid)
        score = (1.0 / (int(rrf_k) + lr) if lr is not None else 0.0)
        score += (1.0 / (int(rrf_k) + gr) if gr is not None else 0.0)
        semantic_score = float(candidate.section.score)
        scores = candidate.section.scores or KGRetrievalScores()
        section = candidate.section.model_copy(update={
            "score": float(score),
            "score_type": "weighted_match",
            "scores": scores.model_copy(update={"weighted_match": float(score)}),
        })
        updated = candidate.model_copy(update={
            "section": section,
            "metadata": {
                **candidate.metadata, **dict(metadata),
                "pool_fusion_policy": "rrf",
                "rrf_k": int(rrf_k),
                "local_channel_rank": lr,
                "graph_channel_rank": gr,
                "pre_rrf_semantic_score": semantic_score,
            },
        })
        overlap = int(lr is not None and gr is not None)
        ranked.append((updated, score, overlap, lr if lr is not None else 10**9, semantic_score))
    ranked.sort(key=lambda x: (-x[1], -x[2], x[3], -x[4], x[0].section_uid))
    return [
        c.model_copy(update={"source_rank": rank, "seed_rank": rank if c.direct else None})
        for rank, (c, *_rest) in enumerate(ranked, start=1)
    ]


class PoolPreservingSemanticConnectionCandidateGenerator:
    """Generate local and graph pools independently, then union them."""

    name = "semantic_weighted_pool_preserving_connection"

    def __init__(
        self, tools: KGSectionSearchProtocol, client: GraphReadClientProtocol,
        seeder: ConceptSeederProtocol, *, config_path: str | Path, mode_name: str,
        graph_candidate_k: int = 50,
        fusion_policy: Literal["semantic_score", "rrf"] = "semantic_score",
        rrf_k: int = 60, exclude_summary_sections: bool = True,
    ) -> None:
        self.core = SemanticWeightedConnectionCandidateGenerator(
            tools, client, seeder, config_path=config_path, mode_name=mode_name,
            exclude_summary_sections=exclude_summary_sections,
        )
        self.graph_candidate_k = _validate_top_k(graph_candidate_k)
        if fusion_policy not in {"semantic_score", "rrf"}:
            raise ValueError("fusion_policy must be 'semantic_score' or 'rrf'")
        self.fusion_policy = fusion_policy
        self.rrf_k = int(rrf_k)
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        self.ranking_mode: KGRankingMode = "weighted_match"

    def generate(self, terms, *, top_k: int, require_all: bool = False, document_ids=None):
        candidates, _ = self.generate_with_seeds(
            terms, top_k=top_k, require_all=require_all, document_ids=document_ids
        )
        return candidates

    def generate_with_seeds(self, terms, *, top_k: int, require_all: bool = False, document_ids=None):
        local_k = _validate_top_k(top_k)
        local, graph, seeds, metadata = self.core._generate_channels_with_seeds(
            terms, local_top_k=local_k, graph_top_k=self.graph_candidate_k,
            require_all=require_all, document_ids=document_ids,
        )
        shared = {
            **metadata,
            "generator": self.name,
            "candidate_pool_policy": "separate_channels_then_union",
            "global_candidate_truncation_before_union": False,
            "pool_fusion_policy": self.fusion_policy,
        }
        if self.fusion_policy == "semantic_score":
            output = _merge_semantic_connection_candidates(
                local, graph, top_k=None, metadata=shared
            )
        else:
            output = _rank_rrf_pool(local, graph, rrf_k=self.rrf_k, metadata=shared)
        union_n = len(output)
        output = [c.model_copy(update={"metadata": {**c.metadata, "union_candidate_count": union_n}}) for c in output]
        return output, list(seeds)
