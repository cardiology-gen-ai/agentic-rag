"""Read-only loader for ontology-mediated local-local bridge artifacts.

The artifact represents a local relation A <-> B that is supported by one or
more external UMLS concepts X. X and the original A-X-B evidence remain in the
data-etl audit artifact; this loader exposes only controlled local-neighbour
expansion plus compact provenance for retrieval experiments.

No Neo4j writes, UMLS API calls, or second-hop ontology traversal occur here.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


_SUPPORTED_SCHEMA = "ontology_bridge_artifact_v1_1"
_PROFILE_TIERS = {
    "strong": {"STRONG"},
    "balanced": {"STRONG", "MEDIUM"},
    "broad": {"STRONG", "MEDIUM", "WEAK"},
}
_TIER_RANK = {"REJECT": 0, "WEAK": 1, "MEDIUM": 2, "STRONG": 3}


@dataclass(frozen=True)
class BridgeNeighbor:
    seed_cui: str
    neighbor_cui: str
    neighbor_preferred_name: str | None
    neighbor_types: tuple[str, ...]
    bridge_id: str
    tier: str
    score: float
    sources: tuple[str, ...]
    external_hub_count: int
    top_external_cuis: tuple[str, ...]
    top_external_names: tuple[str, ...]


@dataclass(frozen=True)
class _ReducedPath:
    source: str
    tier: str
    contribution: float
    external_cui: str
    external_name: str | None


@dataclass(frozen=True)
class _ReducedPair:
    bridge_id: str
    local_cui_a: str
    local_cui_b: str
    local_a: dict[str, Any]
    local_b: dict[str, Any]
    score_top_k: int
    paths: tuple[_ReducedPath, ...]


class FrozenOntologyBridgeArtifact:
    """Validated ontology-bridge v1.1 evidence store.

    Pair evidence is reduced in memory to the fields needed for source-specific
    ablations. Source-filtered scores are recomputed from the original
    per-external-hub contributions, so SNOMED/NCI/OMIM/LNC runs do not inherit a
    score that was boosted by another source.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Bridge manifest not found: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        schema = str(self.manifest.get("schema_version") or "").strip()
        if schema != _SUPPORTED_SCHEMA:
            raise ValueError(
                f"Unsupported bridge schema_version: {schema!r}; "
                f"expected {_SUPPORTED_SCHEMA!r}"
            )

        safety = self.manifest.get("safety") or {}
        for flag in (
            "umls_api_calls",
            "neo4j_writes",
            "second_hop_requests",
            "retrieval_metrics_used",
            "benchmark_tuned",
        ):
            if safety.get(flag) is not False:
                raise ValueError(
                    f"Bridge artifact safety requires {flag}=false; "
                    f"got {safety.get(flag)!r}"
                )

        files = self.manifest.get("files") or {}
        pair_name = str(files.get("pair_evidence") or "pair_evidence.jsonl")
        self.pair_evidence_path = self.artifact_dir / pair_name
        if not self.pair_evidence_path.is_file():
            raise FileNotFoundError(
                f"Bridge pair evidence not found: {self.pair_evidence_path}"
            )

        self.sha256 = hashlib.sha256(
            self.pair_evidence_path.read_bytes()
        ).hexdigest()
        self.sources_requested = tuple(
            str(v) for v in self.manifest.get("sources_requested", [])
        )
        self._pairs = self._load_reduced_pairs()

    def _load_reduced_pairs(self) -> tuple[_ReducedPair, ...]:
        pairs: list[_ReducedPair] = []
        with self.pair_evidence_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                paths: list[_ReducedPath] = []
                for item in row.get("external_path_evidence", []) or []:
                    source = str(item.get("source_vocabulary") or "").strip()
                    tier = str(
                        item.get("best_tier_for_external_path") or ""
                    ).strip()
                    external_cui = str(item.get("external_cui") or "").strip()
                    if not source or tier not in _TIER_RANK or not external_cui:
                        continue
                    paths.append(
                        _ReducedPath(
                            source=source,
                            tier=tier,
                            contribution=float(
                                item.get("bridge_score_contribution_v1_1") or 0.0
                            ),
                            external_cui=external_cui,
                            external_name=(
                                str(item.get("external_preferred_name")).strip()
                                if item.get("external_preferred_name") is not None
                                else None
                            ),
                        )
                    )
                a = str(row.get("local_cui_a") or "").strip()
                b = str(row.get("local_cui_b") or "").strip()
                bridge_id = str(row.get("bridge_id") or "").strip()
                if not a or not b or not bridge_id:
                    raise ValueError(
                        f"Invalid bridge pair at {self.pair_evidence_path}:"
                        f"{line_number}"
                    )
                pairs.append(
                    _ReducedPair(
                        bridge_id=bridge_id,
                        local_cui_a=a,
                        local_cui_b=b,
                        local_a=dict(row.get("local_a") or {}),
                        local_b=dict(row.get("local_b") or {}),
                        score_top_k=int(row.get("score_top_k") or 5),
                        paths=tuple(paths),
                    )
                )
        return tuple(pairs)

    @staticmethod
    def _profile_tiers(profile: str) -> set[str]:
        normalized = str(profile or "").strip().lower()
        if normalized not in _PROFILE_TIERS:
            raise ValueError(
                f"Unsupported bridge profile {profile!r}; "
                f"choose from {sorted(_PROFILE_TIERS)}"
            )
        return set(_PROFILE_TIERS[normalized])

    def build_adjacency(
        self,
        *,
        profile: str = "balanced",
        sources: Sequence[str] | None = None,
        ranking_policy: str = "tier_first",
    ) -> dict[str, tuple[BridgeNeighbor, ...]]:
        allowed_tiers = self._profile_tiers(profile)
        source_filter = {
            str(source).strip()
            for source in (sources or ())
            if str(source).strip()
        }
        unknown = source_filter - set(self.sources_requested)
        if unknown:
            raise ValueError(
                f"Unknown bridge sources: {sorted(unknown)}; "
                f"artifact sources={list(self.sources_requested)}"
            )
        normalized_ranking = str(ranking_policy or "").strip().lower()
        if normalized_ranking not in {"tier_first", "score_only"}:
            raise ValueError(
                "ranking_policy must be 'tier_first' or 'score_only'"
            )

        adjacency: dict[str, list[BridgeNeighbor]] = {}

        for pair in self._pairs:
            usable = [
                path
                for path in pair.paths
                if path.tier in allowed_tiers
                and (not source_filter or path.source in source_filter)
            ]
            if not usable:
                continue

            best_tier = max(
                (path.tier for path in usable),
                key=lambda tier: _TIER_RANK[tier],
            )
            top_paths = sorted(
                usable,
                key=lambda path: (
                    -_TIER_RANK[path.tier],
                    -path.contribution,
                    path.source,
                    path.external_cui,
                ),
            )
            contributions = sorted(
                (path.contribution for path in usable),
                reverse=True,
            )[: pair.score_top_k]
            raw = sum(contributions)
            score = 1.0 - math.exp(-raw) if raw > 0.0 else 0.0
            used_sources = tuple(sorted({path.source for path in usable}))
            top_external = top_paths[:5]

            for seed, target, target_meta in (
                (pair.local_cui_a, pair.local_cui_b, pair.local_b),
                (pair.local_cui_b, pair.local_cui_a, pair.local_a),
            ):
                adjacency.setdefault(seed, []).append(
                    BridgeNeighbor(
                        seed_cui=seed,
                        neighbor_cui=target,
                        neighbor_preferred_name=(
                            str(target_meta.get("preferred_name")).strip()
                            if target_meta.get("preferred_name") is not None
                            else None
                        ),
                        neighbor_types=tuple(
                            str(v)
                            for v in target_meta.get("canonical_types", []) or []
                        ),
                        bridge_id=pair.bridge_id,
                        tier=best_tier,
                        score=score,
                        sources=used_sources,
                        external_hub_count=len(usable),
                        top_external_cuis=tuple(
                            path.external_cui for path in top_external
                        ),
                        top_external_names=tuple(
                            path.external_name or path.external_cui
                            for path in top_external
                        ),
                    )
                )

        def sort_key(item: BridgeNeighbor) -> tuple[Any, ...]:
            if normalized_ranking == "tier_first":
                return (
                    -_TIER_RANK[item.tier],
                    -item.score,
                    item.neighbor_cui,
                )
            return (
                -item.score,
                -_TIER_RANK[item.tier],
                item.neighbor_cui,
            )

        return {
            seed: tuple(sorted(values, key=sort_key))
            for seed, values in sorted(adjacency.items())
        }

    @staticmethod
    def top_neighbors(
        adjacency: dict[str, tuple[BridgeNeighbor, ...]],
        seed_cui: str,
        *,
        top_n: int,
    ) -> tuple[BridgeNeighbor, ...]:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        return adjacency.get(str(seed_cui or "").strip(), ())[:top_n]
