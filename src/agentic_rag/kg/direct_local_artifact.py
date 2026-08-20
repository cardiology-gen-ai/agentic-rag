from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_SUPPORTED_SCHEMA = "direct_local_artifact_v1"
_SUPPORTED_PROFILES = ("strong", "balanced", "broad")


@dataclass(frozen=True)
class DirectNeighbor:
    seed_cui: str
    neighbor_cui: str
    neighbor_preferred_name: str | None
    neighbor_types: tuple[str, ...]
    direct_id: str
    tier: str
    sources: tuple[str, ...]
    relation_families: tuple[str, ...]
    projection_ambiguity: bool


class FrozenDirectLocalArtifact:
    """Read-only loader for direct_local_artifact_v1.

    The artifact already encodes the benchmark-independent semantic policy.
    This loader never calls UMLS or Neo4j and never modifies the artifact.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Direct artifact manifest not found: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        schema = str(self.manifest.get("schema_version") or "").strip()
        if schema != _SUPPORTED_SCHEMA:
            raise ValueError(
                f"Unsupported direct artifact schema_version: {schema!r}; "
                f"expected {_SUPPORTED_SCHEMA!r}"
            )

        if self.manifest.get("inverse_relations_grouped") is not True:
            raise ValueError("Direct artifact must have inverse_relations_grouped=true")
        if self.manifest.get("retrieval_adjacency_is_bidirectional_per_semantic_pair") is not True:
            raise ValueError(
                "Direct artifact must have "
                "retrieval_adjacency_is_bidirectional_per_semantic_pair=true"
            )

        safety = self.manifest.get("safety") or {}
        for flag in (
            "umls_api_calls",
            "neo4j_writes",
            "benchmark_data_used",
            "retrieval_metrics_used",
            "benchmark_tuned",
            "second_hop_traversal",
        ):
            if safety.get(flag) is not False:
                raise ValueError(
                    f"Direct artifact safety requires {flag}=false; "
                    f"got {safety.get(flag)!r}"
                )

        self.pair_evidence_path = self.artifact_dir / "pair_evidence.jsonl"
        if not self.pair_evidence_path.is_file():
            raise FileNotFoundError(
                f"Direct pair evidence not found: {self.pair_evidence_path}"
            )
        self.sha256 = hashlib.sha256(self.pair_evidence_path.read_bytes()).hexdigest()

        self._pair_meta = self._load_pair_meta()
        self.available_sources = tuple(
            source
            for source, row in (self.manifest.get("source_summary") or {}).items()
            if int((row or {}).get("pair_count") or 0) > 0
        )

    def _load_pair_meta(self) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        with self.pair_evidence_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                direct_id = str(row.get("direct_id") or "").strip()
                if not direct_id:
                    raise ValueError(
                        f"Missing direct_id at {self.pair_evidence_path}:{line_number}"
                    )
                if direct_id in output:
                    raise ValueError(f"Duplicate direct_id in artifact: {direct_id}")
                output[direct_id] = row
        return output

    @staticmethod
    def validate_profile(profile: str) -> str:
        normalized = str(profile or "").strip().lower()
        if normalized not in _SUPPORTED_PROFILES:
            raise ValueError(
                f"Unsupported direct profile {profile!r}; "
                f"choose from {_SUPPORTED_PROFILES}"
            )
        return normalized

    def _adjacency_path(self, *, profile: str, source: str | None) -> Path:
        profile = self.validate_profile(profile)
        if source is None:
            return self.artifact_dir / "adjacency" / f"{profile}.json"
        return (
            self.artifact_dir
            / "adjacency"
            / "by_source"
            / f"{source}__{profile}.json"
        )

    def _load_raw_adjacency(
        self, *, profile: str, source: str | None
    ) -> dict[str, list[dict[str, Any]]]:
        path = self._adjacency_path(profile=profile, source=source)
        if not path.is_file():
            raise FileNotFoundError(f"Direct adjacency not found: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Direct adjacency must be a JSON object: {path}")
        return value

    def build_adjacency(
        self,
        *,
        profile: str,
        sources: Sequence[str] | None = None,
    ) -> dict[str, tuple[DirectNeighbor, ...]]:
        """Build source-filtered bidirectional adjacency.

        If ``sources`` is omitted, the frozen combined adjacency is loaded.
        With sources, source-specific frozen adjacencies are unioned and
        deduplicated by (seed, neighbor, direct_id).  No ranking/top-N is
        applied: reachability uses every connection admitted by the profile.
        """
        profile = self.validate_profile(profile)
        source_filter = tuple(
            dict.fromkeys(
                str(source).strip()
                for source in (sources or ())
                if str(source).strip()
            )
        )
        unknown = set(source_filter) - set(self.available_sources)
        if unknown:
            raise ValueError(
                f"Unknown/non-populated direct sources: {sorted(unknown)}; "
                f"available={list(self.available_sources)}"
            )

        raw_sets = (
            [self._load_raw_adjacency(profile=profile, source=s) for s in source_filter]
            if source_filter
            else [self._load_raw_adjacency(profile=profile, source=None)]
        )

        merged: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
        for raw in raw_sets:
            for seed_cui, entries in raw.items():
                for entry in entries or []:
                    direct_id = str(entry.get("direct_id") or "").strip()
                    neighbor_cui = str(entry.get("neighbor_cui") or "").strip()
                    if not direct_id or not neighbor_cui:
                        continue
                    key = (neighbor_cui, direct_id)
                    bucket = merged.setdefault(str(seed_cui), {})
                    if key not in bucket:
                        bucket[key] = dict(entry)
                    else:
                        current = bucket[key]
                        current["sources"] = sorted(
                            set(current.get("sources") or [])
                            | set(entry.get("sources") or [])
                        )
                        current["relation_families"] = sorted(
                            set(current.get("relation_families") or [])
                            | set(entry.get("relation_families") or [])
                        )
                        current["projection_ambiguity"] = bool(
                            current.get("projection_ambiguity")
                            or entry.get("projection_ambiguity")
                        )

        output: dict[str, tuple[DirectNeighbor, ...]] = {}
        for seed_cui, entries in sorted(merged.items()):
            neighbors: list[DirectNeighbor] = []
            for (neighbor_cui, direct_id), entry in entries.items():
                pair = self._pair_meta.get(direct_id)
                if pair is None:
                    raise ValueError(
                        f"Adjacency references unknown direct_id: {direct_id}"
                    )
                if neighbor_cui == str(pair.get("local_cui_a") or ""):
                    meta = pair.get("local_a") or {}
                elif neighbor_cui == str(pair.get("local_cui_b") or ""):
                    meta = pair.get("local_b") or {}
                else:
                    raise ValueError(
                        f"Neighbor {neighbor_cui} is not an endpoint of {direct_id}"
                    )

                names = list(meta.get("names") or [])
                neighbors.append(
                    DirectNeighbor(
                        seed_cui=seed_cui,
                        neighbor_cui=neighbor_cui,
                        neighbor_preferred_name=(str(names[0]) if names else None),
                        neighbor_types=tuple(
                            str(v) for v in (meta.get("canonical_types") or [])
                        ),
                        direct_id=direct_id,
                        tier=str(entry.get("tier") or "").strip(),
                        sources=tuple(sorted(str(v) for v in (entry.get("sources") or []))),
                        relation_families=tuple(
                            sorted(str(v) for v in (entry.get("relation_families") or []))
                        ),
                        projection_ambiguity=bool(entry.get("projection_ambiguity")),
                    )
                )
            # Deterministic only; NOT a relevance ranking.
            output[seed_cui] = tuple(
                sorted(
                    neighbors,
                    key=lambda x: (
                        x.neighbor_cui,
                        x.direct_id,
                    ),
                )
            )
        return output
