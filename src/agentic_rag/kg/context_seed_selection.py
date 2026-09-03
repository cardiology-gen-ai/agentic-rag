"""Frozen context-score Concept selection for controlled retrieval ablations.

This module deliberately separates *selection* from downstream candidate
scoring.  Q0/Q2 scores are loaded from the frozen diagnostic artifact.  The
selected Concept identities may change, but every emitted seed retains its
original Q0 term-to-Concept cosine and Q0 rank.  Therefore the treatment tests
whether context identifies useful Concept representations without also
changing the semantic weighting policy used by candidate preselection.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from agentic_rag.kg.concept_seeders import (
    ConceptCatalogueRecord,
    ConceptSeed,
    EmbeddingConceptSeeder,
)

ContextSeedSelectionPolicy = Literal[
    "q0_top3",
    "q2_top3",
    "q0_q2_top3_union",
    "q2_topk",
]

_REQUIRED_COLUMNS = {
    "question_id",
    "router_term",
    "concept_identity_exact",
    "concept_name",
    "q0_score",
    "q0_rank",
    "q2_score",
    "q2_rank",
}


class FrozenContextScoreConceptSeeder:
    """Select frozen Concept identities from a Q0/Q2 diagnostic artifact.

    The wrapped :class:`EmbeddingConceptSeeder` is used only for the frozen
    exact Concept catalogue and provenance.  No query embeddings are generated
    by this wrapper.
    """

    name = "frozen_context_score_concept_seeder"

    def __init__(
        self,
        base_seeder: EmbeddingConceptSeeder,
        *,
        score_csv: str | Path,
        policy: ContextSeedSelectionPolicy,
        manifest_path: str | Path | None = None,
        top_k: int = 3,
        neighbourhood_m: int | None = None,
    ) -> None:
        if policy not in {"q0_top3", "q2_top3", "q0_q2_top3_union", "q2_topk"}:
            raise ValueError(f"Unsupported context seed selection policy: {policy!r}")
        if policy != "q2_topk" and top_k != 3:
            raise ValueError("legacy context seed policies require top_k=3")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if neighbourhood_m is not None and neighbourhood_m < 3:
            raise ValueError("neighbourhood_m must be at least 3")
        if neighbourhood_m is not None and top_k > neighbourhood_m:
            raise ValueError("top_k cannot exceed neighbourhood_m")
        if not isinstance(base_seeder, EmbeddingConceptSeeder):
            raise TypeError("base_seeder must be an EmbeddingConceptSeeder")

        self.base_seeder = base_seeder
        self.policy: ContextSeedSelectionPolicy = policy
        self.top_k = int(top_k)
        self.neighbourhood_m = int(neighbourhood_m) if neighbourhood_m is not None else None
        if policy == "q0_q2_top3_union":
            self.concepts_per_term = 6
        elif policy == "q2_topk":
            self.concepts_per_term = self.top_k
        else:
            self.concepts_per_term = 3
        self.score_csv = Path(score_csv).expanduser().resolve()
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else self.score_csv.parent / "manifest.json"
        )
        self._current_question_id: str | None = None
        self._rows = self._load_rows(self.score_csv)
        self._manifest = self._load_manifest(self.manifest_path)
        self._catalogue_by_name: dict[str, ConceptCatalogueRecord] = {}

    def prepare(
        self,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> None:
        self.base_seeder.prepare(document_ids=document_ids)
        self._validate_manifest_against_base()
        self._catalogue_by_name = {
            record.concept_name: record
            for record in self.base_seeder.catalogue_records
        }
        missing = sorted(
            {
                str(row["concept_name"])
                for rows in self._rows.values()
                for row in rows
                if str(row["concept_name"]) not in self._catalogue_by_name
            }
        )
        if missing:
            raise RuntimeError(
                "Frozen context score artifact contains Concept identities not "
                f"present in the prepared exact catalogue; first: {missing[:5]}"
            )

    def set_question_id(self, question_id: str) -> None:
        normalized = str(question_id).strip()
        if not normalized:
            raise ValueError("question_id must be non-empty")
        self._current_question_id = normalized

    def seed_concepts(
        self,
        terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> dict[str, list[ConceptSeed]]:
        if self._current_question_id is None:
            raise RuntimeError(
                "FrozenContextScoreConceptSeeder requires set_question_id() before retrieval"
            )
        raw_terms = [terms] if isinstance(terms, str) else list(terms)
        normalized_terms = [str(raw_term).strip() for raw_term in raw_terms if str(raw_term).strip()]

        # Validate artifact coverage before touching the baseline seeder.  In
        # particular, a missing frozen term must remain a deterministic hard
        # failure rather than falling through to an encoder/cache lookup.
        for term in normalized_terms:
            if (self._current_question_id, term) not in self._rows:
                raise KeyError(
                    "No frozen context score rows for "
                    f"question={self._current_question_id!r}, term={term!r}"
                )

        # Recompute the frozen historical Q0 top-3 through the original
        # EmbeddingConceptSeeder.  This is intentionally *not* taken from the
        # diagnostic CSV: Q0 scores in that artifact are numerically
        # reconstructed Concept-by-Concept and can differ from the original
        # matrix-vector implementation by sub-micro float32 amounts.  Those
        # tiny deltas are sufficient to perturb fixed-budget ordering near
        # ties.  The base seeder uses the same frozen Concept/query caches and
        # the same matrix @ query_vector code path as the canonical baseline.
        baseline_groups = self.base_seeder.seed_concepts(
            normalized_terms,
            document_ids=document_ids,
        )

        output: dict[str, list[ConceptSeed]] = {}
        for term in normalized_terms:
            key = (self._current_question_id, term)
            rows = self._rows.get(key)
            if rows is None:
                raise KeyError(
                    "No frozen context score rows for "
                    f"question={self._current_question_id!r}, term={term!r}"
                )

            baseline = list(baseline_groups.get(term) or [])
            if len(baseline) != 3:
                raise RuntimeError(
                    f"Frozen baseline Q0 did not return exactly 3 seeds for {term!r}: "
                    f"received {len(baseline)}"
                )
            baseline_by_name = {seed.concept_name: seed for seed in baseline}
            by_q0 = sorted(
                rows,
                key=lambda row: (
                    int(row["q0_rank"]),
                    str(row["concept_name"]).casefold(),
                    str(row["concept_name"]),
                ),
            )
            artifact_q0_top3 = [str(row["concept_name"]) for row in by_q0[:3]]
            baseline_q0_top3 = [seed.concept_name for seed in baseline]
            if artifact_q0_top3 != baseline_q0_top3:
                raise RuntimeError(
                    "Frozen context artifact Q0 identities differ from the live frozen "
                    f"baseline for {self._current_question_id!r}/{term!r}: "
                    f"artifact={artifact_q0_top3!r}, baseline={baseline_q0_top3!r}"
                )

            selected = self._select(rows)
            seeds: list[ConceptSeed] = []
            for row in selected:
                name = str(row["concept_name"])

                # Whenever the selected Concept belongs to the canonical Q0
                # top-3, reuse the exact historical seed object.  This makes
                # q0_top3 a true replay control and preserves baseline Q0
                # confidence exactly inside the union treatment.
                frozen_seed = baseline_by_name.get(name)
                if frozen_seed is not None:
                    seeds.append(frozen_seed)
                    continue

                # Q2-only rescues can come from Q0 ranks 4..10.  These have no
                # historical top-3 seed object, so retain their frozen Q0 rank
                # and diagnostic Q0 cosine.  They are novel treatment seeds;
                # exact baseline replay is therefore not applicable to them.
                record = self._catalogue_by_name.get(name)
                if record is None:
                    raise RuntimeError(
                        f"Concept {name!r} was not prepared in the exact catalogue"
                    )
                seeds.append(
                    ConceptSeed(
                        query_term=term,
                        concept_name=name,
                        canonical_type=record.canonical_type,
                        umls_cui=None,
                        method="embedding",
                        match_type="embedding",
                        seed_rank=int(row["q0_rank"]),
                        similarity=float(row["q0_score"]),
                        matched_value=f"context_selection:{self.policy}",
                    )
                )
            output[term] = seeds
        return output

    def _select(self, rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
        by_q0 = sorted(
            rows,
            key=lambda row: (
                int(row["q0_rank"]),
                str(row["concept_name"]).casefold(),
                str(row["concept_name"]),
            ),
        )
        if self.policy == "q0_top3":
            selected_names = {str(row["concept_name"]) for row in by_q0[:3]}
        elif self.policy == "q2_top3":
            by_q2 = sorted(
                rows,
                key=lambda row: (
                    int(row["q2_rank"]),
                    str(row["concept_name"]).casefold(),
                    str(row["concept_name"]),
                ),
            )
            selected_names = {str(row["concept_name"]) for row in by_q2[:3]}
        elif self.policy == "q2_topk":
            m = self.neighbourhood_m if self.neighbourhood_m is not None else len(by_q0)
            if len(by_q0) < m:
                raise RuntimeError(
                    f"Context score group has only {len(by_q0)} Concepts but neighbourhood_m={m}"
                )
            eligible = by_q0[:m]
            by_q2 = sorted(
                eligible,
                key=lambda row: (
                    -float(row["q2_score"]),
                    str(row["concept_name"]).casefold(),
                    str(row["concept_name"]),
                ),
            )
            selected_names = {str(row["concept_name"]) for row in by_q2[: self.top_k]}
        else:
            by_q2 = sorted(
                rows,
                key=lambda row: (
                    int(row["q2_rank"]),
                    str(row["concept_name"]).casefold(),
                    str(row["concept_name"]),
                ),
            )
            selected_names = {
                *(str(row["concept_name"]) for row in by_q0[:3]),
                *(str(row["concept_name"]) for row in by_q2[:3]),
            }
        # Always emit in original Q0 order.  This makes Q2 a membership signal
        # only; Q0 remains the confidence/rank signal used downstream.
        return [row for row in by_q0 if str(row["concept_name"]) in selected_names]

    @staticmethod
    def _load_rows(path: Path) -> dict[tuple[str, str], list[dict[str, object]]]:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = sorted(_REQUIRED_COLUMNS - fields)
            if missing:
                raise ValueError(f"Context score CSV missing required columns: {missing}")
            raw_rows = list(reader)

        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for raw in raw_rows:
            qid = str(raw["question_id"]).strip()
            term = str(raw["router_term"]).strip()
            name = str(raw["concept_name"]).strip()
            identity = str(raw["concept_identity_exact"]).strip()
            if not qid or not term or not name or not identity:
                raise ValueError("Context score rows require non-empty exact identities")
            if identity != name:
                raise ValueError(
                    f"v1 requires concept_identity_exact == concept_name: {identity!r} != {name!r}"
                )
            row: dict[str, object] = dict(raw)
            row["q0_rank"] = int(raw["q0_rank"])
            row["q2_rank"] = int(raw["q2_rank"])
            row["q0_score"] = float(raw["q0_score"])
            row["q2_score"] = float(raw["q2_score"])
            grouped.setdefault((qid, term), []).append(row)

        if not grouped:
            raise ValueError("Context score CSV contains no rows")
        for key, rows in grouped.items():
            names = [str(row["concept_name"]) for row in rows]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate Concept identity in context score group {key!r}")
            q0_ranks = sorted(int(row["q0_rank"]) for row in rows)
            q2_ranks = sorted(int(row["q2_rank"]) for row in rows)
            expected = list(range(1, len(rows) + 1))
            if q0_ranks != expected or q2_ranks != expected:
                raise ValueError(
                    f"Context score ranks must be complete permutations in group {key!r}"
                )
            if len(rows) < 3:
                raise ValueError(f"Context score group {key!r} has fewer than 3 Concepts")
        return grouped

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, object]:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "kg_context_score_diagnostic_result_v1":
            raise ValueError(
                f"Unsupported context score manifest schema: {payload.get('schema_version')!r}"
            )
        validation = payload.get("q0_reference_validation") or {}
        if int(validation.get("identity_mismatch_count", -1)) != 0:
            raise RuntimeError("Context score artifact did not pass its Q0 identity gate")
        return payload

    def _validate_manifest_against_base(self) -> None:
        inputs = self._manifest.get("inputs") or {}
        expected_hash = str(inputs.get("concept_catalogue_hash") or "").strip()
        actual_hash = str(self.base_seeder.catalogue_hash or "").strip()
        if not expected_hash:
            raise RuntimeError("Context score manifest has no concept_catalogue_hash")
        if expected_hash != actual_hash:
            raise RuntimeError(
                "Context score Concept catalogue hash differs from prepared baseline: "
                f"{expected_hash} != {actual_hash}"
            )

    # Provenance attributes expected by evaluate_kg_retrieval.py.
    @property
    def catalogue_size(self) -> int:
        return self.base_seeder.catalogue_size

    @property
    def catalogue_hash(self) -> str | None:
        return self.base_seeder.catalogue_hash

    @property
    def catalogue_build_load_seconds(self) -> float:
        return self.base_seeder.catalogue_build_load_seconds

    @property
    def model_load_seconds(self) -> float:
        return self.base_seeder.model_load_seconds

    @property
    def loaded_from_cache(self) -> bool:
        return self.base_seeder.loaded_from_cache

    @property
    def resolved_cache_file(self) -> str | None:
        return self.base_seeder.resolved_cache_file

    @property
    def query_embedding_cache_hits(self) -> int:
        return self.base_seeder.query_embedding_cache_hits

    @property
    def query_embedding_cache_misses(self) -> int:
        return self.base_seeder.query_embedding_cache_misses
