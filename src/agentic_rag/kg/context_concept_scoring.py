"""Gold-free context scoring over a frozen term-derived Concept neighbourhood.

This module is intentionally diagnostic.  Q0 selects a fixed top-M Concept
neighbourhood from frozen router-term and Concept embeddings.  Q1 and Q2 then
score *only those same Concept identities* using whole-question and
contextualized-term embeddings respectively.  No Section candidates are
generated and no retrieval policy is modified here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from agentic_rag.kg.frozen_embeddings import load_concept_embedding_map


ContextScoreRole = Literal["q1_question", "q2_contextualized_term_v1"]
_CONTEXT_CACHE_SCHEMA = "context_score_embedding_cache_v1"


@dataclass(frozen=True)
class ContextConceptScore:
    """Q0/Q1/Q2 scores for one Concept in a frozen Q0 top-M neighbourhood."""

    concept_name: str
    q0_score: float
    q0_rank: int
    q1_score: float
    q1_rank: int
    q2_score: float
    q2_rank: int


@dataclass(frozen=True)
class Q0ReferenceValidation:
    """Validation summary against an already-frozen top-k semantic run."""

    compared_term_count: int
    compared_seed_count: int
    identity_mismatch_count: int
    max_abs_similarity_delta: float


def contextualized_term_text(term: str, question: str) -> str:
    """Return the exact v1 Q2 input text.

    Keep the formatter deliberately simple and version it through the cache
    role.  Any future wording change must use a new role/schema rather than
    silently overwriting frozen embeddings.
    """

    normalized_term = str(term).strip()
    normalized_question = str(question).strip()
    if not normalized_term:
        raise ValueError("term must be a non-empty string")
    if not normalized_question:
        raise ValueError("question must be a non-empty string")
    return f"{normalized_term}. Clinical context: {normalized_question}"


def load_exact_query_embedding_map(
    path: Path,
    *,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int | None = None,
) -> dict[str, np.ndarray]:
    """Load frozen baseline query embeddings keyed by exact text.

    The historical query cache stores the raw float32 API vector.  We apply the
    same row-wise L2 normalization used by :class:`EmbeddingConceptSeeder`
    before cosine scoring.  Exact text is preserved; no case folding is used.
    """

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)

    output: dict[str, np.ndarray] = {}
    for cache_file in sorted(resolved.glob("*.npz")):
        with np.load(cache_file, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            vector = np.asarray(data["embedding"], dtype=np.float32)
        if metadata.get("schema_version") != "query_term_embedding_cache_v1":
            continue
        if metadata.get("embedding_provider") != embedding_provider:
            continue
        if metadata.get("embedding_model") != embedding_model:
            continue
        if metadata.get("embedding_dimensions") != embedding_dimensions:
            continue
        text = str(metadata.get("term") or "").strip()
        if not text:
            continue
        normalized = _normalize_single_vector(vector)
        previous = output.get(text)
        if previous is not None and not np.array_equal(previous, normalized):
            raise RuntimeError(
                f"Conflicting frozen embeddings for exact query text {text!r}"
            )
        output[text] = normalized

    if not output:
        raise RuntimeError(f"No compatible query embeddings found in {resolved}")
    return output


class PersistentContextEmbeddingCache:
    """Persistent exact-text cache for Q1/Q2 embeddings.

    Vectors are L2-normalized exactly once before being persisted.  The load
    path returns the persisted float32 representation verbatim, avoiding the
    build/load drift previously observed when normalized vectors were
    normalized a second time.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimensions: int | None = None,
        read_only: bool = False,
        encoder: Any | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.embedding_provider = str(embedding_provider).strip()
        self.embedding_model = str(embedding_model).strip()
        self.embedding_dimensions = embedding_dimensions
        self.read_only = bool(read_only)
        self.encoder = encoder
        if not self.embedding_provider:
            raise ValueError("embedding_provider must be non-empty")
        if not self.embedding_model:
            raise ValueError("embedding_model must be non-empty")

        self.hits = 0
        self.misses = 0

    def resolve_many(
        self,
        requests: Sequence[tuple[ContextScoreRole, str]],
    ) -> dict[tuple[ContextScoreRole, str], np.ndarray]:
        """Resolve all unique requests, encoding missing entries only in build mode."""

        unique: list[tuple[ContextScoreRole, str]] = []
        seen: set[tuple[ContextScoreRole, str]] = set()
        for role, raw_text in requests:
            text = str(raw_text).strip()
            if not text:
                raise ValueError("Context embedding text must be non-empty")
            key = (role, text)
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)

        output: dict[tuple[ContextScoreRole, str], np.ndarray] = {}
        missing: list[tuple[ContextScoreRole, str]] = []
        for role, text in unique:
            cached = self._load(role, text)
            if cached is None:
                missing.append((role, text))
            else:
                output[(role, text)] = cached
                self.hits += 1

        if missing and self.read_only:
            preview = ", ".join(
                f"{role}:{text!r}" for role, text in missing[:3]
            )
            raise FileNotFoundError(
                "Frozen context embedding cache has "
                f"{len(missing)} missing entries; first: {preview}"
            )

        if missing:
            if self.encoder is None:
                raise RuntimeError(
                    "An embedding encoder is required to populate missing context cache entries"
                )
            texts = [text for _, text in missing]
            matrix = self._encode(texts)
            normalized = _normalize_matrix(matrix)
            for index, (role, text) in enumerate(missing):
                vector = np.asarray(normalized[index], dtype=np.float32)
                self._save(role, text, vector)
                output[(role, text)] = vector
                self.misses += 1

        return output

    def _metadata(self, role: ContextScoreRole, text: str) -> dict[str, Any]:
        return {
            "schema_version": _CONTEXT_CACHE_SCHEMA,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "role": role,
            "text": text,
        }

    def _cache_file(self, role: ContextScoreRole, text: str) -> Path:
        encoded = json.dumps(
            self._metadata(role, text),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.path / f"{hashlib.sha256(encoded).hexdigest()}.npz"

    def _load(self, role: ContextScoreRole, text: str) -> np.ndarray | None:
        cache_file = self._cache_file(role, text)
        if not cache_file.is_file():
            return None
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                vector = np.asarray(data["embedding"], dtype=np.float32)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            if self.read_only:
                raise RuntimeError(
                    f"Invalid frozen context embedding cache entry: {cache_file}"
                ) from exc
            return None
        if metadata != self._metadata(role, text):
            if self.read_only:
                raise RuntimeError(
                    f"Context embedding cache metadata mismatch: {cache_file}"
                )
            return None
        if vector.ndim != 1:
            raise RuntimeError(
                f"Invalid context embedding vector shape in {cache_file}: {vector.shape}"
            )
        # Persisted vectors are already normalized.  Return verbatim.
        return vector

    def _save(self, role: ContextScoreRole, text: str, vector: np.ndarray) -> None:
        cache_file = self._cache_file(role, text)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            embedding=np.asarray(vector, dtype=np.float32),
            metadata=np.array(json.dumps(self._metadata(role, text), sort_keys=True)),
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        try:
            vectors = self.encoder.encode(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except TypeError:
            vectors = self.encoder.encode(list(texts))
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise RuntimeError("Embedding encoder returned an invalid matrix shape")
        return matrix


def score_frozen_q0_neighbourhood(
    *,
    term: str,
    question: str,
    q0_vector: np.ndarray,
    concept_vectors: Mapping[str, np.ndarray],
    q1_vector: np.ndarray,
    q2_vector: np.ndarray,
    top_m: int = 10,
) -> list[ContextConceptScore]:
    """Score Q1/Q2 inside the unchanged Q0 top-M Concept neighbourhood."""

    if top_m < 1:
        raise ValueError("top_m must be positive")
    if not str(term).strip() or not str(question).strip():
        raise ValueError("term and question must be non-empty")
    if not concept_vectors:
        raise ValueError("concept_vectors must be non-empty")

    q0 = _require_vector(q0_vector, "q0_vector")
    q1 = _require_vector(q1_vector, "q1_vector")
    q2 = _require_vector(q2_vector, "q2_vector")

    universe: list[tuple[str, float]] = []
    expected_dim: int | None = None
    for name, raw_vector in concept_vectors.items():
        concept_name = str(name).strip()
        if not concept_name:
            raise ValueError("Concept names must be non-empty")
        vector = _require_vector(raw_vector, f"Concept {concept_name!r}")
        if expected_dim is None:
            expected_dim = vector.shape[0]
        if vector.shape[0] != expected_dim:
            raise ValueError("Concept embedding dimensions are inconsistent")
        universe.append((concept_name, float(np.dot(vector, q0))))

    if expected_dim not in {q0.shape[0], q1.shape[0], q2.shape[0]}:
        raise ValueError("Query and Concept embedding dimensions differ")
    if not (q0.shape[0] == q1.shape[0] == q2.shape[0] == expected_dim):
        raise ValueError("Query and Concept embedding dimensions differ")

    universe.sort(key=lambda item: (-item[1], item[0].casefold(), item[0]))
    q0_top = universe[: min(top_m, len(universe))]

    rows: list[dict[str, Any]] = []
    for q0_rank, (name, q0_score) in enumerate(q0_top, start=1):
        vector = _require_vector(concept_vectors[name], f"Concept {name!r}")
        rows.append(
            {
                "concept_name": name,
                "q0_score": q0_score,
                "q0_rank": q0_rank,
                "q1_score": float(np.dot(vector, q1)),
                "q2_score": float(np.dot(vector, q2)),
            }
        )

    q1_rank = _ranks_within_rows(rows, "q1_score")
    q2_rank = _ranks_within_rows(rows, "q2_score")
    return [
        ContextConceptScore(
            concept_name=str(row["concept_name"]),
            q0_score=float(row["q0_score"]),
            q0_rank=int(row["q0_rank"]),
            q1_score=float(row["q1_score"]),
            q1_rank=q1_rank[str(row["concept_name"])],
            q2_score=float(row["q2_score"]),
            q2_rank=q2_rank[str(row["concept_name"])],
        )
        for row in rows
    ]


def load_frozen_concepts(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Public wrapper used by the diagnostic script and tests."""

    return load_concept_embedding_map(path)


def validate_q0_reference_prefix(
    *,
    scored_by_term: Mapping[tuple[str, str], Sequence[ContextConceptScore]],
    reference_rows: Sequence[Mapping[str, Any]],
    reference_mode: str,
    similarity_tolerance: float = 1e-6,
) -> Q0ReferenceValidation:
    """Validate Q0 top-M prefixes against frozen semantic ConceptSeed traces.

    ``scored_by_term`` is keyed by ``(question_id, router_term)``.  Reference
    identities must match exactly.  Similarity is checked only as a numerical
    sanity gate and is not used to alter scoring.
    """

    by_question = {
        str(row.get("question_id") or ""): row
        for row in reference_rows
        if str(row.get("mode") or "") == reference_mode
    }
    term_count = seed_count = identity_mismatches = 0
    max_delta = 0.0

    for (question_id, term), scored in sorted(scored_by_term.items()):
        row = by_question.get(question_id)
        if row is None:
            raise RuntimeError(
                f"Q0 reference run has no row for question {question_id!r}"
            )
        reference = [
            seed
            for seed in (row.get("concept_seeds") or [])
            if str(seed.get("match_type") or "") == "embedding"
            and str(seed.get("query_term") or "").strip() == term
        ]
        if not reference:
            raise RuntimeError(
                f"Q0 reference run has no embedding seeds for {question_id!r}/{term!r}"
            )
        reference.sort(
            key=lambda seed: (
                int(seed.get("seed_rank") or 10**9),
                str(seed.get("concept_name") or "").casefold(),
                str(seed.get("concept_name") or ""),
            )
        )
        term_count += 1
        for index, seed in enumerate(reference):
            if index >= len(scored):
                raise RuntimeError(
                    f"Q0 top-M shorter than frozen reference for {question_id!r}/{term!r}"
                )
            expected_name = str(seed.get("concept_name") or "")
            actual = scored[index]
            if actual.concept_name != expected_name:
                identity_mismatches += 1
            expected_similarity = seed.get("similarity")
            if expected_similarity is not None:
                delta = abs(actual.q0_score - float(expected_similarity))
                max_delta = max(max_delta, delta)
                if delta > similarity_tolerance:
                    raise RuntimeError(
                        "Q0 similarity differs from frozen reference by more than "
                        f"{similarity_tolerance}: {question_id}/{term}/"
                        f"{expected_name}: {actual.q0_score} vs {expected_similarity}"
                    )
            seed_count += 1

    if identity_mismatches:
        raise RuntimeError(
            f"Q0 Concept identity mismatch count vs frozen reference: {identity_mismatches}"
        )
    return Q0ReferenceValidation(
        compared_term_count=term_count,
        compared_seed_count=seed_count,
        identity_mismatch_count=identity_mismatches,
        max_abs_similarity_delta=max_delta,
    )



def annotate_gold_support_rows(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    gold_concepts_by_question: Mapping[str, Mapping[str, set[str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Annotate already-finalized score rows with post-hoc gold support.

    The function accepts only precomputed score/rank rows; it cannot influence
    Q0/Q1/Q2 construction. ``gold_concepts_by_question`` maps question IDs to
    document-aware gold Section identities and their exact Concept names.
    """

    annotated: list[dict[str, Any]] = []
    for row in score_rows:
        out = dict(row)
        qid = str(out["question_id"])
        name = str(out["concept_name"])
        if qid not in gold_concepts_by_question:
            raise KeyError(f"Missing gold Concept mapping for question {qid!r}")
        supported = sorted(
            identity
            for identity, concepts in gold_concepts_by_question[qid].items()
            if name in concepts
        )
        out["supports_gold_section"] = bool(supported)
        out["supported_gold_sections"] = json.dumps(supported, ensure_ascii=False)
        annotated.append(out)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in annotated:
        key = (str(row["question_id"]), str(row["router_term"]))
        groups.setdefault(key, []).append(row)

    term_rows: list[dict[str, Any]] = []
    for (qid, term), rows in sorted(groups.items()):
        supporters = [row for row in rows if row["supports_gold_section"]]
        out: dict[str, Any] = {
            "question_id": qid,
            "router_term": term,
            "supporting_concept_count_in_q0_top_m": len(supporters),
        }
        for signal in ("q0", "q1", "q2"):
            rank_key = f"{signal}_rank"
            score_key = f"{signal}_score"
            if supporters:
                best_rank = min(int(row[rank_key]) for row in supporters)
                out[f"best_support_rank_{signal}"] = best_rank
                out[f"support_rr_{signal}"] = 1.0 / best_rank
                best_support_score = max(float(row[score_key]) for row in supporters)
                non_supporters = [row for row in rows if not row["supports_gold_section"]]
                out[f"support_margin_{signal}"] = (
                    best_support_score
                    - max(float(row[score_key]) for row in non_supporters)
                    if non_supporters
                    else None
                )
            else:
                out[f"best_support_rank_{signal}"] = None
                out[f"support_rr_{signal}"] = None
                out[f"support_margin_{signal}"] = None
        term_rows.append(out)

    summary = {
        "gold_used_for_score_or_rank_construction": False,
        "gold_used_for_posthoc_annotation": True,
        "router_terms_with_direct_gold_support_in_q0_top_m": sum(
            1 for row in term_rows if row["supporting_concept_count_in_q0_top_m"] > 0
        ),
        "q0": _aggregate_gold_signal(term_rows, "q0"),
        "q1": _aggregate_gold_signal(term_rows, "q1"),
        "q2": _aggregate_gold_signal(term_rows, "q2"),
    }
    return annotated, term_rows, summary


def _aggregate_gold_signal(
    rows: Sequence[Mapping[str, Any]], signal: str
) -> dict[str, Any]:
    rank_key = f"best_support_rank_{signal}"
    rr_key = f"support_rr_{signal}"
    margin_key = f"support_margin_{signal}"
    eligible = [row for row in rows if row.get(rank_key) not in (None, "")]
    margins = [float(row[margin_key]) for row in eligible if row.get(margin_key) is not None]
    return {
        "eligible_router_term_count": len(eligible),
        "support_hit_at_1": (
            sum(1.0 if int(row[rank_key]) <= 1 else 0.0 for row in eligible) / len(eligible)
            if eligible else None
        ),
        "support_hit_at_3": (
            sum(1.0 if int(row[rank_key]) <= 3 else 0.0 for row in eligible) / len(eligible)
            if eligible else None
        ),
        "support_mrr": (
            sum(float(row[rr_key]) for row in eligible) / len(eligible)
            if eligible else None
        ),
        "mean_best_support_rank": (
            sum(float(row[rank_key]) for row in eligible) / len(eligible)
            if eligible else None
        ),
        "mean_support_margin": (sum(margins) / len(margins) if margins else None),
    }

def _ranks_within_rows(
    rows: Sequence[Mapping[str, Any]],
    score_key: str,
) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[score_key]),
            str(row["concept_name"]).casefold(),
            str(row["concept_name"]),
        ),
    )
    return {
        str(row["concept_name"]): rank
        for rank, row in enumerate(ordered, start=1)
    }


def _require_vector(vector: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    return array


def _normalize_single_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != 1:
        raise ValueError("Expected one embedding vector")
    return _normalize_matrix(array)[0]


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = np.asarray(matrix, dtype=np.float32)
    if normalized.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional")
    if normalized.shape[0] == 0:
        return normalized
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (normalized / norms).astype(np.float32)
