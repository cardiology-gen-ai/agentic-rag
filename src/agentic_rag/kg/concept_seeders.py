"""Concept seeders for controlled MENTIONS ablations."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


ConceptSeedingMethod = Literal["lexical", "embedding"]


class ConceptSeed(BaseModel):
    """One local Concept selected from an original MENTIONS plan term."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_term: str
    concept_name: str
    canonical_type: str | None = None
    umls_cui: str | None = None
    method: ConceptSeedingMethod
    match_type: str
    seed_rank: int = Field(ge=1)
    similarity: float | None = None
    matched_value: str | None = None

    @field_validator("query_term", "concept_name", "match_type")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Value must be a non-empty string")
        return normalized

    @field_validator("canonical_type", "umls_cui", "matched_value")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class ConceptCatalogueRecord(BaseModel):
    """Normalized local Concept properties used by Concept seeders."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_name: str
    name: str | None = None
    normalized_name: str | None = None
    umls_canonical_name: str | None = None
    umls_aliases: tuple[str, ...] = Field(default_factory=tuple)
    canonical_type: str | None = None
    umls_cui: str | None = None

    @field_validator("concept_name")
    @classmethod
    def validate_concept_name(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Concept name must be a non-empty string")
        return normalized

    @field_validator(
        "name",
        "normalized_name",
        "umls_canonical_name",
        "canonical_type",
        "umls_cui",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("umls_aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        raw_values = [value] if isinstance(value, str) else list(value)
        aliases: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            alias = str(item).strip()
            if not alias:
                continue
            key = alias.casefold()
            if key in seen:
                continue
            seen.add(key)
            aliases.append(alias)
        return tuple(aliases)


class ConceptCatalogueProviderProtocol(Protocol):
    """Graph tool interface required by Concept seeders."""

    def list_concept_catalogue(
        self,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[dict[str, Any]]: ...


class ConceptSeederProtocol(Protocol):
    """Common interface for local Concept seeders."""

    name: str
    concepts_per_term: int

    def seed_concepts(
        self,
        terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> dict[str, list[ConceptSeed]]: ...


class LexicalConceptSeeder:
    """Select local Concepts with explicit deterministic lexical matching."""

    name = "lexical_concept_seeder"

    def __init__(
        self,
        catalogue_provider: ConceptCatalogueProviderProtocol,
        *,
        concepts_per_term: int = 3,
    ) -> None:
        self.catalogue_provider = catalogue_provider
        self.concepts_per_term = _validate_concepts_per_term(
            concepts_per_term
        )

    def seed_concepts(
        self,
        terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> dict[str, list[ConceptSeed]]:
        normalized_terms = _normalize_terms(terms)
        catalogue = _catalogue_records(
            self.catalogue_provider.list_concept_catalogue(
                document_ids=document_ids,
            )
        )

        output: dict[str, list[ConceptSeed]] = {}
        for term in normalized_terms:
            matches: list[tuple[int, str, str, ConceptSeed]] = []
            for concept in catalogue:
                match = _lexical_match(term, concept)
                if match is None:
                    continue
                category, match_type, matched_value = match
                seed = ConceptSeed(
                    query_term=term,
                    concept_name=concept.concept_name,
                    canonical_type=concept.canonical_type,
                    umls_cui=None,
                    method="lexical",
                    match_type=match_type,
                    matched_value=matched_value,
                    seed_rank=1,
                )
                matches.append(
                    (category, concept.concept_name.casefold(), concept.concept_name, seed)
                )

            matches.sort(key=lambda item: (item[0], item[1], item[2]))
            ranked = [
                seed.model_copy(update={"seed_rank": rank})
                for rank, (_, _, _, seed) in enumerate(
                    _deduplicate_seed_candidates(matches),
                    start=1,
                )
            ]
            output[term] = ranked[: self.concepts_per_term]

        return output


class _OpenAIEmbeddingEncoder:
    """Small synchronous adapter around the OpenAI Embeddings API."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int | None = None,
        batch_size: int = 128,
        client: Any | None = None,
    ) -> None:
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("model must be a non-empty string")
        self.dimensions = _validate_embedding_dimensions(dimensions)
        self.batch_size = _validate_embedding_batch_size(batch_size)

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "The 'openai' package is required for embedding-based "
                    "Concept seeding"
                ) from exc
            client = OpenAI()

        self.client = client

    def encode(
        self,
        texts: Sequence[str],
        **_: Any,
    ) -> np.ndarray:
        normalized_texts = [str(text).strip() for text in texts]
        if not normalized_texts:
            return np.zeros((0, 0), dtype=np.float32)
        if any(not text for text in normalized_texts):
            raise ValueError("Embedding inputs must be non-empty strings")

        vectors: list[list[float]] = []
        for start in range(0, len(normalized_texts), self.batch_size):
            batch = normalized_texts[start : start + self.batch_size]
            request: dict[str, Any] = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float",
            }
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**request)
            ordered = sorted(
                response.data,
                key=lambda item: int(item.index),
            )
            if len(ordered) != len(batch):
                raise RuntimeError(
                    "OpenAI returned an unexpected number of embeddings: "
                    f"expected {len(batch)}, received {len(ordered)}"
                )
            vectors.extend(
                [float(value) for value in item.embedding]
                for item in ordered
            )

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(normalized_texts):
            raise RuntimeError(
                "OpenAI returned an invalid embedding matrix shape"
            )
        return matrix


class EmbeddingConceptSeeder:
    """Select local Concepts by cosine similarity to each plan term."""

    name = "embedding_concept_seeder"

    def __init__(
        self,
        catalogue_provider: ConceptCatalogueProviderProtocol,
        *,
        embedding_model: str,
        concepts_per_term: int = 3,
        min_similarity: float | None = None,
        keep_best_below_min_similarity: bool = False,
        cache_path: str | Path | None = None,
        query_cache_path: str | Path | None = None,
        query_cache_read_only: bool = False,
        encoder: Any | None = None,
        embedding_dimensions: int | None = None,
        embedding_batch_size: int = 128,
        openai_client: Any | None = None,
    ) -> None:
        model_id = str(embedding_model).strip()
        if not model_id:
            raise ValueError("embedding_model must be a non-empty string")

        self.catalogue_provider = catalogue_provider
        self.embedding_provider = "openai"
        self.embedding_model = model_id
        self.embedding_dimensions = _validate_embedding_dimensions(
            embedding_dimensions
        )
        self.embedding_batch_size = _validate_embedding_batch_size(
            embedding_batch_size
        )
        self.concepts_per_term = _validate_concepts_per_term(
            concepts_per_term
        )
        self.min_similarity = _validate_min_similarity(
            min_similarity
        )
        self.keep_best_below_min_similarity = bool(
            keep_best_below_min_similarity
        )
        self.cache_path = (
            Path(cache_path).expanduser() if cache_path is not None else None
        )
        self.query_cache_path = (
            Path(query_cache_path).expanduser()
            if query_cache_path is not None
            else None
        )
        self.query_cache_read_only = bool(query_cache_read_only)
        self._encoder = encoder
        self._openai_client = openai_client
        self._concepts: list[ConceptCatalogueRecord] = []
        self._concept_matrix: np.ndarray | None = None
        self._prepared_cache_key: str | None = None
        self._prepared_document_ids: tuple[str, ...] | None = None

        self.catalogue_size = 0
        self.catalogue_hash: str | None = None
        self.catalogue_build_load_seconds = 0.0
        self.model_load_seconds = 0.0
        self.loaded_from_cache = False
        self.resolved_cache_file: str | None = None
        self.query_embedding_cache_hits = 0
        self.query_embedding_cache_misses = 0

    def prepare(
        self,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> None:
        normalized_document_ids = _normalize_optional_values(document_ids)
        document_id_key = tuple(normalized_document_ids)
        if (
            self._concept_matrix is not None
            and self._prepared_document_ids == document_id_key
        ):
            return

        started = time.perf_counter()

        concepts = _catalogue_records(
            self.catalogue_provider.list_concept_catalogue(
                document_ids=normalized_document_ids,
            )
        )
        catalogue_hash = _catalogue_hash(concepts)
        cache_key = _embedding_cache_key(
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
            catalogue_hash=catalogue_hash,
            document_ids=normalized_document_ids,
        )

        if (
            self._concept_matrix is not None
            and self._prepared_cache_key == cache_key
        ):
            return

        self._concepts = concepts
        self.catalogue_size = len(concepts)
        self.catalogue_hash = catalogue_hash
        self._prepared_cache_key = cache_key
        self._prepared_document_ids = document_id_key
        self.loaded_from_cache = False
        self.resolved_cache_file = None

        cached_matrix = self._load_cached_matrix(
            cache_key=cache_key,
            catalogue_hash=catalogue_hash,
            document_ids=normalized_document_ids,
        )
        if cached_matrix is not None:
            self._concept_matrix = cached_matrix
            self.loaded_from_cache = True
            if concepts:
                self._load_model()
            self.catalogue_build_load_seconds = time.perf_counter() - started
            return

        if not concepts:
            self._concept_matrix = np.zeros((0, 0), dtype=np.float32)
            self.catalogue_build_load_seconds = time.perf_counter() - started
            return

        texts = [_concept_representation(concept) for concept in concepts]
        matrix = self._encode_texts(texts)
        self._concept_matrix = _normalize_matrix(matrix)
        self._save_cached_matrix(
            cache_key=cache_key,
            catalogue_hash=catalogue_hash,
            document_ids=normalized_document_ids,
        )
        self.catalogue_build_load_seconds = time.perf_counter() - started

    @property
    def catalogue_records(self) -> tuple[ConceptCatalogueRecord, ...]:
        """Return the prepared exact-identity Concept catalogue read-only."""

        return tuple(self._concepts)

    def seed_concepts(
        self,
        terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> dict[str, list[ConceptSeed]]:
        normalized_terms = _normalize_terms(terms)
        self.prepare(document_ids=document_ids)

        if self._concept_matrix is None or not self._concepts:
            return {term: [] for term in normalized_terms}

        output: dict[str, list[ConceptSeed]] = {}
        for term in normalized_terms:
            query_vector = self._encode_query_term(term)
            query_vector = _normalize_matrix(query_vector)[0]
            similarities = self._concept_matrix @ query_vector
            ranked_indexes = sorted(
                range(len(self._concepts)),
                key=lambda index: (
                    -float(similarities[index]),
                    self._concepts[index].concept_name.casefold(),
                    self._concepts[index].concept_name,
                ),
            )

            eligible_indexes = [
                index
                for index in ranked_indexes
                if (
                    self.min_similarity is None
                    or float(similarities[index]) >= self.min_similarity
                )
            ]
            if (
                self.keep_best_below_min_similarity
                and not eligible_indexes
                and ranked_indexes
            ):
                eligible_indexes = [ranked_indexes[0]]
            seeds: list[ConceptSeed] = []
            for seed_rank, index in enumerate(
                eligible_indexes[: self.concepts_per_term],
                start=1,
            ):
                concept = self._concepts[index]
                seeds.append(
                    ConceptSeed(
                        query_term=term,
                        concept_name=concept.concept_name,
                        canonical_type=concept.canonical_type,
                        umls_cui=None,
                        method="embedding",
                        match_type="embedding",
                        seed_rank=seed_rank,
                        similarity=float(similarities[index]),
                    )
                )
            output[term] = seeds

        return output

    def _load_model(self) -> Any:
        if self._encoder is not None:
            return self._encoder

        started = time.perf_counter()
        self._encoder = _OpenAIEmbeddingEncoder(
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
            batch_size=self.embedding_batch_size,
            client=self._openai_client,
        )
        self.model_load_seconds += time.perf_counter() - started
        return self._encoder

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        encoder = self._load_model()
        try:
            vectors = encoder.encode(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except TypeError:
            vectors = encoder.encode(list(texts))
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix

    def _encode_query_term(self, term: str) -> np.ndarray:
        """Encode one query term with an optional persistent exact-text cache."""

        cache_file = self._query_cache_file(term)
        metadata = self._query_cache_metadata(term)

        if cache_file is not None and cache_file.is_file():
            try:
                with np.load(cache_file, allow_pickle=False) as data:
                    cached_metadata = json.loads(str(data["metadata"].item()))
                    vector = np.asarray(data["embedding"], dtype=np.float32)
                if cached_metadata == metadata:
                    if vector.ndim == 1:
                        vector = vector.reshape(1, -1)
                    if vector.ndim != 2 or vector.shape[0] != 1:
                        raise ValueError("invalid cached query embedding shape")
                    self.query_embedding_cache_hits += 1
                    return vector


            except (KeyError, ValueError, json.JSONDecodeError):
                if self.query_cache_read_only:
                    raise RuntimeError(
                        f"Invalid frozen query-term embedding cache entry: {cache_file}"
                    )

        if self.query_cache_read_only and cache_file is not None:
            raise FileNotFoundError(
                "Frozen query-term embedding cache miss for "
                f"{term!r}: {cache_file}"
            )

        self.query_embedding_cache_misses += 1
        vector = self._encode_texts([term])

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_file,
                embedding=np.asarray(vector, dtype=np.float32),
                metadata=np.array(json.dumps(metadata, sort_keys=True)),
            )

        return vector

    def cosine_similarities(
        self,
        query_text: str,
        candidate_texts: Sequence[str],
        *,
        cache_path: str | Path | None = None,
        cache_read_only: bool = False,
    ) -> list[float]:
        """Score arbitrary texts with the frozen Concept-embedding backend.

        Contextual target-ranking embeddings may use a dedicated exact-text
        cache.  This cache is deliberately independent from the frozen
        query-term cache used by local Concept seeding.
        """

        normalized_query = str(query_text).strip()
        normalized_candidates = [
            str(text).strip()
            for text in candidate_texts
        ]

        if not normalized_query:
            raise ValueError("query_text must be non-empty")
        if any(not text for text in normalized_candidates):
            raise ValueError(
                "candidate_texts must contain only non-empty strings"
            )
        if not normalized_candidates:
            return []

        contextual_cache_path = (
            Path(cache_path).expanduser()
            if cache_path is not None
            else None
        )
        contextual_cache_read_only = bool(cache_read_only)

        # Fail closed if a contextual experiment is accidentally configured
        # to use the frozen Q2 query-term cache directory.
        if (
            contextual_cache_path is not None
            and self.query_cache_path is not None
            and contextual_cache_path.resolve()
            == self.query_cache_path.resolve()
        ):
            raise ValueError(
                "Contextual embedding cache must be separate from the "
                "frozen query-term embedding cache"
            )

        if (
            contextual_cache_read_only
            and contextual_cache_path is None
        ):
            raise ValueError(
                "cache_read_only=True requires cache_path"
            )

        def encode_contextual_text(text: str) -> np.ndarray:
            metadata = {
                **self._query_cache_metadata(text),
                "cache_role": "contextual_target_ranking_v1",
            }

            cache_file: Path | None = None

            if contextual_cache_path is not None:
                encoded = json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                key = hashlib.sha256(encoded).hexdigest()
                cache_file = contextual_cache_path / f"{key}.npz"

            if cache_file is not None and cache_file.is_file():
                try:
                    with np.load(
                        cache_file,
                        allow_pickle=False,
                    ) as data:
                        cached_metadata = json.loads(
                            str(data["metadata"].item())
                        )
                        vector = np.asarray(
                            data["embedding"],
                            dtype=np.float32,
                        )

                    if cached_metadata != metadata:
                        raise ValueError(
                            "contextual cache metadata mismatch"
                        )

                    if vector.ndim == 1:
                        vector = vector.reshape(1, -1)

                    if vector.ndim != 2 or vector.shape[0] != 1:
                        raise ValueError(
                            "invalid cached contextual embedding shape"
                        )

                    return vector

                except (
                    KeyError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    if contextual_cache_read_only:
                        raise RuntimeError(
                            "Invalid frozen contextual embedding cache "
                            f"entry: {cache_file}"
                        ) from exc

            if contextual_cache_read_only:
                raise FileNotFoundError(
                    "Frozen contextual embedding cache miss for "
                    f"{text!r}: {cache_file}"
                )

            vector = self._encode_texts([text])

            if cache_file is not None:
                cache_file.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                np.savez_compressed(
                    cache_file,
                    embedding=np.asarray(
                        vector,
                        dtype=np.float32,
                    ),
                    metadata=np.array(
                        json.dumps(metadata, sort_keys=True)
                    ),
                )

            return vector

        query_vector = _normalize_matrix(
            encode_contextual_text(normalized_query)
        )[0]

        candidate_matrix = np.concatenate(
            [
                encode_contextual_text(text)
                for text in normalized_candidates
            ],
            axis=0,
        )
        candidate_matrix = _normalize_matrix(candidate_matrix)

        return [
            float(value)
            for value in candidate_matrix @ query_vector
        ]

    def _query_cache_metadata(self, term: str) -> dict[str, Any]:
        return {
            "schema_version": "query_term_embedding_cache_v1",
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "term": str(term),
        }

    def _query_cache_file(self, term: str) -> Path | None:
        if self.query_cache_path is None:
            return None
        encoded = json.dumps(
            self._query_cache_metadata(term),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        return self.query_cache_path / f"{key}.npz"

    def _load_cached_matrix(
        self,
        *,
        cache_key: str,
        catalogue_hash: str,
        document_ids: Sequence[str],
    ) -> np.ndarray | None:
        cache_file = self._cache_file(cache_key)
        if cache_file is None or not cache_file.is_file():
            return None

        with np.load(cache_file, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata != _cache_metadata(
                cache_key=cache_key,
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
                embedding_dimensions=self.embedding_dimensions,
                catalogue_hash=catalogue_hash,
                document_ids=document_ids,
                concepts=self._concepts,
            ):
                return None
            matrix = np.asarray(data["embeddings"], dtype=np.float32)

        if matrix.shape[0] != len(self._concepts):
            return None

        self.resolved_cache_file = str(cache_file)
        # The cache stores the already L2-normalized float32 matrix written by
        # _save_cached_matrix(). Re-normalizing it on load changes some rows by
        # a few float32 ULPs and breaks exact build-vs-load reproducibility of
        # ConceptSeed.similarity. Return the persisted representation verbatim.
        return matrix

    def _save_cached_matrix(
        self,
        *,
        cache_key: str,
        catalogue_hash: str,
        document_ids: Sequence[str],
    ) -> None:
        if self._concept_matrix is None:
            return
        cache_file = self._cache_file(cache_key)
        if cache_file is None:
            return

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        metadata = _cache_metadata(
            cache_key=cache_key,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
            catalogue_hash=catalogue_hash,
            document_ids=document_ids,
            concepts=self._concepts,
        )
        np.savez_compressed(
            cache_file,
            embeddings=self._concept_matrix,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )
        self.resolved_cache_file = str(cache_file)

    def _cache_file(self, cache_key: str) -> Path | None:
        if self.cache_path is None:
            return None
        if self.cache_path.suffix:
            return (
                self.cache_path.parent
                / f"{self.cache_path.stem}-{cache_key}.npz"
            )
        return self.cache_path / f"{cache_key}.npz"


_EXACT_SAFE_WS_RE = re.compile(r"\s+")


def normalize_exact_safe_surface(value: str) -> str:
    """Canonicalize representation only; do not perform semantic rewriting."""

    text = unicodedata.normalize("NFKC", str(value))
    return _EXACT_SAFE_WS_RE.sub(" ", text).strip()


def resolve_exact_safe_seed(
    term: str,
    catalogue: Sequence[ConceptCatalogueRecord],
) -> ConceptSeed | None:
    """Resolve at most one safe exact local Concept for a query term.

    Policy:
    1. prefer one exact-case surface match;
    2. otherwise accept a case-insensitive surface match only when unique;
    3. skip ambiguous casefold collisions.

    Only ``Concept.name`` is considered. No aliases, normalized names, UMLS,
    stemming, substring matching, acronym expansion, or fuzzy matching are
    used by this ablation.
    """

    query_surface = normalize_exact_safe_surface(term)
    if not query_surface:
        return None

    exact_matches: list[ConceptCatalogueRecord] = []
    folded_matches: list[ConceptCatalogueRecord] = []
    query_folded = query_surface.casefold()

    for concept in catalogue:
        value = concept.name or concept.concept_name
        surface = normalize_exact_safe_surface(value)
        if surface == query_surface:
            exact_matches.append(concept)
        if surface.casefold() == query_folded:
            folded_matches.append(concept)

    if len(exact_matches) == 1:
        concept = exact_matches[0]
        match_type = "exact_safe_exact_case"
    elif len(exact_matches) > 1:
        # Exact Concept identity should already be unique, but fail closed if
        # an inconsistent catalogue ever violates that invariant.
        return None
    elif len(folded_matches) == 1:
        concept = folded_matches[0]
        match_type = "exact_safe_unique_casefold"
    else:
        return None

    return ConceptSeed(
        query_term=term,
        concept_name=concept.concept_name,
        canonical_type=concept.canonical_type,
        umls_cui=concept.umls_cui,
        method="lexical",
        match_type=match_type,
        seed_rank=1,
        similarity=None,
        matched_value=concept.name or concept.concept_name,
    )


class ExactSafeAugmentedConceptSeeder:
    """Add one collision-safe exact Concept seed to semantic top-k seeds.

    The semantic seeds remain unchanged. An exact seed is appended only when
    its exact Concept identity is not already present among the semantic
    seeds for that query term. This prevents the exact channel from erasing
    the stored semantic similarity of a duplicate semantic seed.
    """

    name = "embedding_plus_exact_safe_concept_seeder"

    def __init__(self, semantic_seeder: EmbeddingConceptSeeder) -> None:
        self.semantic_seeder = semantic_seeder
        # This is an upper bound: semantic top-k plus at most one exact seed.
        self.concepts_per_term = semantic_seeder.concepts_per_term + 1

    @property
    def catalogue_records(self) -> tuple[ConceptCatalogueRecord, ...]:
        return self.semantic_seeder.catalogue_records

    def seed_concepts(
        self,
        terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> dict[str, list[ConceptSeed]]:
        normalized_terms = _normalize_terms(terms)
        semantic_groups = self.semantic_seeder.seed_concepts(
            normalized_terms,
            document_ids=document_ids,
        )
        catalogue = self.semantic_seeder.catalogue_records

        output: dict[str, list[ConceptSeed]] = {}
        for term in normalized_terms:
            semantic = list(semantic_groups.get(term, ()))
            existing = {seed.concept_name for seed in semantic}
            exact_seed = resolve_exact_safe_seed(term, catalogue)
            if exact_seed is not None and exact_seed.concept_name not in existing:
                semantic.append(exact_seed)
            output[term] = semantic
        return output


class SameCUIAugmentedConceptSeeder:
    """Expand local Concept seeds through exact shared UMLS CUI identity.

    This is a local equivalence closure, not UMLS relational expansion.  For
    each base seed with a non-empty ``umls_cui``, every other *local* Concept
    in the same document-scoped catalogue with exactly the same CUI is added
    for the same query term.  The closure seed inherits the parent seed rank
    and semantic similarity because it represents the same normalized UMLS
    concept, while ``match_type`` records the provenance explicitly.

    No external UMLS nodes, DIRECT/BRIDGE relations, aliases, fuzzy matching,
    or cross-CUI traversal are used.
    """

    name = "same_cui_augmented_concept_seeder"

    def __init__(self, base_seeder: ConceptSeederProtocol) -> None:
        self.base_seeder = base_seeder
        self.concepts_per_term = int(getattr(base_seeder, "concepts_per_term", 0))

    @property
    def catalogue_records(self) -> tuple[ConceptCatalogueRecord, ...]:
        records = getattr(self.base_seeder, "catalogue_records", None)
        if records is None:
            semantic = getattr(self.base_seeder, "semantic_seeder", None)
            records = getattr(semantic, "catalogue_records", None)
        if records is None:
            raise TypeError(
                "same-CUI augmentation requires a seeder exposing "
                "catalogue_records"
            )
        return tuple(records)

    def seed_concepts(
        self,
        terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> dict[str, list[ConceptSeed]]:
        normalized_terms = _normalize_terms(terms)
        base_groups = self.base_seeder.seed_concepts(
            normalized_terms,
            document_ids=document_ids,
        )
        catalogue = self.catalogue_records

        by_cui: dict[str, list[ConceptCatalogueRecord]] = {}
        for concept in catalogue:
            cui = str(concept.umls_cui or "").strip()
            if not cui:
                continue
            by_cui.setdefault(cui, []).append(concept)
        for concepts in by_cui.values():
            concepts.sort(
                key=lambda item: (item.concept_name.casefold(), item.concept_name)
            )

        # EmbeddingConceptSeeder historically serializes semantic seeds with
        # ``umls_cui=None`` even though the prepared catalogue record contains
        # the normalized CUI.  Resolve missing seed metadata through the exact
        # Concept identity rather than changing the frozen baseline seeder.
        # Exact-name lookup is intentional: biomedical case can distinguish
        # different Concepts (for example ``DMD`` vs ``dmd``).
        by_exact_name = {concept.concept_name: concept for concept in catalogue}

        output: dict[str, list[ConceptSeed]] = {}
        for term in normalized_terms:
            base = list(base_groups.get(term, ()))
            existing = {seed.concept_name for seed in base}
            additions: list[ConceptSeed] = []

            # Deterministic parent order ensures stable provenance when more
            # than one base seed points to the same CUI.
            parents = sorted(
                base,
                key=lambda seed: (
                    seed.seed_rank,
                    seed.concept_name.casefold(),
                    seed.concept_name,
                ),
            )
            for parent in parents:
                cui = str(parent.umls_cui or "").strip()
                if not cui:
                    source_record = by_exact_name.get(parent.concept_name)
                    if source_record is not None:
                        cui = str(source_record.umls_cui or "").strip()
                if not cui:
                    continue
                for sibling in by_cui.get(cui, ()):
                    if sibling.concept_name in existing:
                        continue
                    additions.append(
                        ConceptSeed(
                            query_term=parent.query_term,
                            concept_name=sibling.concept_name,
                            canonical_type=sibling.canonical_type,
                            umls_cui=cui,
                            method=parent.method,
                            match_type="same_cui_local_closure",
                            seed_rank=parent.seed_rank,
                            similarity=parent.similarity,
                            matched_value=parent.concept_name,
                        )
                    )
                    existing.add(sibling.concept_name)

            output[term] = base + additions

        return output


def flatten_seed_groups(
    seed_groups: Mapping[str, Sequence[ConceptSeed]],
) -> list[ConceptSeed]:
    """Return de-duplicated seeds in query-term/rank order."""

    output: list[ConceptSeed] = []
    seen: set[tuple[str, str]] = set()

    for seeds in seed_groups.values():
        for seed in seeds:
            key = (seed.query_term.casefold(), seed.concept_name)
            if key in seen:
                continue
            seen.add(key)
            output.append(seed)

    return output


def _catalogue_records(
    rows: Sequence[Mapping[str, Any]],
) -> list[ConceptCatalogueRecord]:
    records: list[ConceptCatalogueRecord] = []
    seen: set[str] = set()

    for row in rows:
        concept_name = row.get("concept_name") or row.get("name")
        if concept_name is None:
            continue
        payload = {
            "concept_name": concept_name,
            "name": row.get("name") or concept_name,
            "normalized_name": row.get("normalized_name"),
            "umls_canonical_name": row.get("umls_canonical_name"),
            "umls_aliases": row.get("umls_aliases"),
            "canonical_type": row.get("canonical_type"),
            "umls_cui": row.get("umls_cui"),
        }
        record = ConceptCatalogueRecord.model_validate(payload)
        key = record.concept_name
        if key in seen:
            continue
        seen.add(key)
        records.append(record)

    records.sort(
        key=lambda item: (item.concept_name.casefold(), item.concept_name)
    )
    return records


def _lexical_match(
    query_term: str,
    concept: ConceptCatalogueRecord,
) -> tuple[int, str, str] | None:
    """Match against local Concept.name only."""

    term = query_term.casefold()
    value = concept.name or concept.concept_name
    value_key = value.casefold()

    if value_key == term:
        return (1, "exact_name", value)
    if value_key.startswith(term):
        return (2, "prefix", value)
    if term in value_key:
        return (3, "partial", value)
    return None


def _deduplicate_seed_candidates(
    matches: Sequence[tuple[int, str, str, ConceptSeed]],
) -> list[tuple[int, str, str, ConceptSeed]]:
    output: list[tuple[int, str, str, ConceptSeed]] = []
    seen: set[tuple[str, str]] = set()
    for match in matches:
        seed = match[3]
        key = (seed.query_term.casefold(), seed.concept_name)
        if key in seen:
            continue
        seen.add(key)
        output.append(match)
    return output


def _concept_representation(
    concept: ConceptCatalogueRecord,
) -> str:
    # KG_LOCAL_ONLY_CONCEPT_REPRESENTATION
    values = [
        concept.name or concept.concept_name,
    ]
    return "; ".join(value for value in values if value)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = np.asarray(matrix, dtype=np.float32)
    if normalized.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional")
    if normalized.shape[0] == 0:
        return normalized
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return normalized / norms


def _catalogue_hash(
    concepts: Sequence[ConceptCatalogueRecord],
) -> str:
    """Hash only fields that affect local-only Concept embeddings."""

    payload = [
        {
            "concept_name": concept.concept_name,
            "name": concept.name,
            "canonical_type": concept.canonical_type,
        }
        for concept in sorted(
            concepts,
            key=lambda item: (item.concept_name.casefold(), item.concept_name),
        )
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _embedding_cache_key(
    *,
    embedding_model: str,
    catalogue_hash: str,
    document_ids: Sequence[str],
    embedding_provider: str = "openai",
    embedding_dimensions: int | None = None,
) -> str:
    payload = {
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimensions": embedding_dimensions,
        "catalogue_hash": catalogue_hash,
        "document_ids": list(document_ids),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_metadata(
    *,
    cache_key: str,
    embedding_model: str,
    catalogue_hash: str,
    document_ids: Sequence[str],
    concepts: Sequence[ConceptCatalogueRecord],
    embedding_provider: str = "openai",
    embedding_dimensions: int | None = None,
) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimensions": embedding_dimensions,
        "catalogue_hash": catalogue_hash,
        "document_ids": list(document_ids),
        "concept_names": [concept.concept_name for concept in concepts],
    }


def _normalize_terms(values: Sequence[str] | str) -> list[str]:
    raw_values = [values] if isinstance(values, str) else list(values)
    output: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        term = str(value).strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(term)
    if not output:
        raise ValueError("At least one non-empty retrieval term is required")
    return output


def _normalize_optional_values(
    values: Sequence[str] | str | None,
) -> list[str]:
    """Normalize an optional sequence without requiring any values."""

    if values is None:
        return []

    raw_values = [values] if isinstance(values, str) else list(values)
    output: list[str] = []
    seen: set[str] = set()

    for value in raw_values:
        item = str(value).strip()
        if not item:
            continue

        key = item.casefold()
        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def _validate_min_similarity(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_similarity must be a float") from exc
    if normalized < -1.0 or normalized > 1.0:
        raise ValueError(
            "min_similarity must be between -1.0 and 1.0"
        )
    return normalized


def _validate_embedding_dimensions(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding_dimensions must be an integer") from exc
    if normalized < 1:
        raise ValueError("embedding_dimensions must be positive")
    return normalized


def _validate_embedding_batch_size(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding_batch_size must be an integer") from exc
    if normalized < 1 or normalized > 2048:
        raise ValueError(
            "embedding_batch_size must be between 1 and 2048"
        )
    return normalized


def _validate_concepts_per_term(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("concepts_per_term must be an integer") from exc
    if normalized < 1 or normalized > 100:
        raise ValueError("concepts_per_term must be between 1 and 100")
    return normalized
