"""Concept seeders for controlled MENTIONS ablations."""

from __future__ import annotations

import hashlib
import json
import time
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
            matches: list[tuple[int, str, ConceptSeed]] = []
            for concept in catalogue:
                match = _lexical_match(term, concept)
                if match is None:
                    continue
                category, match_type, matched_value = match
                seed = ConceptSeed(
                    query_term=term,
                    concept_name=concept.concept_name,
                    canonical_type=concept.canonical_type,
                    umls_cui=concept.umls_cui,
                    method="lexical",
                    match_type=match_type,
                    matched_value=matched_value,
                    seed_rank=1,
                )
                matches.append(
                    (category, concept.concept_name.casefold(), seed)
                )

            matches.sort(key=lambda item: (item[0], item[1]))
            ranked = [
                seed.model_copy(update={"seed_rank": rank})
                for rank, (_, _, seed) in enumerate(
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
        cache_path: str | Path | None = None,
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
        self.cache_path = (
            Path(cache_path).expanduser() if cache_path is not None else None
        )
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
            query_vector = self._encode_texts([term])
            query_vector = _normalize_matrix(query_vector)[0]
            similarities = self._concept_matrix @ query_vector
            ranked_indexes = sorted(
                range(len(self._concepts)),
                key=lambda index: (
                    -float(similarities[index]),
                    self._concepts[index].concept_name.casefold(),
                ),
            )

            seeds: list[ConceptSeed] = []
            for seed_rank, index in enumerate(
                ranked_indexes[: self.concepts_per_term],
                start=1,
            ):
                concept = self._concepts[index]
                seeds.append(
                    ConceptSeed(
                        query_term=term,
                        concept_name=concept.concept_name,
                        canonical_type=concept.canonical_type,
                        umls_cui=concept.umls_cui,
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
        return _normalize_matrix(matrix)

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


def flatten_seed_groups(
    seed_groups: Mapping[str, Sequence[ConceptSeed]],
) -> list[ConceptSeed]:
    """Return de-duplicated seeds in query-term/rank order."""

    output: list[ConceptSeed] = []
    seen: set[tuple[str, str]] = set()

    for seeds in seed_groups.values():
        for seed in seeds:
            key = (seed.query_term.casefold(), seed.concept_name.casefold())
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
        key = record.concept_name.casefold()
        if key in seen:
            continue
        seen.add(key)
        records.append(record)

    records.sort(key=lambda item: item.concept_name.casefold())
    return records


def _lexical_match(
    query_term: str,
    concept: ConceptCatalogueRecord,
) -> tuple[int, str, str] | None:
    term = query_term.casefold()
    fields = [
        (1, "exact_name", concept.name),
        (2, "exact_normalized_name", concept.normalized_name),
        (3, "exact_umls_canonical_name", concept.umls_canonical_name),
    ]
    for category, match_type, value in fields:
        if value and value.casefold() == term:
            return (category, match_type, value)

    for alias in concept.umls_aliases:
        if alias.casefold() == term:
            return (3, "exact_umls_alias", alias)

    prefix_match = _first_text_match(
        term,
        concept,
        predicate=lambda value: value.startswith(term),
    )
    if prefix_match is not None:
        return (4, "prefix", prefix_match)

    partial_match = _first_text_match(
        term,
        concept,
        predicate=lambda value: term in value,
    )
    if partial_match is not None:
        return (5, "partial", partial_match)

    return None


def _first_text_match(
    term: str,
    concept: ConceptCatalogueRecord,
    *,
    predicate: Any,
) -> str | None:
    values = [
        concept.name,
        concept.normalized_name,
        concept.umls_canonical_name,
        *concept.umls_aliases,
    ]
    for value in values:
        if value and predicate(value.casefold()):
            return value
    return None


def _deduplicate_seed_candidates(
    matches: Sequence[tuple[int, str, ConceptSeed]],
) -> list[tuple[int, str, ConceptSeed]]:
    output: list[tuple[int, str, ConceptSeed]] = []
    seen: set[tuple[str, str]] = set()
    for match in matches:
        seed = match[2]
        key = (seed.query_term.casefold(), seed.concept_name.casefold())
        if key in seen:
            continue
        seen.add(key)
        output.append(match)
    return output


def _concept_representation(concept: ConceptCatalogueRecord) -> str:
    values = [
        concept.name,
        concept.normalized_name,
        concept.umls_canonical_name,
        *concept.umls_aliases,
        concept.canonical_type,
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


def _catalogue_hash(concepts: Sequence[ConceptCatalogueRecord]) -> str:
    payload = [
        concept.model_dump(mode="json")
        for concept in sorted(
            concepts,
            key=lambda item: item.concept_name.casefold(),
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
