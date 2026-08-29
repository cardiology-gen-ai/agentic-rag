from __future__ import annotations

import numpy as np
import pytest

from agentic_rag.kg.concept_seeders import EmbeddingConceptSeeder


class CatalogueProvider:
    def __init__(self, rows):
        self.rows = list(rows)

    def list_concept_catalogue(self, *, document_ids=None):
        return list(self.rows)


class CountingEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(tuple(str(x) for x in texts))
        vectors = []
        for text in texts:
            value = str(text).casefold()
            if value == "query" or value.startswith("strong"):
                vectors.append([1.0, 0.0])
            elif value.startswith("weak"):
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return np.asarray(vectors, dtype=np.float32)


def provider():
    return CatalogueProvider(
        [
            {
                "concept_name": "strong",
                "name": "strong",
                "canonical_type": "disease",
            },
            {
                "concept_name": "weak",
                "name": "weak",
                "canonical_type": "disease",
            },
        ]
    )


def test_query_term_embedding_cache_reuses_frozen_vector(tmp_path) -> None:
    concept_cache = tmp_path / "concept-cache"
    query_cache = tmp_path / "query-cache"

    first_encoder = CountingEncoder()
    first = EmbeddingConceptSeeder(
        provider(),
        embedding_model="test-model",
        concepts_per_term=2,
        cache_path=concept_cache,
        query_cache_path=query_cache,
        encoder=first_encoder,
    )
    first.seed_concepts(["query"])
    assert first.query_embedding_cache_misses == 1
    assert first.query_embedding_cache_hits == 0
    assert any(call == ("query",) for call in first_encoder.calls)

    second_encoder = CountingEncoder()
    second = EmbeddingConceptSeeder(
        provider(),
        embedding_model="test-model",
        concepts_per_term=2,
        cache_path=concept_cache,
        query_cache_path=query_cache,
        query_cache_read_only=True,
        encoder=second_encoder,
    )
    seeds = second.seed_concepts(["query"])["query"]

    assert [seed.concept_name for seed in seeds] == ["strong", "weak"]
    assert second.query_embedding_cache_hits == 1
    assert second.query_embedding_cache_misses == 0
    assert second_encoder.calls == []


def test_query_term_embedding_cache_read_only_rejects_miss(tmp_path) -> None:
    seeder = EmbeddingConceptSeeder(
        provider(),
        embedding_model="test-model",
        concepts_per_term=1,
        query_cache_path=tmp_path / "query-cache",
        query_cache_read_only=True,
        encoder=CountingEncoder(),
    )

    with pytest.raises(FileNotFoundError, match="query-term embedding cache miss"):
        seeder.seed_concepts(["query"])
