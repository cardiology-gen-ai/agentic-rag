from __future__ import annotations

import numpy as np

from agentic_rag.kg.concept_seeders import EmbeddingConceptSeeder


class _CatalogueProvider:
    def list_concept_catalogue(self, *, document_ids=None):
        return [
            {"concept_name": "alpha", "canonical_type": "disease"},
            {"concept_name": "beta", "canonical_type": "disease"},
        ]


class _DeterministicEncoder:
    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            value = str(text)
            if "alpha" in value:
                rows.append([0.1, 0.2, 0.3])
            elif "beta" in value:
                rows.append([0.3, 0.2, 0.1])
            elif value == "query":
                rows.append([0.2, 0.4, 0.1])
            else:
                rows.append([0.4, 0.1, 0.2])
        return np.asarray(rows, dtype=np.float32)


def _make_seeder(cache_path):
    return EmbeddingConceptSeeder(
        _CatalogueProvider(),
        embedding_model="test-model",
        concepts_per_term=2,
        cache_path=cache_path,
        encoder=_DeterministicEncoder(),
    )


def test_concept_embedding_cache_build_and_load_are_exactly_equivalent(tmp_path):
    cache_path = tmp_path / "concept-cache"

    built = _make_seeder(cache_path)
    built_seeds = [
        seed.model_dump(mode="json")
        for seed in built.seed_concepts(["query"])["query"]
    ]
    assert built.loaded_from_cache is False

    loaded = _make_seeder(cache_path)
    loaded_seeds = [
        seed.model_dump(mode="json")
        for seed in loaded.seed_concepts(["query"])["query"]
    ]
    assert loaded.loaded_from_cache is True

    assert built_seeds == loaded_seeds
