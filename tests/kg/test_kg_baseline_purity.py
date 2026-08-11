from __future__ import annotations

import numpy as np

from agentic_rag.kg.candidate_generators import _CONCEPT_GRAPH_SEED_MATCH
from agentic_rag.kg.concept_seeders import (
    ConceptCatalogueRecord,
    EmbeddingConceptSeeder,
    LexicalConceptSeeder,
    _catalogue_hash,
    _concept_representation,
    _lexical_match,
)
from agentic_rag.kg.tools import _SEARCH_SECTIONS_BY_CONCEPTS


class CatalogueProvider:
    def __init__(self, rows):
        self.rows = list(rows)

    def list_concept_catalogue(self, *, document_ids=None):
        return list(self.rows)


class DeterministicEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            value = str(text).casefold()
            if value == "query":
                vectors.append([1.0, 0.0])
            elif value.startswith("strong"):
                vectors.append([1.0, 0.0])
            elif value.startswith("weak"):
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return np.asarray(vectors, dtype=np.float32)


def test_direct_mentions_matching_does_not_use_umls_fields() -> None:
    query = _SEARCH_SECTIONS_BY_CONCEPTS.casefold()
    assert "c.normalized_name" not in query
    assert "c.umls_canonical_name" not in query
    assert "c.umls_aliases" not in query
    assert "exact_umls" not in query
    assert "exact_normalized_name" not in query
    assert "c.name" in query


def test_concept_graph_seed_matching_is_local_only() -> None:
    query = _CONCEPT_GRAPH_SEED_MATCH.casefold()
    assert "seed.normalized_name" not in query
    assert "seed.umls_canonical_name" not in query
    assert "seed.umls_aliases" not in query
    assert "exact_umls" not in query
    assert "exact_normalized_name" not in query
    assert "seed.name" in query


def test_lexical_seeder_does_not_match_umls_only_alias() -> None:
    concept = ConceptCatalogueRecord(
        concept_name="hypertrophic cardiomyopathy",
        name="hypertrophic cardiomyopathy",
        normalized_name="hcm normalized",
        umls_canonical_name="HCM",
        umls_aliases=("HCM alias",),
        canonical_type="disease",
        umls_cui="C0000001",
    )

    assert _lexical_match("HCM", concept) is None
    assert _lexical_match(
        "hypertrophic cardiomyopathy",
        concept,
    ) == (1, "exact_name", "hypertrophic cardiomyopathy")


def test_local_lexical_seed_trace_excludes_umls_cui() -> None:
    provider = CatalogueProvider(
        [{
            "concept_name": "hypertrophic cardiomyopathy",
            "name": "hypertrophic cardiomyopathy",
            "canonical_type": "disease",
            "umls_cui": "C0000001",
            "umls_canonical_name": "HCM",
            "umls_aliases": ["HCM"],
        }]
    )
    seeder = LexicalConceptSeeder(provider, concepts_per_term=1)
    result = seeder.seed_concepts(["hypertrophic cardiomyopathy"])
    seed = result["hypertrophic cardiomyopathy"][0]
    assert seed.concept_name == "hypertrophic cardiomyopathy"
    assert seed.umls_cui is None


def test_embedding_representation_uses_local_surface_only() -> None:
    concept = ConceptCatalogueRecord(
        concept_name="hypertrophic cardiomyopathy",
        name="hypertrophic cardiomyopathy",
        normalized_name="UMLS normalized value",
        umls_canonical_name="UMLS canonical value",
        umls_aliases=("UMLS alias",),
        canonical_type="disease",
        umls_cui="C0000001",
    )
    assert _concept_representation(concept) == (
        "hypertrophic cardiomyopathy"
    )


def test_embedding_catalogue_hash_ignores_umls_changes() -> None:
    first = ConceptCatalogueRecord(
        concept_name="hypertrophic cardiomyopathy",
        name="hypertrophic cardiomyopathy",
        canonical_type="disease",
        umls_cui="C0000001",
        umls_canonical_name="First",
        umls_aliases=("Alias A",),
    )
    second = first.model_copy(
        update={
            "umls_cui": "C9999999",
            "umls_canonical_name": "Different",
            "umls_aliases": ("Alias B",),
        }
    )
    assert _catalogue_hash([first]) == _catalogue_hash([second])


def test_embedding_min_similarity_filters_weak_seed() -> None:
    provider = CatalogueProvider(
        [
            {
                "concept_name": "strong",
                "name": "strong",
                "canonical_type": "disease",
                "umls_cui": "C111",
            },
            {
                "concept_name": "weak",
                "name": "weak",
                "canonical_type": "disease",
                "umls_cui": "C222",
            },
        ]
    )
    seeder = EmbeddingConceptSeeder(
        provider,
        embedding_model="test-model",
        concepts_per_term=2,
        min_similarity=0.5,
        encoder=DeterministicEncoder(),
    )
    seeds = seeder.seed_concepts(["query"])["query"]
    assert [seed.concept_name for seed in seeds] == ["strong"]
    assert seeds[0].similarity == 1.0
    assert seeds[0].umls_cui is None


def test_embedding_threshold_default_preserves_previous_top_n_behavior() -> None:
    provider = CatalogueProvider(
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
    seeder = EmbeddingConceptSeeder(
        provider,
        embedding_model="test-model",
        concepts_per_term=2,
        encoder=DeterministicEncoder(),
    )
    seeds = seeder.seed_concepts(["query"])["query"]
    assert [seed.concept_name for seed in seeds] == ["strong", "weak"]
