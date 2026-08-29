from __future__ import annotations

from agentic_rag.kg.concept_seeders import (
    ConceptSeed,
    LexicalConceptSeeder,
    flatten_seed_groups,
)


class _CatalogueProvider:
    def list_concept_catalogue(self, *, document_ids=None):
        return [
            {
                "concept_name": "DMD",
                "name": "DMD",
                "canonical_type": "genetic_factor",
                "umls_cui": "C1414083",
            },
            {
                "concept_name": "dmd",
                "name": "dmd",
                "canonical_type": "disease",
                "umls_cui": "C0013264",
            },
        ]


def test_concept_catalogue_identity_preserves_case_distinct_concepts():
    seeder = LexicalConceptSeeder(
        _CatalogueProvider(),
        concepts_per_term=3,
    )

    seeds = seeder.seed_concepts(["DMD"])["DMD"]

    assert [(seed.concept_name, seed.canonical_type) for seed in seeds] == [
        ("DMD", "genetic_factor"),
        ("dmd", "disease"),
    ]
    assert [seed.seed_rank for seed in seeds] == [1, 2]


def test_flatten_seed_groups_does_not_casefold_concept_identity():
    seeds = [
        ConceptSeed(
            query_term="DMD",
            concept_name="DMD",
            canonical_type="genetic_factor",
            method="embedding",
            match_type="embedding",
            seed_rank=1,
            similarity=0.9,
        ),
        ConceptSeed(
            query_term="dmd",
            concept_name="dmd",
            canonical_type="disease",
            method="embedding",
            match_type="embedding",
            seed_rank=2,
            similarity=0.8,
        ),
    ]

    flattened = flatten_seed_groups({"DMD": seeds})

    assert [seed.concept_name for seed in flattened] == ["DMD", "dmd"]
