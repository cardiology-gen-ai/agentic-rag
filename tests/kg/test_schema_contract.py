from agentic_rag.kg.schema import (
    BASE_GRAPH_SCHEMA,
    CANONICAL_CONCEPT_TYPES,
    ENTITY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SECTION_RETRIEVAL_CONTRACT,
    SECTION_VIEW_SCHEMA_VERSION,
    build_text2cypher_schema,
)


EXPECTED_ENTITY_TYPES = {
    "disease",
    "clinical_finding",
    "exposure_or_lifestyle_factor",
    "genetic_factor",
    "biomarker",
    "diagnostic_test",
    "score_or_risk_model",
    "drug_or_drug_class",
    "procedure_or_intervention",
    "device",
    "care_strategy",
    "anatomical_structure",
    "clinical_outcome",
    "microorganism_or_pathogen",
    "population_or_patient_group",
}


def test_schema_versions_match_data_etl_contract() -> None:
    assert SCHEMA_VERSION == "2.0"
    assert ENTITY_SCHEMA_VERSION == "2.1"
    assert SECTION_VIEW_SCHEMA_VERSION == "1"


def test_canonical_entity_types_match_data_etl_2_1() -> None:
    assert set(CANONICAL_CONCEPT_TYPES) == EXPECTED_ENTITY_TYPES

    obsolete_types = {
        "risk_factor",
        "imaging_modality",
        "complication_or_comorbidity",
    }
    assert obsolete_types.isdisjoint(CANONICAL_CONCEPT_TYPES)


def test_schema_excludes_unmaterialized_recommendation_contract() -> None:
    assert "Recommendation nodes and recommendation-specific" in BASE_GRAPH_SCHEMA
    assert "CONTAINS_RECOMMENDATION" not in BASE_GRAPH_SCHEMA
    assert "RECOMMENDS_ACTION" not in BASE_GRAPH_SCHEMA


def test_retrieval_contract_uses_section_view_eligibility() -> None:
    assert 's.section_view_role = "retrieval"' in SECTION_RETRIEVAL_CONTRACT
    assert "coalesce(s.embed, false) = true" in SECTION_RETRIEVAL_CONTRACT
    assert "coalesce(s.excluded, false) = false" in SECTION_RETRIEVAL_CONTRACT
    assert 'trim(coalesce(s.text, "")) <> ""' in SECTION_RETRIEVAL_CONTRACT
    assert "Do not require s.has_embedding = true" in SECTION_RETRIEVAL_CONTRACT


def test_text2cypher_scope_is_normalized_and_deterministic() -> None:
    schema = build_text2cypher_schema(
        [" Cardiomyopathies_2023 ", "", "CCS_2024", "CCS_2024"]
    )

    assert (
        '["CCS_2024", "Cardiomyopathies_2023"]'
        in schema
    )
