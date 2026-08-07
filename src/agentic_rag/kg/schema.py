from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Final


# KGRetriever-side contract version.
#
# The authoritative preprocessing contracts are maintained in data-etl:
# - knowledge_graph.entity_schema.ENTITY_SCHEMA_VERSION == "2.1"
# - knowledge_graph.graph_loader.SECTION_VIEW_SCHEMA_VERSION == "1"
SCHEMA_VERSION: Final[str] = "2.0"
ENTITY_SCHEMA_VERSION: Final[str] = "2.1"
SECTION_VIEW_SCHEMA_VERSION: Final[str] = "1"

RETRIEVAL_ROLE: Final[str] = "retrieval"
STRUCTURAL_ROLE: Final[str] = "structural"

CANONICAL_CONCEPT_TYPES: Final[tuple[str, ...]] = (
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
)

# These are type-resolution states, not canonical entity types.
SPECIAL_CONCEPT_TYPE_STATES: Final[tuple[str, ...]] = (
    "ambiguous",
    "no_supported_type",
)

REQUIRED_RESULT_ALIASES: Final[tuple[str, ...]] = (
    "section_uid",
    "document_id",
    "section_id",
    "title",
    "text",
    "matched_concepts",
    "score",
)


def _format_bullets(values: Sequence[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


BASE_GRAPH_SCHEMA: Final[str] = f"""
Graph schema for medical-guideline retrieval.

Authoritative preprocessing contract:
- local entity schema version: {ENTITY_SCHEMA_VERSION}
- retrieval Section-view schema version: {SECTION_VIEW_SCHEMA_VERSION}

Node labels and retrieval-relevant properties:

Document
- doc_id: STRING, unique document identifier.
- retrieval_strategy: STRING or null.
- retrieval_max_level: INTEGER or null.
- aggregation_mode: STRING or null.
- section_view_schema_version: STRING.
- section_view_retrieval_count: INTEGER or null.
- section_view_structural_count: INTEGER or null.
- section_view_aggregated_count: INTEGER or null.

Section
- uid: STRING, unique opaque Section-node identifier.
- chunk_id: STRING.
- doc_id: STRING, owning document identifier.
- section_id: STRING.
- printed_section_id: STRING or null.
- title: STRING or null.
- section_type: STRING or null.
- level: INTEGER.
- text: STRING.
- is_empty: BOOLEAN.
- excluded: BOOLEAN.
- embed: BOOLEAN.
- page_start: INTEGER or null.
- page_end: INTEGER or null.
- parent_section_id: STRING or null.
- part_index: INTEGER or null.
- part_count: INTEGER or null.
- section_view_role: STRING, either "{RETRIEVAL_ROLE}" or "{STRUCTURAL_ROLE}".
- section_view_schema_version: STRING.
- section_view_order: INTEGER.
- retrieval_order: INTEGER or null.
- retrieval_unit_id: STRING or null.
- retrieval_strategy: STRING.
- aggregation_mode: STRING.
- aggregation_max_level: INTEGER or null.
- is_aggregated: BOOLEAN.
- content_owner_section_id: STRING or null.
- source_section_ids: LIST<STRING>.
- source_chunk_ids: LIST<STRING>.
- represented_section_ids: LIST<STRING>.
- structural_context_section_ids: LIST<STRING>.
- absorbed_section_ids: LIST<STRING>.
- absorbed_source_section_ids: LIST<STRING>.
- has_embedding: BOOLEAN.
- entity_extracted: BOOLEAN.
- entity_extraction_status: STRING or null.

Concept
- name: STRING, unique normalized lowercase concept name.
- canonical_type: STRING or null.
- observed_types: LIST<STRING>.
- invalid_observed_types: LIST<STRING> or null.
- type_support_pairs: LIST<STRING> or null.
- type_resolution_status: STRING or null.
- needs_type_review: BOOLEAN or null.
- normalized_name: STRING or null.
- normalization_status: STRING or null.
- normalization_method: STRING or null.
- umls_cui: STRING or null.
- umls_canonical_name: STRING or null.
- umls_definition: STRING or null.
- umls_aliases: LIST<STRING> or null.
- umls_semantic_types: LIST<STRING> or null.
- umls_score: FLOAT or null.

Canonical Concept types from entity schema {ENTITY_SCHEMA_VERSION}:
{_format_bullets(CANONICAL_CONCEPT_TYPES)}

Concept.canonical_type may temporarily be null before type resolution.
After type resolution it may also contain one of these non-canonical review states:
{_format_bullets(SPECIAL_CONCEPT_TYPE_STATES)}

Do not treat contextual roles such as risk factor, complication, comorbidity,
outcome role, or target population as additional canonical entity types.
A disease remains a disease when it acts as a risk factor, complication,
comorbidity, or disease outcome in a specific sentence.

Relationships:

(:Document)-[:HAS_SECTION]->(:Section)
- Links a Document to every retained Section node in its active Section view.

(:Section)-[:HAS_CHILD]->(:Section)
- Preserves the retained document hierarchy.
- Either endpoint may be a retrieval or structural Section.

(:Section)-[:NEXT]->(:Section)
- Links consecutive retrieval Sections only.
- It does not include structural Sections.

(:Section)-[:MENTIONS]->(:Concept)
- Created only from eligible retrieval Sections.
- Relevant properties may include:
  observed_types, validation_reason, support_method, matched_text,
  matched_pattern, raw_name, raw_type, acronym_short, acronym_definition,
  acronym_match_method, expanded_from_acronym, quality_flags,
  relationship_family, provenance, provenance_source, provenance_method,
  and doc_id.

(:Concept)-[:SAME_AS]->(:Concept)
- Optional high-confidence identity evidence produced from a shared UMLS CUI.
- Expected properties include method="umls_cui", score=1.0, status="auto",
  relationship_family="normalization", and provenance="umls_normalization".

(:Concept)-[:POSSIBLY_SAME_AS]->(:Concept)
- Optional fuzzy candidate relation with status="candidate".
- It is review evidence and must not be traversed by default retrieval.

Ontology-derived Concept-to-Concept relationships may be materialized
optionally by the UMLS-connections stage. Their exact relation types and
traversal policies are not part of this stable base contract and must not be
assumed unless an explicitly configured retrieval mode enables them.

Recommendation nodes and recommendation-specific relationships are not part
of the graph contract guaranteed by the current data-etl graph loader.
""".strip()


SECTION_RETRIEVAL_CONTRACT: Final[str] = f"""
Retrieval contract:
- Final evidence units must always be Section nodes.
- Concept nodes may identify relevant Sections but are never final evidence.
- Document nodes provide provenance and document filtering.
- Always bind the owning Document through:
  (d:Document)-[:HAS_SECTION]->(s:Section).
- Return only eligible retrieval Sections satisfying all of:
  s.section_view_role = "{RETRIEVAL_ROLE}"
  AND coalesce(s.embed, false) = true
  AND coalesce(s.excluded, false) = false
  AND trim(coalesce(s.text, "")) <> "".
- Do not require s.has_embedding = true: KG retrieval is independent of whether
  a vector embedding has already been written.
- Structural Sections may be traversed through HAS_CHILD but must not be
  returned as final evidence.
- NEXT already connects retrieval Sections only.
- Treat s.uid as an opaque identifier; do not parse or reconstruct it.
- Prefer s.retrieval_unit_id for retrieval-unit provenance when it is present.
- Preserve represented_section_ids, source_section_ids, and
  absorbed_section_ids when evaluating aggregated retrieval Sections.
- Prefer explicit matching on Concept.name.
- Concept.canonical_type filters must use the canonical type list above.
- Do not use "ambiguous" or "no_supported_type" as clinical entity types.
- UMLS properties and SAME_AS edges are optional.
- Never assume every Concept has UMLS metadata.
- POSSIBLY_SAME_AS must not be used for automatic expansion.
- Optional ontology relations must only be used by a separately configured,
  safety-filtered UMLS retrieval mode.
- If a query does not use Concept nodes, return:
  [] AS matched_concepts
  and
  1.0 AS score.
- Never omit any required result alias.

Every generated query must return exactly these required aliases:

s.uid AS section_uid
d.doc_id AS document_id
s.section_id AS section_id
s.title AS title
s.text AS text
matched_concepts
score

The returned rows must be deterministically ordered.
""".strip()


def build_text2cypher_schema(
    document_ids: Sequence[str] | None = None,
) -> str:
    if document_ids:
        normalized_ids = sorted(
            {
                str(document_id).strip()
                for document_id in document_ids
                if str(document_id).strip()
            }
        )
        scope = (
            "Active retrieval scope:\n"
            "Only retrieve Sections belonging to Documents whose doc_id is in "
            f"{json.dumps(normalized_ids, ensure_ascii=False)}."
        )
    else:
        scope = (
            "Active retrieval scope:\n"
            "All loaded Documents may be queried."
        )

    return "\n\n".join(
        (
            BASE_GRAPH_SCHEMA,
            scope,
            SECTION_RETRIEVAL_CONTRACT,
        )
    )
