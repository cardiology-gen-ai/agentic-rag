from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Final


SCHEMA_VERSION: Final[str] = "1.0"

REQUIRED_RESULT_ALIASES: Final[tuple[str, ...]] = (
    "section_uid",
    "document_id",
    "section_id",
    "title",
    "text",
    "matched_concepts",
    "score",
)

# When we change schema, we should update this
BASE_GRAPH_SCHEMA: Final[str] = """
Graph schema for medical guideline retrieval.

Node labels and relevant properties:

Document
- doc_id: STRING, unique document identifier.

Section
- uid: STRING, unique identifier formatted as "<doc_id>::<section_id>".
- doc_id: STRING.
- section_id: STRING.
- printed_section_id: STRING or null.
- title: STRING or null.
- level: INTEGER or null.
- text: STRING or null.
- is_empty: BOOLEAN or null.
- embed: BOOLEAN.
- page_start: INTEGER or null.
- page_end: INTEGER or null.
- part_index: INTEGER or null.
- part_count: INTEGER or null.
- entity_extracted: BOOLEAN or null.
- entity_extraction_status: STRING or null.

Concept
- name: STRING, unique normalized lowercase concept name.
- canonical_type: STRING or null.
- observed_types: LIST<STRING> or null.
- type_resolution_status: STRING or null.
- needs_type_review: BOOLEAN or null.
- normalized_name: STRING or null.
- normalization_status: STRING or null.
- umls_cui: STRING or null.
- umls_canonical_name: STRING or null.
- umls_aliases: LIST<STRING> or null.
- umls_semantic_types: LIST<STRING> or null.
- umls_score: FLOAT or null.

Supported Concept canonical types include:
- disease
- clinical_finding
- risk_factor
- genetic_factor
- biomarker
- diagnostic_test
- imaging_modality
- score_or_risk_model
- drug_or_drug_class
- procedure_or_intervention
- device
- complication_or_comorbidity
- care_strategy
- anatomical_structure

Relationships:

(Document)-[:HAS_SECTION]->(Section)
- A Document is linked to all of its Section nodes.

(Section)-[:HAS_CHILD]->(Section)
- Represents the hierarchical parent-child structure of sections.

(Section)-[:NEXT]->(Section)
- Represents document reading order between consecutive sections.

(Section)-[:MENTIONS]->(Concept)
Relevant relationship properties may include:
- observed_types
- validation_reason
- support_method
- matched_text
- matched_pattern
- raw_name
- raw_type
- acronym_short
- acronym_definition
- expanded_from_acronym

(Concept)-[:SAME_AS]->(Concept)
- High-confidence normalization or identity relation.

(Concept)-[:POSSIBLY_SAME_AS]->(Concept)
- Lower-confidence candidate identity relation.
- May have score, method, and status properties.
""".strip()


SECTION_RETRIEVAL_CONTRACT: Final[str] = """
Retrieval contract:

- The final evidence units must always be Section nodes.
- Concept nodes may be used to identify relevant Sections.
- Document nodes may be used for provenance and document filtering.
- Only return Sections where coalesce(s.embed, false) = true.
- Only return Sections with non-empty text.
- Prefer explicit matches on Concept.name.
- UMLS properties are optional and must only be used with safe fallbacks.
- Never assume every Concept has UMLS metadata.
- Never return Concept nodes alone as final evidence.

Every generated query must return exactly these aliases:

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
            f"Only retrieve Sections belonging to Documents whose doc_id is in "
            f"{json.dumps(normalized_ids, ensure_ascii=False)}."
        )
    else:
        scope = "Active retrieval scope:\nAll loaded Documents may be queried."

    return "\n\n".join(
        (
            BASE_GRAPH_SCHEMA,
            scope,
            SECTION_RETRIEVAL_CONTRACT,
        )
    )