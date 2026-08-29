"""Parameterized Cypher tools for Section retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from agentic_rag.kg.client import Neo4jKGClient
from agentic_rag.kg.concept_seeders import ConceptSeed
from agentic_rag.kg.models import KGSectionResult, KGRankingMode


_MAX_TOP_K = 100
_MAX_HIERARCHY_DEPTH = 8

_EXCLUDED_TITLE_PREFIXES = [
    "key messages",
    "gaps in evidence",
    "what to do",
    "references",
    "bibliography",
]

_MATCH_WEIGHTS = {
    "exact_name": 5.0,
    "exact_normalized_name": 5.0,
    "exact_umls_canonical_name": 4.0,
    "exact_umls_alias": 4.0,
    "prefix": 2.0,
    "partial": 1.0,
}

_TITLE_MATCH_BONUS = 3.0


_SEARCH_SECTIONS_BY_LOCAL_CONCEPTS = """
UNWIND $terms AS term

MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(c:Concept)

WHERE ($document_ids = [] OR d.doc_id IN $document_ids)
  AND s.section_view_role = 'retrieval'
  AND coalesce(s.embed, false) = true
  AND coalesce(s.excluded, false) = false
  AND trim(coalesce(s.text, '')) <> ''
  AND (
      NOT $exclude_summary_sections
      OR NOT any(
          excluded_title IN $excluded_title_prefixes
          WHERE toLower(trim(coalesce(s.title, ''))) STARTS WITH excluded_title
      )
  )

WITH
    d,
    s,
    c,
    term,
    toLower(trim(coalesce(c.name, ''))) AS name_lower

WITH
    d,
    s,
    c,
    term,
    CASE
        WHEN name_lower = term THEN 'exact_name'
        WHEN name_lower STARTS WITH term THEN 'prefix'
        WHEN name_lower CONTAINS term THEN 'partial'
        ELSE null
    END AS match_type

WHERE match_type IS NOT NULL

WITH
    d,
    s,
    term,
    collect(DISTINCT c.name) AS concepts_for_term,
    collect(DISTINCT {
        query_term: term,
        concept_name: c.name,
        matched_value: c.name,
        match_type: match_type,
        weight: 1.0
    }) AS diagnostics_for_term

WITH
    d,
    s,
    count(DISTINCT term) AS matched_term_count,
    collect(DISTINCT term) AS matched_terms,
    collect(concepts_for_term) AS concept_groups,
    collect(diagnostics_for_term) AS diagnostic_groups

WHERE NOT $require_all
   OR matched_term_count = size($terms)

WITH
    d,
    s,
    matched_term_count,
    matched_terms,
    reduce(
        flattened = [],
        concept_group IN concept_groups |
        flattened + concept_group
    ) AS flattened_concepts,
    reduce(
        flattened = [],
        diagnostic_group IN diagnostic_groups |
        flattened + diagnostic_group
    ) AS match_diagnostics

WITH
    d,
    s,
    matched_term_count,
    matched_terms,
    reduce(
        unique_concepts = [],
        concept_name IN flattened_concepts |
        CASE
            WHEN concept_name IS NULL OR concept_name IN unique_concepts
                THEN unique_concepts
            ELSE unique_concepts + concept_name
        END
    ) AS matched_concepts,
    match_diagnostics

WITH
    d,
    s,
    matched_concepts,
    matched_terms,
    match_diagnostics,
    toFloat(matched_term_count) AS concept_match_score

RETURN
    s.uid AS section_uid,
    d.doc_id AS document_id,
    s.section_id AS section_id,
    s.printed_section_id AS printed_section_id,
    s.title AS title,
    s.level AS level,
    s.text AS text,
    s.page_start AS page_start,
    s.page_end AS page_end,
    s.part_index AS part_index,
    s.part_count AS part_count,
    s.retrieval_unit_id AS retrieval_unit_id,
    s.section_view_schema_version AS section_view_schema_version,
    s.section_view_role AS section_view_role,
    s.retrieval_strategy AS retrieval_strategy,
    s.aggregation_mode AS aggregation_mode,
    coalesce(s.is_aggregated, false) AS is_aggregated,
    s.content_owner_section_id AS content_owner_section_id,
    coalesce(s.source_section_ids, []) AS source_section_ids,
    coalesce(s.source_chunk_ids, []) AS source_chunk_ids,
    coalesce(s.represented_section_ids, []) AS represented_section_ids,
    coalesce(
        s.structural_context_section_ids,
        []
    ) AS structural_context_section_ids,
    coalesce(s.absorbed_section_ids, []) AS absorbed_section_ids,
    coalesce(
        s.absorbed_source_section_ids,
        []
    ) AS absorbed_source_section_ids,
    matched_concepts,
    concept_match_score AS score,
    matched_terms,
    'concept_match' AS score_type,
    {
        concept_match: concept_match_score,
        weighted_match: concept_match_score
    } AS scores,
    match_diagnostics

ORDER BY
    score DESC,
    section_uid ASC

LIMIT $top_k
"""


_SEARCH_SECTIONS_BY_CONCEPTS = """
UNWIND $terms AS term

MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(c:Concept)

WHERE ($document_ids = [] OR d.doc_id IN $document_ids)
  AND s.section_view_role = 'retrieval'
  AND coalesce(s.embed, false) = true
  AND coalesce(s.excluded, false) = false
  AND trim(coalesce(s.text, '')) <> ''
  AND (
      NOT $exclude_summary_sections
      OR NOT any(
          excluded_title IN $excluded_title_prefixes
          WHERE toLower(trim(coalesce(s.title, ''))) STARTS WITH excluded_title
      )
  )

// KG_BASELINE_MATCHING_LOCAL_ONLY
WITH
    d,
    s,
    c,
    term,
    toLower(coalesce(c.name, '')) AS name_lower

WITH
    d,
    s,
    c,
    term,
    CASE
        WHEN name_lower = term THEN 'exact_name'
        WHEN name_lower STARTS WITH term THEN 'prefix'
        WHEN name_lower CONTAINS term THEN 'partial'
        ELSE null
    END AS match_type,
    CASE
        WHEN name_lower = term
          OR name_lower STARTS WITH term
          OR name_lower CONTAINS term
            THEN coalesce(c.name, '')
        ELSE null
    END AS matched_value

WITH
    d,
    s,
    c,
    term,
    match_type,
    matched_value,
    CASE match_type
        WHEN 'exact_name' THEN $exact_name_weight
        WHEN 'prefix' THEN $prefix_weight
        WHEN 'partial' THEN $partial_weight
        ELSE 0.0
    END AS match_weight

WHERE match_type IS NOT NULL

WITH
    d,
    s,
    term,
    max(match_weight) AS best_weight_for_term,
    collect(DISTINCT c.name) AS concepts_for_term,
    collect(DISTINCT {
        query_term: term,
        concept_name: c.name,
        matched_value: matched_value,
        match_type: match_type,
        weight: toFloat(match_weight)
    }) AS diagnostics_for_term

WITH
    d,
    s,
    count(DISTINCT term) AS matched_term_count,
    sum(best_weight_for_term) AS concept_weight_sum,
    collect(DISTINCT term) AS matched_terms,
    collect(concepts_for_term) AS concept_groups,
    collect(diagnostics_for_term) AS diagnostic_groups

WHERE NOT $require_all
   OR matched_term_count = size($terms)

WITH
    d,
    s,
    matched_term_count,
    concept_weight_sum,
    matched_terms,
    reduce(
        flattened = [],
        concept_group IN concept_groups |
        flattened + concept_group
    ) AS flattened_concepts,
    reduce(
        flattened = [],
        diagnostic_group IN diagnostic_groups |
        flattened + diagnostic_group
    ) AS match_diagnostics

WITH
    d,
    s,
    matched_term_count,
    concept_weight_sum,
    matched_terms,
    reduce(
        unique_concepts = [],
        concept_name IN flattened_concepts |
        CASE
            WHEN concept_name IS NULL OR concept_name IN unique_concepts
                THEN unique_concepts
            ELSE unique_concepts + concept_name
        END
    ) AS matched_concepts,
    match_diagnostics,
    size([
        matched_term IN matched_terms
        WHERE toLower(coalesce(s.title, '')) CONTAINS matched_term
    ]) AS title_match_count

WITH
    d,
    s,
    matched_concepts,
    matched_terms,
    match_diagnostics,
    toFloat(matched_term_count) AS concept_match_score,
    toFloat(
        concept_weight_sum + ($title_match_bonus * title_match_count)
    ) AS weighted_match_score

WITH
    d,
    s,
    matched_concepts,
    matched_terms,
    match_diagnostics,
    concept_match_score,
    weighted_match_score,
    CASE
        WHEN $ranking_mode = 'weighted_match'
            THEN weighted_match_score
        ELSE concept_match_score
    END AS active_score

RETURN
    s.uid AS section_uid,
    d.doc_id AS document_id,
    s.section_id AS section_id,
    s.printed_section_id AS printed_section_id,
    s.title AS title,
    s.level AS level,
    s.text AS text,
    s.page_start AS page_start,
    s.page_end AS page_end,
    s.part_index AS part_index,
    s.part_count AS part_count,
    s.retrieval_unit_id AS retrieval_unit_id,
    s.section_view_schema_version AS section_view_schema_version,
    s.section_view_role AS section_view_role,
    s.retrieval_strategy AS retrieval_strategy,
    s.aggregation_mode AS aggregation_mode,
    coalesce(s.is_aggregated, false) AS is_aggregated,
    s.content_owner_section_id AS content_owner_section_id,
    coalesce(s.source_section_ids, []) AS source_section_ids,
    coalesce(s.source_chunk_ids, []) AS source_chunk_ids,
    coalesce(s.represented_section_ids, []) AS represented_section_ids,
    coalesce(
        s.structural_context_section_ids,
        []
    ) AS structural_context_section_ids,
    coalesce(s.absorbed_section_ids, []) AS absorbed_section_ids,
    coalesce(
        s.absorbed_source_section_ids,
        []
    ) AS absorbed_source_section_ids,
    matched_concepts,
    active_score AS score,
    matched_terms,
    $ranking_mode AS score_type,
    {
        concept_match: concept_match_score,
        weighted_match: weighted_match_score
    } AS scores,
    match_diagnostics

ORDER BY
    score DESC,
    CASE
        WHEN $ranking_mode = 'weighted_match'
            THEN concept_match_score
        ELSE 0.0
    END DESC,
    section_uid ASC

LIMIT $top_k
"""


_SEARCH_SECTIONS_BY_TITLE = """
UNWIND $terms AS term

MATCH (d:Document)-[:HAS_SECTION]->(s:Section)

WHERE ($document_ids = [] OR d.doc_id IN $document_ids)
  AND s.section_view_role = 'retrieval'
  AND coalesce(s.embed, false) = true
  AND coalesce(s.excluded, false) = false
  AND trim(coalesce(s.text, '')) <> ''
  AND (
      NOT $exclude_summary_sections
      OR NOT any(
          excluded_title IN $excluded_title_prefixes
          WHERE toLower(trim(coalesce(s.title, ''))) STARTS WITH excluded_title
      )
  )

WITH
    d,
    s,
    term,
    toLower(trim(coalesce(s.title, ''))) AS title_lower

WITH
    d,
    s,
    term,
    CASE
        WHEN title_lower = term THEN 5.0
        WHEN title_lower STARTS WITH term THEN 3.0
        WHEN title_lower CONTAINS term THEN 1.0
        ELSE 0.0
    END AS title_weight,
    CASE
        WHEN title_lower = term THEN 'exact_title'
        WHEN title_lower STARTS WITH term THEN 'title_prefix'
        WHEN title_lower CONTAINS term THEN 'title_contains'
        ELSE null
    END AS match_type

WHERE match_type IS NOT NULL

WITH
    d,
    s,
    count(DISTINCT term) AS matched_term_count,
    sum(title_weight) AS weighted_match_score,
    collect(DISTINCT term) AS matched_terms,
    collect(DISTINCT {
        query_term: term,
        concept_name: null,
        matched_value: s.title,
        match_type: match_type,
        weight: toFloat(title_weight)
    }) AS match_diagnostics

WHERE NOT $require_all
   OR matched_term_count = size($terms)

WITH
    d,
    s,
    matched_terms,
    match_diagnostics,
    toFloat(matched_term_count) AS concept_match_score,
    toFloat(weighted_match_score) AS weighted_match_score

WITH
    d,
    s,
    matched_terms,
    match_diagnostics,
    concept_match_score,
    weighted_match_score,
    CASE
        WHEN $ranking_mode = 'weighted_match'
            THEN weighted_match_score
        ELSE concept_match_score
    END AS active_score

RETURN
    s.uid AS section_uid,
    d.doc_id AS document_id,
    s.section_id AS section_id,
    s.printed_section_id AS printed_section_id,
    s.title AS title,
    s.level AS level,
    s.text AS text,
    s.page_start AS page_start,
    s.page_end AS page_end,
    s.part_index AS part_index,
    s.part_count AS part_count,
    s.retrieval_unit_id AS retrieval_unit_id,
    s.section_view_schema_version AS section_view_schema_version,
    s.section_view_role AS section_view_role,
    s.retrieval_strategy AS retrieval_strategy,
    s.aggregation_mode AS aggregation_mode,
    coalesce(s.is_aggregated, false) AS is_aggregated,
    s.content_owner_section_id AS content_owner_section_id,
    coalesce(s.source_section_ids, []) AS source_section_ids,
    coalesce(s.source_chunk_ids, []) AS source_chunk_ids,
    coalesce(s.represented_section_ids, []) AS represented_section_ids,
    coalesce(
        s.structural_context_section_ids,
        []
    ) AS structural_context_section_ids,
    coalesce(s.absorbed_section_ids, []) AS absorbed_section_ids,
    coalesce(
        s.absorbed_source_section_ids,
        []
    ) AS absorbed_source_section_ids,
    [] AS matched_concepts,
    active_score AS score,
    matched_terms,
    $ranking_mode AS score_type,
    {
        concept_match: concept_match_score,
        weighted_match: weighted_match_score
    } AS scores,
    match_diagnostics

ORDER BY
    score DESC,
    weighted_match_score DESC,
    concept_match_score DESC,
    section_uid ASC

LIMIT $top_k
"""

_LIST_CONCEPT_CATALOGUE = """
MATCH (c:Concept)
WHERE trim(coalesce(c.name, '')) <> ''
  AND EXISTS {
      MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(c)
      WHERE ($document_ids = [] OR d.doc_id IN $document_ids)
        AND s.section_view_role = 'retrieval'
        AND coalesce(s.embed, false) = true
        AND coalesce(s.excluded, false) = false
        AND trim(coalesce(s.text, '')) <> ''
  }

RETURN DISTINCT
    c.name AS concept_name,
    c.name AS name,
    c.normalized_name AS normalized_name,
    c.umls_canonical_name AS umls_canonical_name,
    coalesce(c.umls_aliases, []) AS umls_aliases,
    c.canonical_type AS canonical_type,
    c.umls_cui AS umls_cui

ORDER BY concept_name ASC
"""

_SEARCH_SECTIONS_BY_CONCEPT_SEEDS = """
UNWIND $seeds AS seed

MATCH (c:Concept {name: seed.concept_name})
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(c)

WHERE ($document_ids = [] OR d.doc_id IN $document_ids)
  AND s.section_view_role = 'retrieval'
  AND coalesce(s.embed, false) = true
  AND coalesce(s.excluded, false) = false
  AND trim(coalesce(s.text, '')) <> ''
  AND (
      NOT $exclude_summary_sections
      OR NOT any(
          excluded_title IN $excluded_title_prefixes
          WHERE toLower(trim(coalesce(s.title, ''))) STARTS WITH excluded_title
      )
  )

WITH DISTINCT
    d,
    s,
    c,
    seed,
    coalesce(toString(seed.query_term), '') AS query_term,
    coalesce(toString(seed.query_term_key), '') AS query_term_key,
    toInteger(seed.seed_rank) AS seed_rank,
    coalesce(toString(seed.method), '') AS seeding_method,
    coalesce(toString(seed.match_type), '') AS match_type,
    seed.matched_value AS matched_value,
    seed.similarity AS similarity,
    seed.umls_cui AS umls_cui

ORDER BY
    s.uid ASC,
    query_term_key ASC,
    seed_rank ASC,
    c.name ASC

WITH
    d,
    s,
    query_term_key,
    head(collect(query_term)) AS query_term,
    min(seed_rank) AS best_seed_rank_for_term,
    collect(DISTINCT c.name) AS concepts_for_term,
    collect(DISTINCT {
        query_term: query_term,
        concept_name: c.name,
        matched_value: matched_value,
        match_type: match_type,
        weight: 1.0,
        seed_rank: seed_rank,
        seeding_method: seeding_method,
        similarity: CASE
            WHEN similarity IS NULL THEN null
            ELSE toFloat(similarity)
        END,
        umls_cui: umls_cui
    }) AS diagnostics_for_term

WITH
    d,
    s,
    count(DISTINCT query_term_key) AS matched_term_count,
    collect(DISTINCT query_term_key) AS matched_term_keys,
    collect(best_seed_rank_for_term) AS best_seed_ranks,
    collect(query_term) AS matched_terms,
    collect(concepts_for_term) AS concept_groups,
    collect(diagnostics_for_term) AS diagnostic_groups

WHERE NOT $require_all
   OR all(term_key IN $term_keys WHERE term_key IN matched_term_keys)

WITH
    d,
    s,
    matched_term_count,
    reduce(
        total = 0,
        seed_rank IN best_seed_ranks |
        total + seed_rank
    ) AS seed_rank_sum,
    reduce(
        best = 2147483647,
        seed_rank IN best_seed_ranks |
        CASE
            WHEN seed_rank < best THEN seed_rank
            ELSE best
        END
    ) AS best_seed_rank,
    matched_terms,
    reduce(
        flattened = [],
        concept_group IN concept_groups |
        flattened + concept_group
    ) AS flattened_concepts,
    reduce(
        flattened = [],
        diagnostic_group IN diagnostic_groups |
        flattened + diagnostic_group
    ) AS match_diagnostics

WITH
    d,
    s,
    matched_term_count,
    seed_rank_sum,
    best_seed_rank,
    reduce(
        unique_terms = [],
        term IN matched_terms |
        CASE
            WHEN term IS NULL OR term IN unique_terms
                THEN unique_terms
            ELSE unique_terms + term
        END
    ) AS matched_terms,
    reduce(
        unique_concepts = [],
        concept_name IN flattened_concepts |
        CASE
            WHEN concept_name IS NULL OR concept_name IN unique_concepts
                THEN unique_concepts
            ELSE unique_concepts + concept_name
        END
    ) AS matched_concepts,
    match_diagnostics

ORDER BY
    matched_term_count DESC,
    seed_rank_sum ASC,
    best_seed_rank ASC,
    s.uid ASC

LIMIT $top_k

RETURN
    s.uid AS section_uid,
    d.doc_id AS document_id,
    s.section_id AS section_id,
    s.printed_section_id AS printed_section_id,
    s.title AS title,
    s.level AS level,
    s.text AS text,
    s.page_start AS page_start,
    s.page_end AS page_end,
    s.part_index AS part_index,
    s.part_count AS part_count,
    s.retrieval_unit_id AS retrieval_unit_id,
    s.section_view_schema_version AS section_view_schema_version,
    s.section_view_role AS section_view_role,
    s.retrieval_strategy AS retrieval_strategy,
    s.aggregation_mode AS aggregation_mode,
    coalesce(s.is_aggregated, false) AS is_aggregated,
    s.content_owner_section_id AS content_owner_section_id,
    coalesce(s.source_section_ids, []) AS source_section_ids,
    coalesce(s.source_chunk_ids, []) AS source_chunk_ids,
    coalesce(s.represented_section_ids, []) AS represented_section_ids,
    coalesce(
        s.structural_context_section_ids,
        []
    ) AS structural_context_section_ids,
    coalesce(s.absorbed_section_ids, []) AS absorbed_section_ids,
    coalesce(
        s.absorbed_source_section_ids,
        []
    ) AS absorbed_source_section_ids,
    matched_concepts,
    toFloat(matched_term_count) AS score,
    matched_terms,
    'concept_match' AS score_type,
    {
        concept_match: toFloat(matched_term_count),
        weighted_match: toFloat(matched_term_count)
    } AS scores,
    match_diagnostics
"""


_FIND_HIERARCHICAL_CONTEXT_MATCHES = """
UNWIND $anchor_uids AS anchor_uid
MATCH (anchor:Section {uid: anchor_uid})
WHERE anchor.section_view_role = 'retrieval'
  AND coalesce(anchor.embed, false) = true
  AND coalesce(anchor.excluded, false) = false
  AND trim(coalesce(anchor.text, '')) <> ''

UNWIND $context_uids AS context_uid
MATCH (context:Section {uid: context_uid})
WHERE context.section_view_role = 'retrieval'
  AND coalesce(context.embed, false) = true
  AND coalesce(context.excluded, false) = false
  AND trim(coalesce(context.text, '')) <> ''

MATCH path = (context)-[:HAS_CHILD*0..8]->(anchor)
WHERE length(path) <= $max_depth
  AND context.doc_id = anchor.doc_id

RETURN
    anchor.uid AS anchor_uid,
    anchor.doc_id AS anchor_document_id,
    anchor.section_id AS anchor_section_id,
    anchor.printed_section_id AS anchor_printed_section_id,
    anchor.title AS anchor_title,
    context.uid AS context_uid,
    context.doc_id AS context_document_id,
    context.section_id AS context_section_id,
    context.printed_section_id AS context_printed_section_id,
    context.title AS context_title,
    length(path) AS hierarchy_distance

ORDER BY
    anchor_uid ASC,
    hierarchy_distance ASC,
    context_uid ASC
"""


class KGSectionTools:
    """Controlled graph-retrieval tools returning Section results."""

    def __init__(self, client: Neo4jKGClient) -> None:
        self.client = client

    def search_sections_by_concepts(
        self,
        concepts: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
        top_k: int = 10,
        require_all: bool = False,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
        local_only: bool = False,
    ) -> list[KGSectionResult]:
        """
        Retrieve Sections mentioning one or more clinical concepts.

        ``local_only=True`` is the pure MENTIONS baseline: query terms are
        matched only against local ``Concept.name`` values, without UMLS
        aliases/canonical names, ``normalized_name``, title bonuses, or
        weighted tie-breaking. The exact Section text stored in Neo4j is
        returned unchanged.

        ``local_only=False`` preserves the legacy enriched matcher.
        """

        terms = _normalize_values(
            concepts,
            field_name="concepts",
            required=True,
        )
        normalized_document_ids = _normalize_values(
            document_ids,
            field_name="document_ids",
            required=False,
        )
        validated_top_k = _validate_top_k(top_k)
        validated_ranking_mode = _validate_ranking_mode(ranking_mode)

        query = (
            _SEARCH_SECTIONS_BY_LOCAL_CONCEPTS
            if local_only
            else _SEARCH_SECTIONS_BY_CONCEPTS
        )

        parameters: dict[str, Any] = {
            "terms": [term.casefold() for term in terms],
            "document_ids": normalized_document_ids,
            "require_all": bool(require_all),
            "top_k": validated_top_k,
            "ranking_mode": validated_ranking_mode,
            "exclude_summary_sections": bool(exclude_summary_sections),
            "excluded_title_prefixes": _EXCLUDED_TITLE_PREFIXES,
        }
        if not local_only:
            parameters.update(
                {
                    "exact_name_weight": _MATCH_WEIGHTS["exact_name"],
                    "exact_normalized_name_weight": _MATCH_WEIGHTS[
                        "exact_normalized_name"
                    ],
                    "exact_umls_name_weight": _MATCH_WEIGHTS[
                        "exact_umls_canonical_name"
                    ],
                    "exact_umls_alias_weight": _MATCH_WEIGHTS[
                        "exact_umls_alias"
                    ],
                    "prefix_weight": _MATCH_WEIGHTS["prefix"],
                    "partial_weight": _MATCH_WEIGHTS["partial"],
                    "title_match_bonus": _TITLE_MATCH_BONUS,
                }
            )

        rows = self.client.run_read(query, parameters)

        return _rows_to_results(rows)

    def search_sections_by_title(
        self,
        title_terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
        top_k: int = 10,
        require_all: bool = False,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> list[KGSectionResult]:
        """
        Retrieve Sections whose titles contain the supplied terms.

        Both ranking modes are exposed through the same result contract. The
        exact Section text stored in Neo4j is returned unchanged.
        """

        terms = _normalize_values(
            title_terms,
            field_name="title_terms",
            required=True,
        )
        normalized_document_ids = _normalize_values(
            document_ids,
            field_name="document_ids",
            required=False,
        )
        validated_top_k = _validate_top_k(top_k)
        validated_ranking_mode = _validate_ranking_mode(ranking_mode)

        rows = self.client.run_read(
            _SEARCH_SECTIONS_BY_TITLE,
            {
                "terms": [term.casefold() for term in terms],
                "document_ids": normalized_document_ids,
                "require_all": bool(require_all),
                "top_k": validated_top_k,
                "ranking_mode": validated_ranking_mode,
                "exclude_summary_sections": bool(
                    exclude_summary_sections
                ),
                "excluded_title_prefixes": _EXCLUDED_TITLE_PREFIXES,
            },
        )

        return _rows_to_results(rows)

    def list_concept_catalogue(
        self,
        *,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[dict[str, Any]]:
        """List distinct local Concepts available for deterministic seeding."""

        normalized_document_ids = _normalize_values(
            document_ids,
            field_name="document_ids",
            required=False,
        )
        return self.client.run_read(
            _LIST_CONCEPT_CATALOGUE,
            {
                "document_ids": normalized_document_ids,
            },
        )

    def search_sections_by_concept_seeds(
        self,
        seeds: Sequence[ConceptSeed],
        *,
        query_terms: Sequence[str] | str | None = None,
        document_ids: Sequence[str] | str | None = None,
        top_k: int = 10,
        require_all: bool = False,
        exclude_summary_sections: bool = True,
    ) -> list[KGSectionResult]:
        """Retrieve Sections from explicit Concept.name seeds only."""

        normalized_seeds = _normalize_concept_seeds(seeds)
        if not normalized_seeds:
            return []

        normalized_query_terms = (
            _normalize_values(
                query_terms,
                field_name="query_terms",
                required=True,
            )
            if query_terms is not None
            else _normalize_values(
                [seed.query_term for seed in normalized_seeds],
                field_name="query_terms",
                required=True,
            )
        )
        normalized_document_ids = _normalize_values(
            document_ids,
            field_name="document_ids",
            required=False,
        )
        validated_top_k = _validate_top_k(top_k)

        rows = self.client.run_read(
            _SEARCH_SECTIONS_BY_CONCEPT_SEEDS,
            {
                "seeds": [
                    _concept_seed_payload(seed)
                    for seed in normalized_seeds
                ],
                "term_keys": [
                    term.casefold() for term in normalized_query_terms
                ],
                "document_ids": normalized_document_ids,
                "require_all": bool(require_all),
                "top_k": validated_top_k,
                "exclude_summary_sections": bool(
                    exclude_summary_sections
                ),
                "excluded_title_prefixes": _EXCLUDED_TITLE_PREFIXES,
            },
        )

        return _rows_to_results(rows)

    def find_hierarchical_context_matches(
        self,
        anchor_uids: Sequence[str] | str,
        context_uids: Sequence[str] | str,
        *,
        max_depth: int = 6,
    ) -> list[dict[str, Any]]:
        """Find context Sections that are ancestors of anchor Sections.

        A zero-length path is allowed, so the same Section can provide both
        anchor and context evidence. The method is intentionally structural:
        it uses only existing HAS_CHILD relationships and does not infer new
        clinical relations.
        """

        normalized_anchor_uids = _normalize_values(
            anchor_uids,
            field_name="anchor_uids",
            required=True,
        )
        normalized_context_uids = _normalize_values(
            context_uids,
            field_name="context_uids",
            required=True,
        )
        validated_max_depth = _validate_hierarchy_depth(max_depth)

        return self.client.run_read(
            _FIND_HIERARCHICAL_CONTEXT_MATCHES,
            {
                "anchor_uids": normalized_anchor_uids,
                "context_uids": normalized_context_uids,
                "max_depth": validated_max_depth,
            },
        )


def _normalize_values(
    values: Sequence[str] | str | None,
    *,
    field_name: str,
    required: bool,
) -> list[str]:
    if values is None:
        raw_values: list[Any] = []
    elif isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)

    normalized: list[str] = []
    seen: set[str] = set()

    for value in raw_values:
        if value is None:
            continue

        item = str(value).strip()
        if not item:
            continue

        key = item.casefold()
        if key in seen:
            continue

        seen.add(key)
        normalized.append(item)

    if required and not normalized:
        raise ValueError(
            f"{field_name} must contain at least one non-empty value"
        )

    return normalized


def _normalize_concept_seeds(
    seeds: Sequence[ConceptSeed],
) -> list[ConceptSeed]:
    normalized: list[ConceptSeed] = []
    seen: set[tuple[str, str]] = set()

    for seed in seeds:
        item = (
            seed
            if isinstance(seed, ConceptSeed)
            else ConceptSeed.model_validate(seed)
        )
        key = (item.query_term.casefold(), item.concept_name)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)

    return normalized


def _concept_seed_payload(seed: ConceptSeed) -> dict[str, Any]:
    payload = seed.model_dump(mode="json")
    payload["query_term_key"] = seed.query_term.casefold()
    return payload



def _validate_hierarchy_depth(max_depth: int) -> int:
    try:
        normalized = int(max_depth)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_depth must be an integer") from exc

    if normalized < 0:
        raise ValueError("max_depth must be at least 0")

    if normalized > _MAX_HIERARCHY_DEPTH:
        raise ValueError(
            f"max_depth must not exceed {_MAX_HIERARCHY_DEPTH}"
        )

    return normalized


def _validate_top_k(top_k: int) -> int:
    try:
        normalized = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc

    if normalized < 1:
        raise ValueError("top_k must be at least 1")

    if normalized > _MAX_TOP_K:
        raise ValueError(
            f"top_k must not exceed {_MAX_TOP_K}"
        )

    return normalized


def _validate_ranking_mode(
    ranking_mode: str,
) -> Literal["concept_match", "weighted_match"]:
    normalized = str(ranking_mode).strip().lower()

    if normalized not in {"concept_match", "weighted_match"}:
        raise ValueError(
            "ranking_mode must be 'concept_match' or 'weighted_match'"
        )

    return normalized  # type: ignore[return-value]


def _rows_to_results(
    rows: Sequence[dict[str, Any]],
) -> list[KGSectionResult]:
    """
    Validate Neo4j records and deduplicate identical Section nodes.

    Deduplication is performed only on section_uid. Parent and child Sections
    remain distinct retrieval units. The Section text is never truncated or
    normalized.
    """

    results: list[KGSectionResult] = []
    seen_section_uids: set[str] = set()

    for row in rows:
        result = KGSectionResult.from_record(row)

        if result.section_uid in seen_section_uids:
            continue

        seen_section_uids.add(result.section_uid)

        results.append(
            result.model_copy(
                update={"rank": len(results) + 1}
            )
        )

    return results
