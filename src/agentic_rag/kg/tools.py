"""Parameterized Cypher tools for Section retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from agentic_rag.kg.client import Neo4jKGClient
from agentic_rag.kg.models import KGSectionResult, KGRankingMode


_MAX_TOP_K = 100

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


_SEARCH_SECTIONS_BY_CONCEPTS = """
UNWIND $terms AS term

MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(c:Concept)

WHERE ($document_ids = [] OR d.doc_id IN $document_ids)
  AND coalesce(s.embed, false) = true
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
    toLower(coalesce(c.name, '')) AS name_lower,
    toLower(coalesce(c.normalized_name, '')) AS normalized_name_lower,
    toLower(coalesce(c.umls_canonical_name, '')) AS umls_name_lower,
    [alias IN coalesce(c.umls_aliases, []) | toString(alias)] AS aliases

WITH
    d,
    s,
    c,
    term,
    aliases,
    name_lower,
    normalized_name_lower,
    umls_name_lower,
    [alias IN aliases | toLower(alias)] AS aliases_lower

WITH
    d,
    s,
    c,
    term,
    CASE
        WHEN name_lower = term
            THEN 'exact_name'
        WHEN normalized_name_lower = term
            THEN 'exact_normalized_name'
        WHEN umls_name_lower = term
            THEN 'exact_umls_canonical_name'
        WHEN any(alias IN aliases_lower WHERE alias = term)
            THEN 'exact_umls_alias'
        WHEN name_lower STARTS WITH term
          OR normalized_name_lower STARTS WITH term
          OR umls_name_lower STARTS WITH term
          OR any(alias IN aliases_lower WHERE alias STARTS WITH term)
            THEN 'prefix'
        WHEN name_lower CONTAINS term
          OR normalized_name_lower CONTAINS term
          OR umls_name_lower CONTAINS term
          OR any(alias IN aliases_lower WHERE alias CONTAINS term)
            THEN 'partial'
        ELSE null
    END AS match_type,
    CASE
        WHEN name_lower = term
            THEN coalesce(c.name, '')
        WHEN normalized_name_lower = term
            THEN coalesce(c.normalized_name, '')
        WHEN umls_name_lower = term
            THEN coalesce(c.umls_canonical_name, '')
        WHEN any(alias IN aliases_lower WHERE alias = term)
            THEN head([alias IN aliases WHERE toLower(alias) = term])
        WHEN name_lower STARTS WITH term
            THEN coalesce(c.name, '')
        WHEN normalized_name_lower STARTS WITH term
            THEN coalesce(c.normalized_name, '')
        WHEN umls_name_lower STARTS WITH term
            THEN coalesce(c.umls_canonical_name, '')
        WHEN any(alias IN aliases_lower WHERE alias STARTS WITH term)
            THEN head([
                alias IN aliases
                WHERE toLower(alias) STARTS WITH term
            ])
        WHEN name_lower CONTAINS term
            THEN coalesce(c.name, '')
        WHEN normalized_name_lower CONTAINS term
            THEN coalesce(c.normalized_name, '')
        WHEN umls_name_lower CONTAINS term
            THEN coalesce(c.umls_canonical_name, '')
        WHEN any(alias IN aliases_lower WHERE alias CONTAINS term)
            THEN head([
                alias IN aliases
                WHERE toLower(alias) CONTAINS term
            ])
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
        WHEN 'exact_normalized_name' THEN $exact_normalized_name_weight
        WHEN 'exact_umls_canonical_name' THEN $exact_umls_name_weight
        WHEN 'exact_umls_alias' THEN $exact_umls_alias_weight
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
    weighted_match_score DESC,
    concept_match_score DESC,
    section_uid ASC

LIMIT $top_k
"""


_SEARCH_SECTIONS_BY_TITLE = """
UNWIND $terms AS term

MATCH (d:Document)-[:HAS_SECTION]->(s:Section)

WHERE ($document_ids = [] OR d.doc_id IN $document_ids)
  AND coalesce(s.embed, false) = true
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
    ) -> list[KGSectionResult]:
        """
        Retrieve Sections mentioning one or more clinical concepts.

        Both the unweighted concept-match score and the weighted diagnostic
        score are returned. ``score`` and row ordering use ``ranking_mode``.
        The exact Section text stored in Neo4j is returned unchanged.
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

        rows = self.client.run_read(
            _SEARCH_SECTIONS_BY_CONCEPTS,
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
            },
        )

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
