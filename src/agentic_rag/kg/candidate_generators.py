"""Modular candidate generators for knowledge-graph retrieval.

Candidate generation is deliberately separated from expansion and reranking.
This makes simple MENTIONS-only retrieval directly measurable and allows more
advanced graph strategies to be added as controlled ablations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_rag.kg.models import KGSectionResult, KGRankingMode


KGCandidateSource = Literal[
    "mentions",
    "title",
    "descendant",
    "same_as",
    "umls_neighbor",
]

_EXCLUDED_TITLE_PREFIXES = [
    "key messages",
    "gaps in evidence",
    "what to do",
    "references",
    "bibliography",
]

_CONCEPT_GRAPH_DIRECT_WEIGHT = 1.0
_CONCEPT_GRAPH_SAME_AS_WEIGHT = 0.9
_CONCEPT_GRAPH_UMLS_WEIGHT = 0.5

_CONCEPT_GRAPH_EXACT_WEIGHT = 3.0
_CONCEPT_GRAPH_PREFIX_WEIGHT = 2.0
_CONCEPT_GRAPH_PARTIAL_WEIGHT = 1.0

_CONCEPT_GRAPH_SEED_MATCH = """
UNWIND $terms AS term

MATCH (seed:Concept)

WITH
    term,
    seed,
    toLower(coalesce(seed.name, '')) AS name_lower,
    toLower(coalesce(seed.normalized_name, '')) AS normalized_name_lower,
    toLower(coalesce(seed.umls_canonical_name, '')) AS umls_name_lower,
    [alias IN coalesce(seed.umls_aliases, []) | toString(alias)] AS aliases

WITH
    term,
    seed,
    aliases,
    name_lower,
    normalized_name_lower,
    umls_name_lower,
    [alias IN aliases | toLower(alias)] AS aliases_lower

WITH
    term,
    seed,
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
            THEN coalesce(seed.name, '')
        WHEN normalized_name_lower = term
            THEN coalesce(seed.normalized_name, '')
        WHEN umls_name_lower = term
            THEN coalesce(seed.umls_canonical_name, '')
        WHEN any(alias IN aliases_lower WHERE alias = term)
            THEN head([alias IN aliases WHERE toLower(alias) = term])
        WHEN name_lower STARTS WITH term
            THEN coalesce(seed.name, '')
        WHEN normalized_name_lower STARTS WITH term
            THEN coalesce(seed.normalized_name, '')
        WHEN umls_name_lower STARTS WITH term
            THEN coalesce(seed.umls_canonical_name, '')
        WHEN any(alias IN aliases_lower WHERE alias STARTS WITH term)
            THEN head([
                alias IN aliases
                WHERE toLower(alias) STARTS WITH term
            ])
        WHEN name_lower CONTAINS term
            THEN coalesce(seed.name, '')
        WHEN normalized_name_lower CONTAINS term
            THEN coalesce(seed.normalized_name, '')
        WHEN umls_name_lower CONTAINS term
            THEN coalesce(seed.umls_canonical_name, '')
        WHEN any(alias IN aliases_lower WHERE alias CONTAINS term)
            THEN head([
                alias IN aliases
                WHERE toLower(alias) CONTAINS term
            ])
        ELSE null
    END AS matched_value

WITH
    term,
    seed,
    match_type,
    matched_value,
    CASE
        WHEN match_type IN [
            'exact_name',
            'exact_normalized_name',
            'exact_umls_canonical_name',
            'exact_umls_alias'
        ]
            THEN $exact_weight
        WHEN match_type = 'prefix'
            THEN $prefix_weight
        WHEN match_type = 'partial'
            THEN $partial_weight
        ELSE 0.0
    END AS lexical_weight

WHERE match_type IS NOT NULL
"""

_CONCEPT_GRAPH_SECTION_FILTER = """
  AND ($document_ids = [] OR d.doc_id IN $document_ids)
  AND coalesce(s.embed, false) = true
  AND trim(coalesce(s.text, '')) <> ''
  AND (
      NOT $exclude_summary_sections
      OR NOT any(
          excluded_title IN $excluded_title_prefixes
          WHERE toLower(trim(coalesce(s.title, ''))) STARTS WITH excluded_title
      )
  )
"""

_CONCEPT_GRAPH_EVIDENCE_RETURN = """
RETURN DISTINCT
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
    term AS query_term,
    coalesce(
        mentioned.name,
        mentioned.normalized_name,
        mentioned.umls_canonical_name,
        mentioned.umls_cui
    ) AS concept_name,
    matched_value,
    match_type,
    toFloat(lexical_weight) AS lexical_weight,
    evidence_source,
    relation_type,
    traversal_policy,
    review_needed,
    toFloat(evidence_weight) AS evidence_weight,
    seed_concept_name,
    seed_cui,
    target_cui
ORDER BY
    section_uid ASC,
    query_term ASC,
    evidence_source ASC,
    concept_name ASC
"""

_DIRECT_CONCEPT_GRAPH_EVIDENCE = (
    _CONCEPT_GRAPH_SEED_MATCH
    + """
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(mentioned:Concept)
WHERE mentioned = seed
"""
    + _CONCEPT_GRAPH_SECTION_FILTER
    + """
WITH
    d,
    s,
    mentioned,
    term,
    match_type,
    matched_value,
    lexical_weight,
    'direct' AS evidence_source,
    'MENTIONS' AS relation_type,
    null AS traversal_policy,
    false AS review_needed,
    $direct_weight AS evidence_weight,
    coalesce(
        seed.name,
        seed.normalized_name,
        seed.umls_canonical_name,
        seed.umls_cui
    ) AS seed_concept_name,
    seed.umls_cui AS seed_cui,
    mentioned.umls_cui AS target_cui
"""
    + _CONCEPT_GRAPH_EVIDENCE_RETURN
)

_SAME_AS_CONCEPT_GRAPH_EVIDENCE = (
    _CONCEPT_GRAPH_SEED_MATCH
    + """
MATCH (seed)-[:SAME_AS]-(same:Concept)
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(mentioned:Concept)
WHERE (
      mentioned = same
      OR (
          trim(coalesce(same.umls_cui, '')) <> ''
          AND mentioned.umls_cui = same.umls_cui
      )
  )
"""
    + _CONCEPT_GRAPH_SECTION_FILTER
    + """
WITH
    d,
    s,
    mentioned,
    term,
    match_type,
    matched_value,
    lexical_weight,
    'same_as' AS evidence_source,
    'SAME_AS' AS relation_type,
    null AS traversal_policy,
    false AS review_needed,
    $same_as_weight AS evidence_weight,
    coalesce(
        seed.name,
        seed.normalized_name,
        seed.umls_canonical_name,
        seed.umls_cui
    ) AS seed_concept_name,
    seed.umls_cui AS seed_cui,
    mentioned.umls_cui AS target_cui
"""
    + _CONCEPT_GRAPH_EVIDENCE_RETURN
)

_UMLS_SEED_CONCEPT_GRAPH_EVIDENCE = (
    _CONCEPT_GRAPH_SEED_MATCH
    + """
MATCH (origin_rep:Concept)
WHERE trim(coalesce(seed.umls_cui, '')) <> ''
  AND origin_rep.umls_cui = seed.umls_cui
MATCH (origin_rep)-[r]->(target:Concept)
WHERE type(r) STARTS WITH 'UMLS_'
  AND coalesce(toString(r.traversal_policy), '') IN $umls_policies
  AND ($include_review_needed OR coalesce(r.review_needed, false) = false)
  AND trim(coalesce(target.umls_cui, '')) <> ''
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(mentioned:Concept)
WHERE mentioned.umls_cui = target.umls_cui
"""
    + _CONCEPT_GRAPH_SECTION_FILTER
    + """
WITH
    d,
    s,
    mentioned,
    term,
    match_type,
    matched_value,
    lexical_weight,
    'umls_neighbor' AS evidence_source,
    type(r) AS relation_type,
    coalesce(toString(r.traversal_policy), '') AS traversal_policy,
    coalesce(r.review_needed, false) AS review_needed,
    $umls_neighbor_weight AS evidence_weight,
    coalesce(
        seed.name,
        seed.normalized_name,
        seed.umls_canonical_name,
        seed.umls_cui
    ) AS seed_concept_name,
    seed.umls_cui AS seed_cui,
    target.umls_cui AS target_cui
"""
    + _CONCEPT_GRAPH_EVIDENCE_RETURN
)

_UMLS_SAME_AS_CONCEPT_GRAPH_EVIDENCE = (
    _CONCEPT_GRAPH_SEED_MATCH
    + """
MATCH (seed)-[:SAME_AS]-(same:Concept)
MATCH (origin_rep:Concept)
WHERE trim(coalesce(same.umls_cui, '')) <> ''
  AND origin_rep.umls_cui = same.umls_cui
MATCH (origin_rep)-[r]->(target:Concept)
WHERE type(r) STARTS WITH 'UMLS_'
  AND coalesce(toString(r.traversal_policy), '') IN $umls_policies
  AND ($include_review_needed OR coalesce(r.review_needed, false) = false)
  AND trim(coalesce(target.umls_cui, '')) <> ''
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(mentioned:Concept)
WHERE mentioned.umls_cui = target.umls_cui
"""
    + _CONCEPT_GRAPH_SECTION_FILTER
    + """
WITH
    d,
    s,
    mentioned,
    term,
    match_type,
    matched_value,
    lexical_weight,
    'umls_neighbor' AS evidence_source,
    type(r) AS relation_type,
    coalesce(toString(r.traversal_policy), '') AS traversal_policy,
    coalesce(r.review_needed, false) AS review_needed,
    $umls_neighbor_weight AS evidence_weight,
    coalesce(
        seed.name,
        seed.normalized_name,
        seed.umls_canonical_name,
        seed.umls_cui
    ) AS seed_concept_name,
    same.umls_cui AS seed_cui,
    target.umls_cui AS target_cui
"""
    + _CONCEPT_GRAPH_EVIDENCE_RETURN
)


class KGCandidate(BaseModel):
    """One Section candidate plus provenance from the retrieval pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section: KGSectionResult
    source: KGCandidateSource
    source_rank: int = Field(ge=1)
    final_rank: int | None = Field(default=None, ge=1)

    direct: bool = True
    seed_uid: str | None = None
    seed_rank: int | None = Field(default=None, ge=1)
    graph_distance: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("seed_uid")
    @classmethod
    def normalize_seed_uid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @property
    def section_uid(self) -> str:
        return self.section.section_uid

    @property
    def document_id(self) -> str:
        return self.section.document_id

    @property
    def printed_section_id(self) -> str | None:
        return self.section.printed_section_id

    @property
    def title(self) -> str | None:
        return self.section.title


class KGSectionSearchProtocol(Protocol):
    """Subset of ``KGSectionTools`` needed by candidate generators."""

    def search_sections_by_concepts(
        self,
        concepts: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
        top_k: int = 10,
        require_all: bool = False,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> list[KGSectionResult]: ...

    def search_sections_by_title(
        self,
        title_terms: Sequence[str] | str,
        *,
        document_ids: Sequence[str] | str | None = None,
        top_k: int = 10,
        require_all: bool = False,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> list[KGSectionResult]: ...


class GraphReadClientProtocol(Protocol):
    """Read-only graph client interface used by concept expansion."""

    def run_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class CandidateGeneratorProtocol(Protocol):
    """Common interface for Section candidate generators."""

    name: str

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]: ...


class MentionsCandidateGenerator:
    """Generate Section candidates through ``Section-[:MENTIONS]->Concept``.

    With ``ranking_mode='concept_match'`` this is the pure graph baseline:
    candidates and their order depend only on MENTIONS concept matches.
    ``weighted_match`` keeps the same candidate set but applies the existing
    lexical match weights and title bonus implemented by ``KGSectionTools``.
    """

    name = "mentions"

    def __init__(
        self,
        tools: KGSectionSearchProtocol,
        *,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> None:
        self.tools = tools
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]:
        normalized_terms = _normalize_terms(terms)
        validated_top_k = _validate_top_k(top_k)

        results = self.tools.search_sections_by_concepts(
            normalized_terms,
            document_ids=document_ids,
            top_k=validated_top_k,
            require_all=bool(require_all),
            ranking_mode=self.ranking_mode,
            exclude_summary_sections=self.exclude_summary_sections,
        )
        return _wrap_results(results, source="mentions")


class ConceptGraphExpansionCandidateGenerator:
    """Generate Section candidates using controlled Concept expansion.

    Seed concepts are selected by the same lexical rules used for the pure
    MENTIONS search. Direct ``Section-[:MENTIONS]->Concept`` evidence is always
    included; optional ``SAME_AS`` and safe UMLS neighbor expansion remain
    explicit in candidate metadata and diagnostics.
    """

    name = "concept_graph_expansion"

    def __init__(
        self,
        client: GraphReadClientProtocol,
        *,
        include_same_as: bool,
        umls_policies: Sequence[str] = (),
        include_review_needed: bool = False,
        ranking_mode: KGRankingMode = "concept_match",
        exclude_summary_sections: bool = True,
    ) -> None:
        self.client = client
        self.include_same_as = bool(include_same_as)
        self.umls_policies = _normalize_policy_values(umls_policies)
        self.include_review_needed = bool(include_review_needed)
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]:
        normalized_terms = _normalize_terms(terms)
        normalized_document_ids = _normalize_optional_values(document_ids)
        validated_top_k = _validate_top_k(top_k)

        params = {
            "terms": [term.casefold() for term in normalized_terms],
            "document_ids": normalized_document_ids,
            "exclude_summary_sections": self.exclude_summary_sections,
            "excluded_title_prefixes": _EXCLUDED_TITLE_PREFIXES,
            "umls_policies": list(self.umls_policies),
            "include_review_needed": self.include_review_needed,
            "direct_weight": _CONCEPT_GRAPH_DIRECT_WEIGHT,
            "same_as_weight": _CONCEPT_GRAPH_SAME_AS_WEIGHT,
            "umls_neighbor_weight": _CONCEPT_GRAPH_UMLS_WEIGHT,
            "exact_weight": _CONCEPT_GRAPH_EXACT_WEIGHT,
            "prefix_weight": _CONCEPT_GRAPH_PREFIX_WEIGHT,
            "partial_weight": _CONCEPT_GRAPH_PARTIAL_WEIGHT,
        }
        evidence_rows = _run_concept_graph_evidence_queries(
            self.client,
            params=params,
            include_same_as=self.include_same_as,
            include_umls=bool(self.umls_policies),
        )

        results = _evidence_rows_to_results(
            evidence_rows,
            terms=params["terms"],
            require_all=bool(require_all),
            top_k=validated_top_k,
            ranking_mode=self.ranking_mode,
        )
        return _wrap_concept_graph_results(results)


class RescueConceptGraphExpansionCandidateGenerator:
    """Preserve direct MENTIONS order and append only expansion rescues."""

    name = "concept_graph_rescue"

    def __init__(
        self,
        direct_generator: MentionsCandidateGenerator,
        expansion_generator: ConceptGraphExpansionCandidateGenerator,
        *,
        expanded_top_k_multiplier: int = 3,
    ) -> None:
        self.direct_generator = direct_generator
        self.expansion_generator = expansion_generator
        self.expanded_top_k_multiplier = _validate_expanded_top_k_multiplier(
            expanded_top_k_multiplier
        )
        self.ranking_mode = direct_generator.ranking_mode

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]:
        validated_top_k = _validate_top_k(top_k)
        direct_candidates = self.direct_generator.generate(
            terms,
            top_k=validated_top_k,
            require_all=bool(require_all),
            document_ids=document_ids,
        )
        expansion_candidates = self.expansion_generator.generate(
            terms,
            top_k=min(
                100,
                validated_top_k * self.expanded_top_k_multiplier,
            ),
            require_all=bool(require_all),
            document_ids=document_ids,
        )

        output: list[KGCandidate] = []
        index_by_uid: dict[str, int] = {}

        for candidate in direct_candidates:
            if candidate.section_uid in index_by_uid:
                continue
            index_by_uid[candidate.section_uid] = len(output)
            output.append(candidate)

        for candidate in expansion_candidates:
            evidence_sources = _candidate_expansion_evidence_sources(
                candidate
            )
            if not evidence_sources:
                continue

            existing_index = index_by_uid.get(candidate.section_uid)
            if existing_index is not None:
                output[existing_index] = _merge_expansion_support(
                    output[existing_index],
                    candidate,
                    evidence_sources,
                )
                continue

            if len(output) >= validated_top_k:
                continue

            rescue_candidate = _as_rescue_candidate(
                candidate,
                evidence_sources,
            )
            index_by_uid[rescue_candidate.section_uid] = len(output)
            output.append(rescue_candidate)

        return output[:validated_top_k]


class TitleCandidateGenerator:
    """Generate Section candidates by matching section titles.

    This generator is retained for the advanced role-aware pipeline. It is not
    required by the MENTIONS-only baseline.
    """

    name = "title"

    def __init__(
        self,
        tools: KGSectionSearchProtocol,
        *,
        ranking_mode: KGRankingMode = "weighted_match",
        exclude_summary_sections: bool = True,
    ) -> None:
        self.tools = tools
        self.ranking_mode = _validate_ranking_mode(ranking_mode)
        self.exclude_summary_sections = bool(exclude_summary_sections)

    def generate(
        self,
        terms: Sequence[str] | str,
        *,
        top_k: int,
        require_all: bool = False,
        document_ids: Sequence[str] | str | None = None,
    ) -> list[KGCandidate]:
        normalized_terms = _normalize_terms(terms)
        validated_top_k = _validate_top_k(top_k)

        results = self.tools.search_sections_by_title(
            normalized_terms,
            document_ids=document_ids,
            top_k=validated_top_k,
            require_all=bool(require_all),
            ranking_mode=self.ranking_mode,
            exclude_summary_sections=self.exclude_summary_sections,
        )
        return _wrap_results(results, source="title")


def deduplicate_candidates(
    candidates: Sequence[KGCandidate],
) -> list[KGCandidate]:
    """Keep the first occurrence of each canonical Section node."""

    output: list[KGCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.section_uid in seen:
            continue
        seen.add(candidate.section_uid)
        output.append(candidate)
    return output


def _wrap_results(
    results: Sequence[KGSectionResult],
    *,
    source: Literal["mentions", "title"],
) -> list[KGCandidate]:
    candidates: list[KGCandidate] = []
    seen: set[str] = set()

    for fallback_rank, result in enumerate(results, start=1):
        if result.section_uid in seen:
            continue
        seen.add(result.section_uid)
        source_rank = result.rank or fallback_rank
        candidates.append(
            KGCandidate(
                section=result,
                source=source,
                source_rank=source_rank,
                direct=True,
                seed_uid=result.section_uid,
                seed_rank=source_rank,
                graph_distance=0,
            )
        )
    return candidates


def _run_concept_graph_evidence_queries(
    client: GraphReadClientProtocol,
    *,
    params: dict[str, Any],
    include_same_as: bool,
    include_umls: bool,
) -> list[dict[str, Any]]:
    rows = list(client.run_read(_DIRECT_CONCEPT_GRAPH_EVIDENCE, params))

    if include_same_as:
        rows.extend(
            client.run_read(_SAME_AS_CONCEPT_GRAPH_EVIDENCE, params)
        )

    if include_umls:
        rows.extend(
            client.run_read(_UMLS_SEED_CONCEPT_GRAPH_EVIDENCE, params)
        )
        if include_same_as:
            rows.extend(
                client.run_read(
                    _UMLS_SAME_AS_CONCEPT_GRAPH_EVIDENCE,
                    params,
                )
            )

    return rows


def _evidence_rows_to_results(
    rows: Sequence[dict[str, Any]],
    *,
    terms: Sequence[str],
    require_all: bool,
    top_k: int,
    ranking_mode: KGRankingMode,
) -> list[KGSectionResult]:
    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        section_uid = str(row.get("section_uid") or "").strip()
        if not section_uid:
            continue

        group = grouped.setdefault(
            section_uid,
            {
                "section": row,
                "diagnostics": [],
                "diagnostic_keys": set(),
            },
        )
        diagnostic = _diagnostic_from_evidence_row(row)
        diagnostic_key = _diagnostic_key(diagnostic)
        if diagnostic_key in group["diagnostic_keys"]:
            continue
        group["diagnostic_keys"].add(diagnostic_key)
        group["diagnostics"].append(diagnostic)

    results: list[KGSectionResult] = []
    required_terms = {str(term).casefold() for term in terms}

    for group in grouped.values():
        diagnostics = group["diagnostics"]
        matched_terms = _ordered_unique(
            str(item["query_term"]).casefold()
            for item in diagnostics
            if item.get("query_term")
        )
        if require_all and set(matched_terms) != required_terms:
            continue

        matched_concepts = _ordered_unique(
            str(item["concept_name"])
            for item in diagnostics
            if item.get("concept_name")
        )
        scores = _score_concept_graph_diagnostics(diagnostics)
        active_score = scores[ranking_mode]
        section_row = dict(group["section"])
        section_row.update(
            {
                "matched_concepts": matched_concepts,
                "matched_terms": matched_terms,
                "score": active_score,
                "score_type": ranking_mode,
                "scores": scores,
                "match_diagnostics": diagnostics,
            }
        )
        results.append(KGSectionResult.from_record(section_row))

    results.sort(
        key=lambda result: (
            -(result.score or 0.0),
            -(
                result.scores.weighted_match
                if result.scores is not None
                else 0.0
            ),
            -(
                result.scores.concept_match
                if result.scores is not None
                else 0.0
            ),
            result.section_uid,
        )
    )

    return [
        result.model_copy(update={"rank": rank})
        for rank, result in enumerate(results[:top_k], start=1)
    ]


def _diagnostic_from_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_term": str(row.get("query_term") or "").strip(),
        "concept_name": _optional_str(row.get("concept_name")),
        "matched_value": _optional_str(row.get("matched_value")),
        "match_type": str(row.get("match_type") or "").strip(),
        "weight": _float_or_default(row.get("evidence_weight"), 0.0),
        "evidence_source": _optional_str(row.get("evidence_source")),
        "relation_type": _optional_str(row.get("relation_type")),
        "traversal_policy": _optional_str(row.get("traversal_policy")),
        "review_needed": bool(row.get("review_needed", False)),
        "lexical_weight": _float_or_default(row.get("lexical_weight"), 0.0),
        "seed_concept_name": _optional_str(row.get("seed_concept_name")),
        "seed_cui": _optional_str(row.get("seed_cui")),
        "target_cui": _optional_str(row.get("target_cui")),
    }


def _diagnostic_key(diagnostic: dict[str, Any]) -> tuple[Any, ...]:
    return (
        diagnostic.get("query_term"),
        diagnostic.get("concept_name"),
        diagnostic.get("matched_value"),
        diagnostic.get("match_type"),
        diagnostic.get("evidence_source"),
        diagnostic.get("relation_type"),
        diagnostic.get("traversal_policy"),
        diagnostic.get("seed_cui"),
        diagnostic.get("target_cui"),
    )


def _score_concept_graph_diagnostics(
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, float]:
    by_term: dict[str, list[dict[str, Any]]] = {}
    for diagnostic in diagnostics:
        term = str(diagnostic.get("query_term") or "").casefold()
        if not term:
            continue
        by_term.setdefault(term, []).append(diagnostic)

    lexical_score_sum = 0.0
    evidence_score_sum = 0.0
    weighted_match_score = 0.0

    for term_diagnostics in by_term.values():
        lexical_score_sum += max(
            _float_or_default(item.get("lexical_weight"), 0.0)
            for item in term_diagnostics
        )
        evidence_score_sum += max(
            _float_or_default(item.get("weight"), 0.0)
            for item in term_diagnostics
        )
        weighted_match_score += max(
            _float_or_default(item.get("lexical_weight"), 0.0)
            * _float_or_default(item.get("weight"), 0.0)
            for item in term_diagnostics
        )

    concept_match_score = (
        float(len(by_term))
        + (lexical_score_sum / 10.0)
        + (evidence_score_sum / 100.0)
    )
    return {
        "concept_match": concept_match_score,
        "weighted_match": weighted_match_score,
    }


def _wrap_concept_graph_results(
    results: Sequence[KGSectionResult],
) -> list[KGCandidate]:
    candidates: list[KGCandidate] = []
    seen: set[str] = set()

    for fallback_rank, result in enumerate(results, start=1):
        if result.section_uid in seen:
            continue
        seen.add(result.section_uid)
        source_rank = result.rank or fallback_rank
        source = _candidate_source_from_diagnostics(result)
        candidates.append(
            KGCandidate(
                section=result,
                source=source,
                source_rank=source_rank,
                direct=(source == "mentions"),
                seed_uid=(
                    result.section_uid if source == "mentions" else None
                ),
                seed_rank=source_rank if source == "mentions" else None,
                graph_distance=_source_graph_distance(source),
                metadata={
                    "generator": "concept_graph_expansion",
                    "evidence_sources": _diagnostic_evidence_sources(result),
                },
            )
        )

    return candidates


def _candidate_expansion_evidence_sources(
    candidate: KGCandidate,
) -> list[Literal["same_as", "umls_neighbor"]]:
    sources = _ordered_unique(
        [
            *candidate.metadata.get("evidence_sources", []),
            *[
                diagnostic.evidence_source
                for diagnostic in candidate.section.match_diagnostics
            ],
            candidate.source,
        ]
    )
    return [
        source
        for source in sources
        if source in {"same_as", "umls_neighbor"}
    ]  # type: ignore[list-item]


def _merge_expansion_support(
    direct_candidate: KGCandidate,
    expansion_candidate: KGCandidate,
    evidence_sources: Sequence[Literal["same_as", "umls_neighbor"]],
) -> KGCandidate:
    metadata = _expansion_support_metadata(
        direct_candidate.metadata,
        evidence_sources,
        expansion_candidate=expansion_candidate,
    )
    return direct_candidate.model_copy(update={"metadata": metadata})


def _as_rescue_candidate(
    candidate: KGCandidate,
    evidence_sources: Sequence[Literal["same_as", "umls_neighbor"]],
) -> KGCandidate:
    source = (
        candidate.source
        if candidate.source in {"same_as", "umls_neighbor"}
        else _primary_expansion_source(evidence_sources)
    )
    return candidate.model_copy(
        update={
            "source": source,
            "direct": False,
            "seed_uid": None,
            "seed_rank": None,
            "graph_distance": _source_graph_distance(source),
            "metadata": _expansion_support_metadata(
                candidate.metadata,
                evidence_sources,
                expansion_candidate=candidate,
            ),
        }
    )


def _expansion_support_metadata(
    metadata: dict[str, Any],
    evidence_sources: Sequence[Literal["same_as", "umls_neighbor"]],
    *,
    expansion_candidate: KGCandidate,
) -> dict[str, Any]:
    merged = dict(metadata)
    sources = _ordered_unique(
        [
            *merged.get("expansion_evidence_sources", []),
            *evidence_sources,
        ]
    )
    merged.update(
        {
            "has_expansion_support": True,
            "expansion_evidence_sources": sources,
            "expansion_source": _combined_expansion_source(sources),
            "expansion_candidate_source": expansion_candidate.source,
            "expansion_candidate_source_rank": (
                expansion_candidate.source_rank
            ),
        }
    )
    return merged


def _primary_expansion_source(
    evidence_sources: Sequence[Literal["same_as", "umls_neighbor"]],
) -> Literal["same_as", "umls_neighbor"]:
    if "umls_neighbor" in evidence_sources:
        return "umls_neighbor"
    return "same_as"


def _combined_expansion_source(
    evidence_sources: Sequence[str],
) -> Literal["same_as", "umls_neighbor", "mixed"]:
    unique_sources = set(evidence_sources)
    if unique_sources == {"same_as"}:
        return "same_as"
    if unique_sources == {"umls_neighbor"}:
        return "umls_neighbor"
    return "mixed"


def _candidate_source_from_diagnostics(
    result: KGSectionResult,
) -> KGCandidateSource:
    sources = _diagnostic_evidence_sources(result)
    if "direct" in sources:
        return "mentions"
    if "same_as" in sources:
        return "same_as"
    if "umls_neighbor" in sources:
        return "umls_neighbor"
    return "mentions"


def _diagnostic_evidence_sources(result: KGSectionResult) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for diagnostic in result.match_diagnostics:
        source = diagnostic.evidence_source
        if source is None or source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


def _source_graph_distance(source: KGCandidateSource) -> int:
    if source == "mentions":
        return 0
    if source == "same_as":
        return 1
    if source == "umls_neighbor":
        return 2
    return 0


def _rows_to_results(
    rows: Sequence[dict[str, Any]],
) -> list[KGSectionResult]:
    results: list[KGSectionResult] = []
    seen: set[str] = set()

    for row in rows:
        result = KGSectionResult.from_record(row)
        if result.section_uid in seen:
            continue
        seen.add(result.section_uid)
        results.append(
            result.model_copy(update={"rank": len(results) + 1})
        )

    return results


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
    if values is None:
        return []
    return _normalize_values_without_required(values)


def _normalize_policy_values(values: Sequence[str] | str) -> tuple[str, ...]:
    return tuple(
        item.casefold()
        for item in _normalize_values_without_required(values)
    )


def _normalize_values_without_required(
    values: Sequence[str] | str,
) -> list[str]:
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


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)

    return output


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    item = str(value).strip()
    return item or None


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_top_k(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc
    if normalized < 1 or normalized > 100:
        raise ValueError("top_k must be between 1 and 100")
    return normalized


def _validate_expanded_top_k_multiplier(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "expanded_top_k_multiplier must be an integer"
        ) from exc
    if normalized < 1 or normalized > 100:
        raise ValueError(
            "expanded_top_k_multiplier must be between 1 and 100"
        )
    return normalized


def _validate_ranking_mode(value: str) -> KGRankingMode:
    normalized = str(value).strip().lower()
    if normalized not in {"concept_match", "weighted_match"}:
        raise ValueError(
            "ranking_mode must be 'concept_match' or 'weighted_match'"
        )
    return normalized  # type: ignore[return-value]
