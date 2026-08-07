"""Optional graph expansion stages for modular KG retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from agentic_rag.kg.candidate_generators import (
    KGCandidate,
    deduplicate_candidates,
)
from agentic_rag.kg.models import KGSectionResult


_MAX_HIERARCHY_DEPTH = 8
_MAX_EXPANDED_ROWS = 5000

_EXCLUDED_TITLE_PREFIXES = [
    "key messages",
    "gaps in evidence",
    "what to do",
    "references",
    "bibliography",
]

_FIND_DESCENDANTS = """
UNWIND $seeds AS seed
MATCH (root:Section {uid: seed.uid})
MATCH path = (root)-[:HAS_CHILD*1..8]->(s:Section)
WHERE length(path) <= $max_depth
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
MATCH (d:Document)-[:HAS_SECTION]->(s)
WITH seed, d, s, min(length(path)) AS hierarchy_distance
RETURN
    seed.uid AS seed_uid,
    toInteger(seed.rank) AS seed_rank,
    hierarchy_distance,
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
    [] AS matched_terms,
    null AS score,
    null AS score_type,
    null AS scores,
    [] AS match_diagnostics
ORDER BY
    seed_rank ASC,
    hierarchy_distance ASC,
    section_uid ASC
LIMIT $max_rows
"""


class GraphReadProtocol(Protocol):
    """Read-only graph client interface used by expanders."""

    def run_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class CandidateExpanderProtocol(Protocol):
    """Common interface for optional candidate expansion."""

    name: str

    def expand(
        self,
        candidates: Sequence[KGCandidate],
    ) -> list[KGCandidate]: ...


class NoOpExpander:
    """Return the direct candidates unchanged."""

    name = "none"

    def expand(
        self,
        candidates: Sequence[KGCandidate],
    ) -> list[KGCandidate]:
        return deduplicate_candidates(candidates)


class DescendantExpander:
    """Add descendant Section nodes reached through ``HAS_CHILD``.

    This is true hierarchy expansion, not the existing hierarchy-support
    reranking. Direct MENTIONS candidates remain in the output and descendants
    carry the seed UID, seed rank, and graph distance that produced them.
    ``NEXT`` is intentionally not traversed because it represents reading order,
    not parent-child structure.
    """

    name = "descendants"

    def __init__(
        self,
        client: GraphReadProtocol,
        *,
        max_depth: int = 3,
        max_rows: int = 1000,
        exclude_summary_sections: bool = True,
    ) -> None:
        self.client = client
        self.max_depth = _validate_max_depth(max_depth)
        self.max_rows = _validate_max_rows(max_rows)
        self.exclude_summary_sections = bool(exclude_summary_sections)

    def expand(
        self,
        candidates: Sequence[KGCandidate],
    ) -> list[KGCandidate]:
        direct_candidates = deduplicate_candidates(candidates)
        if not direct_candidates or self.max_depth == 0:
            return direct_candidates

        seed_payload = [
            {"uid": item.section_uid, "rank": item.source_rank}
            for item in direct_candidates
        ]
        rows = self.client.run_read(
            _FIND_DESCENDANTS,
            {
                "seeds": seed_payload,
                "max_depth": self.max_depth,
                "max_rows": self.max_rows,
                "exclude_summary_sections": self.exclude_summary_sections,
                "excluded_title_prefixes": _EXCLUDED_TITLE_PREFIXES,
            },
        )

        direct_uids = {item.section_uid for item in direct_candidates}
        best_descendant: dict[str, KGCandidate] = {}

        for row in rows:
            section_uid = str(row.get("section_uid") or "").strip()
            seed_uid = str(row.get("seed_uid") or "").strip()
            if not section_uid or not seed_uid or section_uid in direct_uids:
                continue

            try:
                distance = int(row.get("hierarchy_distance"))
                seed_rank = int(row.get("seed_rank"))
            except (TypeError, ValueError):
                continue

            result = KGSectionResult.from_record(row)
            candidate = KGCandidate(
                section=result,
                source="descendant",
                source_rank=seed_rank,
                direct=False,
                seed_uid=seed_uid,
                seed_rank=seed_rank,
                graph_distance=distance,
                metadata={"expansion": "HAS_CHILD"},
            )

            current = best_descendant.get(section_uid)
            if current is None or _descendant_key(candidate) < _descendant_key(
                current
            ):
                best_descendant[section_uid] = candidate

        descendants = sorted(best_descendant.values(), key=_descendant_key)
        return direct_candidates + descendants


def _descendant_key(candidate: KGCandidate) -> tuple[int, int, str, str]:
    return (
        candidate.seed_rank or 10**9,
        candidate.graph_distance or 10**9,
        candidate.seed_uid or "",
        candidate.section_uid,
    )


def _validate_max_depth(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_depth must be an integer") from exc
    if normalized < 0 or normalized > _MAX_HIERARCHY_DEPTH:
        raise ValueError(
            f"max_depth must be between 0 and {_MAX_HIERARCHY_DEPTH}"
        )
    return normalized


def _validate_max_rows(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_rows must be an integer") from exc
    if normalized < 1 or normalized > _MAX_EXPANDED_ROWS:
        raise ValueError(
            f"max_rows must be between 1 and {_MAX_EXPANDED_ROWS}"
        )
    return normalized
