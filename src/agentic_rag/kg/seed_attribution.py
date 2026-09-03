"""Pure helpers for post-hoc semantic-seed/gold attribution diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SeedSupport:
    router_term: str
    seed_rank: int
    concept_name: str
    similarity: float | None
    df: int | None
    exact_supported: bool


def section_identity(document_id: str, printed_section_id: str) -> tuple[str, str]:
    return (str(document_id).strip().casefold(), str(printed_section_id).strip())


def build_section_concept_index(
    payload: Mapping[str, Any],
) -> dict[tuple[str, str], set[str]]:
    section_concepts = payload.get("section_concepts") or {}
    section_documents = payload.get("section_documents") or {}
    result: dict[tuple[str, str], set[str]] = {}

    for uid, concepts in section_concepts.items():
        uid_s = str(uid)
        document_id = str(section_documents.get(uid_s) or "").strip()
        printed = ""
        if "::" in uid_s:
            uid_doc, printed = uid_s.split("::", 1)
            if not document_id:
                document_id = uid_doc
        if not document_id or not printed:
            continue
        result[section_identity(document_id, printed)] = {
            str(value).strip()
            for value in (concepts or [])
            if str(value).strip()
        }
    return result


def seed_support_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    gold_concepts: set[str],
) -> list[SeedSupport]:
    supports: list[SeedSupport] = []
    for row in seed_rows:
        concept = str(row.get("concept_name") or "").strip()
        if concept not in gold_concepts:
            continue
        similarity = row.get("similarity")
        df = row.get("df")
        supports.append(
            SeedSupport(
                router_term=str(row.get("router_term") or ""),
                seed_rank=int(row.get("seed_rank") or 0),
                concept_name=concept,
                similarity=float(similarity) if similarity not in (None, "") else None,
                df=int(float(df)) if df not in (None, "") else None,
                exact_supported=str(row.get("exact_supported") or "").strip().lower() == "true"
                or row.get("exact_supported") is True,
            )
        )
    return sorted(
        supports,
        key=lambda x: (x.seed_rank, x.router_term.casefold(), x.concept_name.casefold(), x.concept_name),
    )


def classify_support(
    *,
    treatment_top_k: int,
    supports: Sequence[SeedSupport],
) -> str:
    base = [s for s in supports if s.seed_rank <= treatment_top_k]
    incremental = [s for s in supports if treatment_top_k < s.seed_rank <= 3]
    if incremental and not base:
        return "new_direct_reachability"
    if incremental and base:
        return "additional_gold_support_or_budget_boost"
    if base and not incremental:
        return "preselection_redistribution_without_new_gold_support"
    return "unattributed_by_direct_concept_membership"
