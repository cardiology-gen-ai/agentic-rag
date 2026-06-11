"""Canonical identifiers and matching helpers for retrieval evaluation.

The gold dataset intentionally remains independent from Neo4j-internal UIDs.
Gold section labels are matched to retrieved KG sections through the stable,
human-readable tuple:

    document_id + printed_section_id + normalized title

No fuzzy matching is used. Normalization only removes irrelevant formatting
and typographic differences.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable

from agentic_rag.kg.models import KGSectionResult


_SECTION_LABEL_PATTERN = re.compile(
    r"^\s*"
    r"(?P<section_id>\d+(?:\.\d+)*)"
    r"(?:\.)?"
    r"\s+"
    r"(?P<title>.+?)"
    r"\s*$"
)

_TRANSLATION_TABLE = str.maketrans(
    {
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\u200b": "",
        "\ufeff": "",
        "’": "'",
        "‘": "'",
        "`": "'",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


@dataclass(frozen=True, slots=True)
class SectionKey:
    """Canonical, backend-independent identity of one semantic section."""

    document_id: str
    printed_section_id: str
    title: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _require_non_empty(value: object, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must not be None")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")

    return text


def normalize_document_id(value: str) -> str:
    """Normalize a guideline identifier without changing its semantics.

    A trailing ``.md`` or ``.pdf`` extension is ignored so that a dataset
    filename and a Neo4j ``doc_id`` can be compared directly.
    """

    normalized = unicodedata.normalize(
        "NFKC",
        _require_non_empty(value, "document_id"),
    )
    normalized = normalized.translate(_TRANSLATION_TABLE).strip()

    for suffix in (".md", ".pdf"):
        if normalized.casefold().endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip()
            break

    if not normalized:
        raise ValueError("document_id is empty after normalization")

    return normalized.casefold()


def normalize_printed_section_id(value: str) -> str:
    """Normalize the section number printed in the source guideline."""

    normalized = unicodedata.normalize(
        "NFKC",
        _require_non_empty(value, "printed_section_id"),
    )
    normalized = normalized.translate(_TRANSLATION_TABLE)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")

    if not normalized:
        raise ValueError("printed_section_id is empty after normalization")

    return normalized


def normalize_section_title(value: str) -> str:
    """Normalize harmless title formatting while preserving meaning."""

    normalized = unicodedata.normalize(
        "NFKC",
        _require_non_empty(value, "section title"),
    )
    normalized = normalized.translate(_TRANSLATION_TABLE)
    normalized = " ".join(normalized.split())
    normalized = normalized.rstrip(" .:;")

    if not normalized:
        raise ValueError("section title is empty after normalization")

    return normalized.casefold()


def parse_gold_section_label(section_label: str) -> tuple[str, str]:
    """Split a gold label into printed section ID and title.

    Expected examples include ``"7.1.1.1. Diagnostic criteria"`` and
    ``"3.1. Definitions"``.
    """

    label = _require_non_empty(section_label, "section_label")
    match = _SECTION_LABEL_PATTERN.fullmatch(label)

    if match is None:
        raise ValueError(
            "Invalid gold section label. Expected '<number>. <title>', "
            f"got: {section_label!r}"
        )

    return (
        normalize_printed_section_id(match.group("section_id")),
        normalize_section_title(match.group("title")),
    )


def section_key_from_gold(
    document_id: str,
    section_label: str,
) -> SectionKey:
    """Build a canonical key from one gold-dataset section annotation."""

    printed_section_id, title = parse_gold_section_label(section_label)
    return SectionKey(
        document_id=normalize_document_id(document_id),
        printed_section_id=printed_section_id,
        title=title,
    )


def section_key_from_result(result: KGSectionResult) -> SectionKey:
    """Build a canonical key from one retrieved Neo4j Section result."""

    if result.printed_section_id is None:
        raise ValueError(
            f"Retrieved section {result.section_uid!r} has no "
            "printed_section_id"
        )
    if result.title is None:
        raise ValueError(
            f"Retrieved section {result.section_uid!r} has no title"
        )

    return SectionKey(
        document_id=normalize_document_id(result.document_id),
        printed_section_id=normalize_printed_section_id(
            result.printed_section_id
        ),
        title=normalize_section_title(result.title),
    )


def sections_match(
    gold_document_id: str,
    gold_section_label: str,
    retrieved: KGSectionResult,
) -> bool:
    """Return whether one retrieved section exactly matches one gold label."""

    return section_key_from_gold(
        gold_document_id,
        gold_section_label,
    ) == section_key_from_result(retrieved)


def deduplicate_section_keys(
    keys: Iterable[SectionKey],
) -> list[SectionKey]:
    """Deduplicate keys while preserving the first-occurrence ranking."""

    unique: list[SectionKey] = []
    seen: set[SectionKey] = set()

    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)

    return unique