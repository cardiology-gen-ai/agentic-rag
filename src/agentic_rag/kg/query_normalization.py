"""Deterministic, semantics-preserving normalization of KG router terms.

The normalizer is deliberately conservative. It repairs representation-level
artifacts (Unicode compatibility forms, whitespace, quote/dash variants and a
small allow-list of known serialization delimiters) without introducing
synonyms, stemming, lemmatization, ontology expansion or answer inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Literal

from agentic_rag.agent.output import KGMentionsPlan


RouterTermNormalizationMode = Literal["none", "safe_v1"]

_WS_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
# Known structured-output serialization artifacts observed in frozen router
# plans. Keep this list intentionally small: these are not semantic separators.
_SERIALIZATION_SPLIT_RE = re.compile(r"(?:」「|\]\s*\[)")
_SURROUNDING_QUOTES = " \t\r\n\"'`“”‘’「」"


@dataclass(frozen=True)
class RouterTermNormalizationResult:
    """Normalized plan plus an auditable description of any changes."""

    plan: KGMentionsPlan
    changed: bool
    split_count: int
    overflow_avoided: bool
    original_terms: tuple[str, ...]
    normalized_terms: tuple[str, ...]


def normalize_mentions_plan(
    plan: KGMentionsPlan,
    *,
    mode: RouterTermNormalizationMode = "none",
) -> RouterTermNormalizationResult:
    """Return a normalized MENTIONS plan without semantic rewriting.

    ``none`` preserves the validated input exactly.

    ``safe_v1`` applies only representation-level cleanup. If splitting a
    malformed term would exceed the KGMentionsPlan five-term contract, the
    split is conservatively abandoned rather than silently dropping terms.
    """

    if mode not in {"none", "safe_v1"}:
        raise ValueError(f"Unsupported router term normalization mode: {mode!r}")

    original = tuple(plan.terms)
    if mode == "none":
        return RouterTermNormalizationResult(
            plan=plan,
            changed=False,
            split_count=0,
            overflow_avoided=False,
            original_terms=original,
            normalized_terms=original,
        )

    normalized_unsplit = tuple(_normalize_surface(term) for term in original)
    split_terms: list[str] = []
    split_count = 0
    for term in normalized_unsplit:
        parts = [_clean_part(part) for part in _SERIALIZATION_SPLIT_RE.split(term)]
        parts = [part for part in parts if part]
        if len(parts) > 1:
            split_count += len(parts) - 1
        split_terms.extend(parts or [term])

    split_terms = _deduplicate(split_terms)
    overflow_avoided = len(split_terms) > 5
    if overflow_avoided:
        final_terms = _deduplicate(normalized_unsplit)
        split_count = 0
    else:
        final_terms = split_terms

    normalized_plan = KGMentionsPlan(
        terms=final_terms,
        require_all=plan.require_all,
    )
    normalized = tuple(normalized_plan.terms)
    return RouterTermNormalizationResult(
        plan=normalized_plan,
        changed=(normalized != original),
        split_count=split_count,
        overflow_avoided=overflow_avoided,
        original_terms=original,
        normalized_terms=normalized,
    )


def _normalize_surface(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = _DASH_RE.sub("-", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _clean_part(value: str) -> str:
    return _WS_RE.sub(" ", str(value).strip(_SURROUNDING_QUOTES)).strip()


def _deduplicate(values) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = _clean_part(value)
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(term)
    return output
