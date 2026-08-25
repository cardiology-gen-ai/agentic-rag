"""Document-scope guards shared by controlled KG retrieval experiments.

The helpers are intentionally dependency-light so snapshot/replay scripts can
use them without importing the full evaluation stack.
"""
from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_TRANSLATION_TABLE = str.maketrans({"−": "-", "–": "-", "—": "-", "‑": "-"})


def normalize_document_id(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("document_id must be non-empty")
    normalized = unicodedata.normalize("NFKC", text).translate(_TRANSLATION_TABLE).strip()
    for suffix in (".md", ".pdf"):
        if normalized.casefold().endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip()
            break
    if not normalized:
        raise ValueError("document_id is empty after normalization")
    return normalized.casefold()


def normalize_document_scope(values: Sequence[str] | str | None) -> list[str]:
    """Normalize and de-duplicate a document scope while preserving input spelling/order."""
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else list(values)
    output: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value).strip()
        if not text:
            continue
        key = normalize_document_id(text)
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def question_gold_document_ids(question: Mapping[str, Any]) -> list[str]:
    sources = question.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Question must contain a non-empty 'sources' list")
    output: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("Every source must be an object")
        document_id = str(source.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("Every source must contain document_id")
        key = normalize_document_id(document_id)
        if key not in seen:
            seen.add(key)
            output.append(key)
    return output


def validate_selected_gold_document_scope(
    indexed_questions: Mapping[str, Mapping[str, Any]],
    selected_ids: Sequence[str],
    document_ids: Sequence[str] | str | None,
) -> None:
    """Require every selected gold document to lie inside the retrieval scope."""
    scope = normalize_document_scope(document_ids)
    if not scope:
        return
    normalized_scope = {normalize_document_id(value) for value in scope}
    violations: list[dict[str, Any]] = []
    for question_id in selected_ids:
        docs = question_gold_document_ids(indexed_questions[question_id])
        outside = sorted(set(docs) - normalized_scope)
        if outside:
            violations.append(
                {
                    "question_id": question_id,
                    "gold_document_ids": docs,
                    "outside_scope": outside,
                }
            )
    if violations:
        raise ValueError(
            "Selected dataset contains gold documents outside the configured "
            f"KG document scope: {violations[:10]!r}"
        )


def result_document_id(result: Any) -> str:
    """Read a document id from a KG result or result-like mapping."""
    if isinstance(result, Mapping):
        value = result.get("document_id")
        if value is None and isinstance(result.get("section"), Mapping):
            value = result["section"].get("document_id")
    else:
        value = getattr(result, "document_id", None)
        if value is None:
            section = getattr(result, "section", None)
            value = getattr(section, "document_id", None) if section is not None else None
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"KG result has no document_id: {result!r}")
    return text


def validate_result_document_scope(
    results: Sequence[Any],
    document_ids: Sequence[str] | str | None,
    *,
    stage: str,
) -> None:
    """Fail if any retrieved Section belongs to a document outside the scope."""
    scope = normalize_document_scope(document_ids)
    if not scope:
        return
    normalized_scope = {normalize_document_id(value) for value in scope}
    outside: list[tuple[str, str]] = []
    for result in results:
        doc_id = result_document_id(result)
        if normalize_document_id(doc_id) not in normalized_scope:
            uid = ""
            if isinstance(result, Mapping):
                uid = str(result.get("section_uid") or "")
                if not uid and isinstance(result.get("section"), Mapping):
                    uid = str(result["section"].get("section_uid") or "")
            else:
                uid = str(getattr(result, "section_uid", "") or "")
            outside.append((doc_id, uid))
    if outside:
        raise RuntimeError(
            f"KG document-scope leak at stage={stage!r}: "
            f"allowed={scope!r}, outside={outside[:20]!r}"
        )


def manifest_document_scope(manifest: Mapping[str, Any]) -> list[str]:
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        return []
    return normalize_document_scope(configuration.get("document_filtering"))


def load_run_document_scope(run_dir: str | Path) -> list[str]:
    path = Path(run_dir).expanduser().resolve() / "manifest.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid KG run manifest: {path}")
    return manifest_document_scope(payload)
