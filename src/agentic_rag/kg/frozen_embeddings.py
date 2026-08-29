"""Read-only loaders for frozen KG embedding artifacts used by rerankers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def resolve_single_concept_cache(path: Path) -> Path:
    """Resolve a dedicated Concept-cache directory to its single NPZ artifact."""

    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = sorted(resolved.glob("*.npz"))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one Concept embedding cache in {resolved}, "
            f"found {len(files)}"
        )
    return files[0]


def load_concept_embedding_map(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load exact-case Concept names and their already-normalized vectors."""

    cache_file = resolve_single_concept_cache(path)
    with np.load(cache_file, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        matrix = np.asarray(data["embeddings"], dtype=np.float32)

    names = [str(value) for value in metadata.get("concept_names") or []]
    if matrix.ndim != 2 or matrix.shape[0] != len(names):
        raise RuntimeError("Concept embedding cache metadata/matrix mismatch")

    output: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        if name in output:
            raise RuntimeError(f"Duplicate exact Concept identity in cache: {name!r}")
        output[name] = matrix[index]
    metadata = dict(metadata)
    metadata["resolved_cache_file"] = str(cache_file)
    return output, metadata


def load_query_term_embedding_map(
    path: Path,
    *,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int | None = None,
) -> dict[str, np.ndarray]:
    """Load and L2-normalize every compatible frozen query-term vector."""

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)

    output: dict[str, np.ndarray] = {}
    for cache_file in sorted(resolved.glob("*.npz")):
        with np.load(cache_file, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            vector = np.asarray(data["embedding"], dtype=np.float32)
        if metadata.get("schema_version") != "query_term_embedding_cache_v1":
            continue
        if metadata.get("embedding_provider") != embedding_provider:
            continue
        if metadata.get("embedding_model") != embedding_model:
            continue
        if metadata.get("embedding_dimensions") != embedding_dimensions:
            continue
        term = str(metadata.get("term") or "").strip()
        if not term:
            continue
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise RuntimeError(f"Invalid query-term vector shape in {cache_file}")
        normalized = _normalize_vector(vector[0])
        key = term.casefold()
        previous = output.get(key)
        if previous is not None and not np.array_equal(previous, normalized):
            raise RuntimeError(
                f"Conflicting frozen embeddings for casefold-equivalent term {term!r}"
            )
        output[key] = normalized

    if not output:
        raise RuntimeError(f"No compatible query-term embeddings found in {resolved}")
    return output


def load_section_concepts(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Load a gold-free Section→Concept snapshot."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    raw = payload.get("section_concepts")
    if not isinstance(raw, Mapping):
        raise ValueError("Section Concept snapshot has no section_concepts mapping")
    output: dict[str, list[str]] = {}
    for uid, values in raw.items():
        names = [str(value).strip() for value in values or [] if str(value).strip()]
        output[str(uid)] = names
    return output, payload


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return array
    return (array / norm).astype(np.float32)
