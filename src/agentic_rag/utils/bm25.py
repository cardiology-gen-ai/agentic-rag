"""BM25Plus index construction and deterministic ranked retrieval utilities."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cardiology_gen_ai import BM25Dict, BM25Vectorstore
from langchain_core.documents import Document
from rank_bm25 import BM25Plus


BM25_ALGORITHM = "BM25Plus"
BM25_TOKENIZER = "lowercase_regex_word_tokens"


def load_documents_from_faiss_pickle(
    path: str | Path,
) -> tuple[Document, ...]:
    """Load documents from a LangChain FAISS ``.pkl`` docstore artifact.

    LangChain stores ``(docstore, index_to_docstore_id)`` separately from the
    binary FAISS index. No embedding model is needed to read these documents.
    """

    pickle_path = Path(path).resolve()
    if not pickle_path.is_file():
        raise FileNotFoundError(f"FAISS docstore pickle not found: {pickle_path}")

    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, tuple) or len(payload) != 2:
        raise TypeError(
            "Expected a LangChain FAISS pickle containing "
            "(docstore, index_to_docstore_id)"
        )

    docstore, index_to_docstore_id = payload
    if not isinstance(index_to_docstore_id, Mapping):
        raise TypeError("FAISS index_to_docstore_id must be a mapping")

    ordered_keys = sorted(index_to_docstore_id, key=_sortable_index_key)
    documents: list[Document] = []

    for index_position in ordered_keys:
        docstore_id = index_to_docstore_id[index_position]
        document = _lookup_document(docstore, docstore_id)
        documents.append(_copy_document(document))

    if not documents:
        raise ValueError(f"FAISS docstore contains no documents: {pickle_path}")

    return tuple(documents)


def resolve_faiss_pickle(
    *,
    folder: str | Path,
    index_name: str,
) -> Path:
    """Resolve the docstore pickle for a configured FAISS index."""

    root = Path(folder).resolve()
    preferred = (
        root / f"{index_name}.pkl",
        root / "index.pkl",
    )

    for candidate in preferred:
        if candidate.is_file():
            return candidate

    candidates = sorted(
        path
        for path in root.glob("*.pkl")
        if not path.name.endswith("_bm25.pkl")
    )

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Could not resolve a unique FAISS docstore pickle in "
        f"{root}. Checked {[str(path) for path in preferred]!r}; "
        f"other candidates={list(map(str, candidates))!r}"
    )


def validate_bm25_documents(
    documents: Sequence[Document],
    *,
    expected_count: int | None = None,
    expected_source_type: str | None = None,
    expected_document_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate a BM25 corpus using document-scoped retrieval identities.

    ``record_id`` is only locally unique inside a guideline. Multi-document
    corpora therefore use ``(doc_id, record_id)`` as the required identity.
    When ``retrieval_unit_key`` is present, its global uniqueness is checked
    as an additional storage-level invariant.

    ``record_id_count`` is retained for backward-compatible manifests; in a
    valid multi-document corpus it can be smaller than ``document_count``.
    """

    if not documents:
        raise ValueError("Cannot build BM25Plus over an empty document list")

    if expected_count is not None and len(documents) != expected_count:
        raise ValueError(
            f"Document count mismatch: expected {expected_count}, "
            f"found {len(documents)}"
        )

    record_ids: set[str] = set()
    scoped_identities: set[tuple[str, str]] = set()
    retrieval_unit_keys: set[str] = set()
    document_counts: dict[str, int] = {}
    source_types: set[str] = set()
    empty_text_count = 0

    for position, document in enumerate(documents):
        if not isinstance(document, Document):
            raise TypeError(
                f"Item {position} is not a LangChain Document: "
                f"{type(document)!r}"
            )

        text = str(document.page_content or "")
        if not text.strip():
            empty_text_count += 1

        metadata = document.metadata
        record_id = _required_metadata_string(metadata, "record_id")
        record_ids.add(record_id)

        doc_id = _first_non_empty_string(
            metadata,
            ("doc_id", "document_id"),
        )
        if not doc_id:
            raise ValueError(
                f"Document {record_id!r} has no doc_id/document_id"
            )

        scoped_identity = (doc_id, record_id)
        if scoped_identity in scoped_identities:
            raise ValueError(
                "Duplicate document-scoped retrieval identity in corpus: "
                f"{scoped_identity!r}"
            )
        scoped_identities.add(scoped_identity)
        document_counts[doc_id] = document_counts.get(doc_id, 0) + 1

        retrieval_unit_key = _first_non_empty_string(
            metadata,
            ("retrieval_unit_key",),
        )
        if retrieval_unit_key:
            if retrieval_unit_key in retrieval_unit_keys:
                raise ValueError(
                    "Duplicate retrieval_unit_key in corpus: "
                    f"{retrieval_unit_key!r}"
                )
            retrieval_unit_keys.add(retrieval_unit_key)

        source_type = _required_metadata_string(
            metadata,
            "prebuilt_source_type",
        )
        source_types.add(source_type)

        if source_type == "fixed_chunks":
            if not _metadata_string_list(
                metadata,
                "source_section_ids",
                allow_fallback_key="section_id",
            ):
                raise ValueError(
                    f"Fixed document {record_id!r} has no source section ids"
                )
        elif source_type == "hierarchical_section_view":
            _required_metadata_string(metadata, "retrieval_unit_id")
            if not _metadata_string_list(
                metadata,
                "represented_section_ids",
                allow_fallback_key="source_section_ids",
            ):
                raise ValueError(
                    "Hierarchical document "
                    f"{record_id!r} has no represented section ids"
                )
        else:
            raise ValueError(
                f"Unsupported prebuilt_source_type {source_type!r} "
                f"for document {record_id!r}"
            )

    if expected_source_type is not None and source_types != {
        expected_source_type
    }:
        raise ValueError(
            "Source type mismatch: expected only "
            f"{expected_source_type!r}, found {sorted(source_types)!r}"
        )

    if expected_document_counts is not None:
        expected_docs = {
            str(doc_id): int(count)
            for doc_id, count in expected_document_counts.items()
        }
        if document_counts != expected_docs:
            raise ValueError(
                "Per-document count mismatch: expected "
                f"{dict(sorted(expected_docs.items()))!r}, found "
                f"{dict(sorted(document_counts.items()))!r}"
            )

    token_counts = [
        len(BM25Vectorstore.tokenize(str(document.page_content or "")))
        for document in documents
    ]
    if not any(token_counts):
        raise ValueError("Every document tokenizes to an empty sequence")

    return {
        "document_count": len(documents),
        "record_id_count": len(record_ids),
        "document_scoped_identity_count": len(scoped_identities),
        "retrieval_unit_key_count": len(retrieval_unit_keys),
        "documents": dict(sorted(document_counts.items())),
        "source_types": sorted(source_types),
        "empty_text_count": empty_text_count,
        "min_token_count": min(token_counts),
        "max_token_count": max(token_counts),
        "avg_token_count": sum(token_counts) / len(token_counts),
    }


def build_bm25_dict(documents: Sequence[Document]) -> BM25Dict:
    """Create the exact BM25Plus payload expected by ``BM25Vectorstore``."""

    corpus = [
        BM25Vectorstore.tokenize(str(document.page_content or ""))
        for document in documents
    ]
    if not corpus or not any(corpus):
        raise ValueError("Cannot initialize BM25Plus from an empty corpus")

    return BM25Dict(
        bm25=BM25Plus(corpus),
        documents=[_copy_document(document) for document in documents],
    )


def save_bm25_dict(
    artifact: BM25Dict,
    path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Atomically save a BM25 payload using the package's expected format."""

    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        raise FileExistsError(
            f"BM25 artifact already exists: {output_path}. Use --force."
        )

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)
    return output_path


def rank_bm25_documents(
    vectorstore: BM25Dict,
    query: str,
    *,
    k: int,
) -> list[Document]:
    """Return a stable BM25Plus ranking with query-time scores in metadata.

    Ties are resolved by original corpus position. Returned documents are
    copies, so query-specific scores never mutate the persistent corpus.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if vectorstore.bm25 is None:
        raise ValueError("BM25Dict has no BM25 model")
    if not vectorstore.documents:
        return []

    query_tokens = BM25Vectorstore.tokenize(query)
    query_token_set = set(query_tokens)
    scores = vectorstore.bm25.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )[: min(k, len(scores))]

    results: list[Document] = []
    for rank, index in enumerate(ranked_indices, start=1):
        document = vectorstore.documents[index]
        score = float(scores[index])
        document_tokens = set(
            BM25Vectorstore.tokenize(
                str(document.page_content or "")
            )
        )
        matched_query_tokens = sorted(
            query_token_set.intersection(document_tokens)
        )
        metadata = dict(document.metadata)
        metadata.update(
            {
                "raw_score": score,
                "retrieval_backend": "bm25",
                "retrieval_algorithm": BM25_ALGORITHM,
                "bm25_query_token_count": len(query_tokens),
                "bm25_query_token_overlap_count":
                    len(matched_query_tokens),
                "bm25_has_query_token_overlap":
                    bool(matched_query_tokens),
                "bm25_matched_query_tokens":
                    matched_query_tokens,
                "bm25_score_positive": score > 0.0,
                "bm25_corpus_position": index,
                "bm25_raw_rank": rank,
            }
        )
        results.append(_copy_document(document, metadata=metadata))

    return results


def document_fingerprint(documents: Iterable[Document]) -> str:
    """Return a stable SHA-256 over ordered text and metadata."""

    digest = hashlib.sha256()
    for document in documents:
        payload = {
            "page_content": str(document.page_content or ""),
            "metadata": _json_safe(document.metadata),
            "id": getattr(document, "id", None),
        }
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup_document(docstore: Any, docstore_id: Any) -> Document:
    if hasattr(docstore, "search"):
        result = docstore.search(docstore_id)
        if isinstance(result, Document):
            return result

    raw_mapping = getattr(docstore, "_dict", None)
    if isinstance(raw_mapping, Mapping):
        result = raw_mapping.get(docstore_id)
        if isinstance(result, Document):
            return result

    raise KeyError(
        f"Document {docstore_id!r} not found or not a LangChain Document"
    )


def _copy_document(
    document: Document,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Document:
    kwargs: dict[str, Any] = {
        "page_content": str(document.page_content or ""),
        "metadata": dict(metadata if metadata is not None else document.metadata),
    }
    document_id = getattr(document, "id", None)
    if document_id is not None:
        kwargs["id"] = document_id
    return Document(**kwargs)


def _sortable_index_key(value: Any) -> tuple[int, str]:
    if isinstance(value, int):
        return (0, str(value).zfill(20))
    try:
        return (0, str(int(value)).zfill(20))
    except (TypeError, ValueError):
        return (1, str(value))


def _required_metadata_string(
    metadata: Mapping[str, Any],
    key: str,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty metadata field {key!r}")
    return value.strip()


def _first_non_empty_string(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _metadata_string_list(
    metadata: Mapping[str, Any],
    key: str,
    *,
    allow_fallback_key: str | None = None,
) -> list[str]:
    value = metadata.get(key)
    if value is None and allow_fallback_key is not None:
        value = metadata.get(allow_fallback_key)
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set, frozenset)):
        value = [value]
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
