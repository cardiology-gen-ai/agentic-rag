"""Plan-driven retrieval matrix evaluation for multi-document corpora."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cardiology_gen_ai import IndexingConfig
from langchain_core.documents import Document

from agentic_rag.config.manager import SearchConfig
from agentic_rag.managers.search_manager import SearchManager
from agentic_rag.utils.bm25 import sha256_file
from agentic_rag.utils.bm25_build_plan import (
    BM25BuildSpec,
    load_build_plan as load_bm25_build_plan,
    validate_source_run_contract,
)


PLAN_SCHEMA = "retrieval_eval_plan_v1"
ALLOWED_METHODS = (
    "dense",
    "bm25_plus",
    "dense_bm25plus_rerank",
    "hybrid_rrf",
)


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    dense_app_id: str
    bm25_representation: str
    source_type: str
    expected_count: int
    expected_documents: dict[str, int]
    bm25_spec: BM25BuildSpec


@dataclass(frozen=True)
class RetrievalEvalPlan:
    path: Path
    sha256: str
    corpus_id: str
    data_etl_config: Path
    bm25_build_plan: Path
    candidate_k: tuple[int, ...]
    cutoffs: tuple[int, ...]
    methods: tuple[str, ...]
    rrf_k: int
    dense_weight: float
    bm25_weight: float
    representations: tuple[RepresentationSpec, ...]
    artifact_freeze: Path | None


def load_retrieval_eval_plan(path: str | Path) -> RetrievalEvalPlan:
    plan_path = Path(path).resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"Expected schema_version={PLAN_SCHEMA!r}")

    corpus_id = _required_string(payload, "corpus_id")
    data_etl_config = _resolve(
        plan_path, _required_string(payload, "data_etl_config")
    )
    bm25_build_plan = _resolve(
        plan_path, _required_string(payload, "bm25_build_plan")
    )
    if not data_etl_config.is_file():
        raise FileNotFoundError(f"data-etl config not found: {data_etl_config}")
    if not bm25_build_plan.is_file():
        raise FileNotFoundError(f"BM25 build plan not found: {bm25_build_plan}")

    candidate_k = _positive_increasing_ints(payload.get("candidate_k"), "candidate_k")
    cutoffs = _positive_increasing_ints(payload.get("cutoffs"), "cutoffs")
    if min(candidate_k) < max(cutoffs):
        raise ValueError("Every candidate_k must cover the largest cutoff")

    methods_raw = payload.get("methods", list(ALLOWED_METHODS))
    if not isinstance(methods_raw, list) or not methods_raw:
        raise ValueError("methods must be a non-empty list")
    methods = tuple(str(item) for item in methods_raw)
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not contain duplicates")
    unknown_methods = set(methods) - set(ALLOWED_METHODS)
    if unknown_methods:
        raise ValueError(f"Unsupported methods: {sorted(unknown_methods)!r}")

    rrf = payload.get("rrf", {})
    if not isinstance(rrf, Mapping):
        raise ValueError("rrf must be an object")
    rrf_k = int(rrf.get("rrf_k", 60))
    dense_weight = float(rrf.get("dense_weight", 1.0))
    bm25_weight = float(rrf.get("bm25_weight", 1.0))
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    if dense_weight <= 0 or bm25_weight <= 0:
        raise ValueError("RRF weights must be > 0")

    dense_config = json.loads(data_etl_config.read_text(encoding="utf-8"))
    bm25_specs = {
        spec.representation: spec
        for spec in load_bm25_build_plan(bm25_build_plan)
    }

    raw_representations = payload.get("representations")
    if not isinstance(raw_representations, list) or not raw_representations:
        raise ValueError("representations must be a non-empty list")

    seen: set[str] = set()
    representations: list[RepresentationSpec] = []
    for raw in raw_representations:
        if not isinstance(raw, Mapping):
            raise TypeError("Every representation must be an object")
        name = _required_string(raw, "name")
        if name in seen:
            raise ValueError(f"Duplicate representation name: {name!r}")
        seen.add(name)
        dense_app_id = _required_string(raw, "dense_app_id")
        bm25_representation = _required_string(raw, "bm25_representation")

        if dense_app_id not in dense_config:
            raise KeyError(f"Dense app not found: {dense_app_id!r}")
        if bm25_representation not in bm25_specs:
            raise KeyError(
                f"BM25 representation not found: {bm25_representation!r}"
            )
        bm25_spec = bm25_specs[bm25_representation]
        if bm25_spec.source_app_id != dense_app_id:
            raise ValueError(
                f"{name}: dense app {dense_app_id!r} does not match BM25 "
                f"source app {bm25_spec.source_app_id!r}"
            )

        dense_app = dense_config[dense_app_id]
        validate_source_run_contract(bm25_spec, dense_app)

        representations.append(
            RepresentationSpec(
                name=name,
                dense_app_id=dense_app_id,
                bm25_representation=bm25_representation,
                source_type=bm25_spec.expected_source_type,
                expected_count=bm25_spec.expected_count,
                expected_documents=dict(bm25_spec.expected_documents),
                bm25_spec=bm25_spec,
            )
        )

    artifact_freeze_raw = payload.get("artifact_freeze")
    artifact_freeze = (
        _resolve(plan_path, str(artifact_freeze_raw))
        if isinstance(artifact_freeze_raw, str) and artifact_freeze_raw.strip()
        else None
    )
    if artifact_freeze is not None and not artifact_freeze.is_file():
        raise FileNotFoundError(
            f"artifact_freeze not found: {artifact_freeze}"
        )

    return RetrievalEvalPlan(
        path=plan_path,
        sha256=sha256_file(plan_path),
        corpus_id=corpus_id,
        data_etl_config=data_etl_config,
        bm25_build_plan=bm25_build_plan,
        candidate_k=candidate_k,
        cutoffs=cutoffs,
        methods=methods,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        representations=tuple(representations),
        artifact_freeze=artifact_freeze,
    )


def build_dense_manager(
    plan: RetrievalEvalPlan,
    representation: RepresentationSpec,
    *,
    candidate_k: int,
) -> SearchManager:
    payload = json.loads(plan.data_etl_config.read_text(encoding="utf-8"))
    app = payload[representation.dense_app_id]
    indexing = dict(app["indexing"])
    indexing["folder"] = str(
        (plan.data_etl_config.parent / indexing["folder"]).resolve()
    )
    embeddings = app.get("embeddings")
    if embeddings is not None:
        indexing["embeddings"] = embeddings
    return _build_manager(indexing, candidate_k=candidate_k)


def build_bm25_manager(
    representation: RepresentationSpec,
    *,
    candidate_k: int,
) -> SearchManager:
    spec = representation.bm25_spec
    _validate_bm25_manifest(spec)
    indexing = {
        "name": spec.target_index_name,
        "description": (
            f"BM25Plus {representation.name} for {spec.corpus_id}"
        ),
        "type": "bm25",
        "retrieval_mode": "sparse",
        "folder": str(spec.target_folder),
    }
    return _build_manager(indexing, candidate_k=candidate_k)


def _build_manager(
    indexing: Mapping[str, Any],
    *,
    candidate_k: int,
) -> SearchManager:
    return SearchManager(
        index_config=IndexingConfig.from_config(dict(indexing)),
        search_config=SearchConfig.from_config(
            {
                "type": "similarity",
                "k": candidate_k,
                "top_k": candidate_k,
            }
        ),
    )


def search(manager: SearchManager, query: str) -> tuple[Document, ...]:
    return tuple(manager.search(query).chunks)


def document_identity(document: Document) -> tuple[str, str]:
    metadata = document.metadata
    document_id = None
    for key in ("doc_id", "document_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            document_id = value.strip()
            break
    if document_id is None:
        raise ValueError("Retrieved document has no doc_id/document_id")

    local_id = None
    for key in (
        "retrieval_unit_key",
        "retrieval_unit_id",
        "record_id",
        "chunk_id",
        "id",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            local_id = value.strip()
            break
    if local_id is None:
        value = getattr(document, "id", None)
        if value is not None and str(value).strip():
            local_id = str(value).strip()
    if local_id is None:
        raise ValueError("Retrieved document has no stable local identity")
    return document_id, local_id


def assert_same_membership(
    before: Sequence[Document],
    after: Sequence[Document],
) -> None:
    before_ids = [document_identity(item) for item in before]
    after_ids = [document_identity(item) for item in after]
    if len(before_ids) != len(after_ids):
        raise AssertionError("Candidate-pool cardinality changed")
    if len(before_ids) != len(set(before_ids)):
        raise AssertionError("Original candidate pool contains duplicates")
    if len(after_ids) != len(set(after_ids)):
        raise AssertionError("Result candidate pool contains duplicates")
    if set(before_ids) != set(after_ids):
        raise AssertionError("Candidate-pool membership changed")


def provenance_files(plan: RetrievalEvalPlan) -> dict[str, Any]:
    result: dict[str, Any] = {
        "retrieval_plan": {
            "path": str(plan.path),
            "sha256": plan.sha256,
        },
        "data_etl_config": {
            "path": str(plan.data_etl_config),
            "sha256": sha256_file(plan.data_etl_config),
        },
        "bm25_build_plan": {
            "path": str(plan.bm25_build_plan),
            "sha256": sha256_file(plan.bm25_build_plan),
        },
    }
    if plan.artifact_freeze is not None:
        result["artifact_freeze"] = {
            "path": str(plan.artifact_freeze),
            "sha256": sha256_file(plan.artifact_freeze),
        }

    representations: dict[str, Any] = {}
    dense_config = json.loads(plan.data_etl_config.read_text(encoding="utf-8"))
    for rep in plan.representations:
        dense_indexing = dense_config[rep.dense_app_id]["indexing"]
        dense_folder = (
            plan.data_etl_config.parent / dense_indexing["folder"]
        ).resolve()
        dense_manifest = dense_folder / "build_manifest.json"
        bm25_manifest = rep.bm25_spec.manifest_path
        representations[rep.name] = {
            "dense_app_id": rep.dense_app_id,
            "dense_build_manifest": (
                {
                    "path": str(dense_manifest),
                    "sha256": sha256_file(dense_manifest),
                }
                if dense_manifest.is_file()
                else None
            ),
            "bm25_representation": rep.bm25_representation,
            "bm25_manifest": {
                "path": str(bm25_manifest),
                "sha256": sha256_file(bm25_manifest),
            },
            "expected_count": rep.expected_count,
            "expected_documents": rep.expected_documents,
            "source_type": rep.source_type,
        }
    result["representations"] = representations
    return result


def _validate_bm25_manifest(spec: BM25BuildSpec) -> None:
    if not spec.artifact_path.is_file():
        raise FileNotFoundError(f"BM25 artifact missing: {spec.artifact_path}")
    if not spec.manifest_path.is_file():
        raise FileNotFoundError(f"BM25 manifest missing: {spec.manifest_path}")
    manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("corpus_id") != spec.corpus_id:
        raise ValueError("BM25 manifest corpus_id mismatch")
    if manifest.get("representation") != spec.representation:
        raise ValueError("BM25 manifest representation mismatch")
    if manifest.get("source_app_id") != spec.source_app_id:
        raise ValueError("BM25 manifest source_app_id mismatch")
    if manifest.get("target_index_name") != spec.target_index_name:
        raise ValueError("BM25 manifest target index mismatch")
    if manifest.get("artifact_sha256") != sha256_file(spec.artifact_path):
        raise ValueError("BM25 artifact SHA256 mismatch")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("BM25 manifest has no validation object")
    if int(validation.get("document_count", -1)) != spec.expected_count:
        raise ValueError("BM25 manifest document_count mismatch")
    if validation.get("documents") != spec.expected_documents:
        raise ValueError("BM25 manifest per-document counts mismatch")
    if int(validation.get("document_scoped_identity_count", -1)) != spec.expected_count:
        raise ValueError("BM25 manifest scoped identity count mismatch")
    if int(validation.get("empty_text_count", -1)) != 0:
        raise ValueError("BM25 manifest reports empty documents")


def _resolve(plan_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (plan_path.parent / path).resolve()


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _positive_increasing_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    items = tuple(int(item) for item in value)
    if tuple(sorted(set(items))) != items or any(item < 1 for item in items):
        raise ValueError(f"{name} must contain unique increasing positive integers")
    return items
