"""Declarative BM25Plus build-plan contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from agentic_rag.utils.bm25 import sha256_file

PLAN_SCHEMA = "bm25_build_plan_v1"
MANIFEST_SCHEMA = "bm25_index_build_v2"


@dataclass(frozen=True)
class BM25BuildSpec:
    plan_path: Path
    plan_sha256: str
    corpus_id: str
    representation: str
    source_config: Path
    source_app_id: str
    target_app_id: str
    target_folder: Path
    target_index_name: str
    expected_count: int
    expected_source_type: str
    expected_documents: dict[str, int]

    @property
    def artifact_path(self) -> Path:
        return self.target_folder / f"{self.target_index_name}_bm25.pkl"

    @property
    def manifest_path(self) -> Path:
        return self.target_folder / "bm25_manifest.json"


def load_build_plan(path: str | Path) -> tuple[BM25BuildSpec, ...]:
    plan_path = Path(path).resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"Expected schema_version={PLAN_SCHEMA!r}")

    corpus_id = _required_string(payload, "corpus_id")
    source_config = _resolve(
        plan_path,
        _required_string(payload, "source_config"),
    )
    if not source_config.is_file():
        raise FileNotFoundError(f"Source config not found: {source_config}")

    builds = payload.get("builds")
    if not isinstance(builds, list) or not builds:
        raise ValueError("Build plan needs a non-empty builds list")

    seen_representations: set[str] = set()
    seen_target_folders: set[Path] = set()
    plan_sha = sha256_file(plan_path)
    specs: list[BM25BuildSpec] = []

    for build in builds:
        if not isinstance(build, Mapping):
            raise TypeError("Every build entry must be an object")

        representation = _required_string(build, "representation")
        if representation in seen_representations:
            raise ValueError(f"Duplicate representation: {representation}")
        seen_representations.add(representation)

        expected_count = _positive_int(build, "expected_count")
        expected_documents_raw = build.get("expected_documents")
        if not isinstance(expected_documents_raw, Mapping):
            raise ValueError(f"{representation}: missing expected_documents")
        expected_documents = {
            str(k): int(v) for k, v in expected_documents_raw.items()
        }
        if sum(expected_documents.values()) != expected_count:
            raise ValueError(
                f"{representation}: per-document counts do not sum "
                f"to expected_count={expected_count}"
            )

        target = build.get("target")
        if not isinstance(target, Mapping):
            raise ValueError(f"{representation}: missing target")
        target_folder = _resolve(
            plan_path,
            _required_string(target, "folder"),
        )
        if target_folder in seen_target_folders:
            raise ValueError(f"Duplicate target folder: {target_folder}")
        seen_target_folders.add(target_folder)

        specs.append(
            BM25BuildSpec(
                plan_path=plan_path,
                plan_sha256=plan_sha,
                corpus_id=corpus_id,
                representation=representation,
                source_config=source_config,
                source_app_id=_required_string(build, "source_app_id"),
                target_app_id=_required_string(build, "target_app_id"),
                target_folder=target_folder,
                target_index_name=_required_string(target, "index_name"),
                expected_count=expected_count,
                expected_source_type=_required_string(
                    build, "expected_source_type"
                ),
                expected_documents=dict(sorted(expected_documents.items())),
            )
        )

    return tuple(specs)


def load_source_app(spec: BM25BuildSpec) -> dict[str, Any]:
    payload = json.loads(spec.source_config.read_text(encoding="utf-8"))
    app = payload.get(spec.source_app_id)
    if not isinstance(app, dict):
        raise KeyError(f"Source app not found: {spec.source_app_id}")
    indexing = app.get("indexing")
    if not isinstance(indexing, dict) or indexing.get("type") != "faiss":
        raise ValueError(f"{spec.source_app_id} is not a FAISS app")
    return app


def source_index_contract(
    spec: BM25BuildSpec,
    app: Mapping[str, Any],
) -> tuple[Path, str]:
    indexing = app["indexing"]
    folder = _required_string(indexing, "folder")
    name = _required_string(indexing, "name")
    return (spec.source_config.parent / folder).resolve(), name


def validate_source_run_contract(
    spec: BM25BuildSpec,
    app: Mapping[str, Any],
) -> None:
    run = app.get("run")
    if not isinstance(run, Mapping):
        raise ValueError(f"{spec.source_app_id} has no run object")

    if run.get("mode") == "prebuilt_multi":
        total = run.get("expected_total_chunk_count")
        if total is None or int(total) != spec.expected_count:
            raise ValueError(
                f"{spec.source_app_id}: expected_total_chunk_count "
                "does not match plan"
            )
        sources = run.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{spec.source_app_id}: missing sources[]")
        types = {str(s.get("source_type")) for s in sources}
        if types != {spec.expected_source_type}:
            raise ValueError(
                f"{spec.source_app_id}: source types {sorted(types)} "
                f"!= {[spec.expected_source_type]}"
            )
        counts = [int(s["expected_chunk_count"]) for s in sources]
        if sum(counts) != spec.expected_count:
            raise ValueError(
                f"{spec.source_app_id}: source counts do not sum to plan"
            )
        return

    count = run.get("expected_chunk_count")
    if count is not None and int(count) != spec.expected_count:
        raise ValueError(
            f"{spec.source_app_id}: expected_chunk_count does not match plan"
        )
    source_type = run.get("source_type")
    if source_type is not None and str(source_type) != spec.expected_source_type:
        raise ValueError(
            f"{spec.source_app_id}: source_type does not match plan"
        )


def _resolve(plan_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (plan_path.parent / path).resolve()


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = int(mapping.get(key))
    if value < 1:
        raise ValueError(f"{key} must be positive")
    return value
