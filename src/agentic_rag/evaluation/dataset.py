"""Load canonical retrieval evaluation datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agentic_rag.evaluation.evidence import EvidenceSection


_SECTION_LABEL = re.compile(
    r"^\s*(?P<section_id>\d+(?:\.\d+)*)\.\s+.+$"
)

_ALLOWED_GROUPS = frozenset(
    {
        "single_section",
        "multi_section",
        "reasoning_multi_hop",
        "graph_hop_path_verified",
        # Backward-compatible legacy aggregate labels:
        "standard",
        "multi_hop",
    }
)


@dataclass(frozen=True)
class EvaluationQuestion:
    """One retrieval-evaluation question and its gold section set."""

    question_id: str
    question: str
    group: str
    complexity: str
    gold_sections: frozenset[EvidenceSection]

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.group not in _ALLOWED_GROUPS:
            raise ValueError(
                "group must be one of "
                f"{sorted(_ALLOWED_GROUPS)!r}; got {self.group!r}"
            )
        if not self.gold_sections:
            raise ValueError("gold_sections must not be empty")


def load_evaluation_questions(
    path: str | Path,
) -> tuple[EvaluationQuestion, ...]:
    """Load a canonical JSON dataset containing ``questions``."""

    dataset_path = Path(path).resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))

    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError(
            "Dataset must contain a non-empty 'questions' list"
        )

    questions = tuple(
        _parse_question(item)
        for item in raw_questions
    )

    ids = [item.question_id for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset contains duplicate question ids")

    declared_total = (
        payload.get("metadata", {}).get("total_questions")
        if isinstance(payload.get("metadata"), Mapping)
        else None
    )

    if declared_total is not None and declared_total != len(questions):
        raise ValueError(
            "metadata.total_questions does not match questions length: "
            f"{declared_total} != {len(questions)}"
        )

    return questions


def parse_section_id(section_label: str) -> str:
    """Extract a numeric section identifier from a canonical label."""

    if not isinstance(section_label, str):
        raise TypeError("section label must be a string")

    match = _SECTION_LABEL.match(section_label)
    if match is None:
        raise ValueError(
            "Section label must start with '<numeric id>. <title>': "
            f"{section_label!r}"
        )

    return match.group("section_id")


def _parse_question(raw: Any) -> EvaluationQuestion:
    if not isinstance(raw, Mapping):
        raise TypeError("Every dataset question must be an object")

    question_id = _required_string(raw, "id")
    question = _required_string(raw, "question")
    complexity = _required_string(raw, "complexity")

    metadata = raw.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"Question {question_id!r} must contain metadata"
        )

    group = _resolve_evaluation_group(metadata)

    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            f"Question {question_id!r} must contain sources"
        )

    gold: set[EvidenceSection] = set()

    for source in sources:
        if not isinstance(source, Mapping):
            raise TypeError(
                f"Question {question_id!r} contains an invalid source"
            )

        document_id = _required_string(source, "document_id")
        sections = source.get("sections")

        if not isinstance(sections, list) or not sections:
            raise ValueError(
                f"Question {question_id!r} source {document_id!r} "
                "must contain sections"
            )

        for section_label in sections:
            gold.add(
                EvidenceSection(
                    document_id=document_id,
                    section_id=parse_section_id(section_label),
                )
            )

    return EvaluationQuestion(
        question_id=question_id,
        question=question,
        group=group,
        complexity=complexity,
        gold_sections=frozenset(gold),
    )


def _resolve_evaluation_group(
    metadata: Mapping[str, Any],
) -> str:
    explicit = metadata.get("evaluation_group")
    if isinstance(explicit, str) and explicit.strip():
        group = explicit.strip()
        if group not in _ALLOWED_GROUPS:
            raise ValueError(
                "Unsupported metadata.evaluation_group "
                f"{group!r}; expected one of "
                f"{sorted(_ALLOWED_GROUPS)!r}"
            )
        return group

    # Backward compatibility with schema <= 2.x.
    question_type = _required_string(metadata, "question_type")
    return (
        "multi_hop"
        if question_type.startswith("multi_hop")
        else "standard"
    )


def _required_string(
    mapping: Mapping[str, Any],
    key: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty string field {key!r}")
    return value.strip()
