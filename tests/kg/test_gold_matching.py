"""Tests for exact gold-to-KG section matching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.evaluation.retrieval import (
    SectionKey,
    deduplicate_section_keys,
    normalize_document_id,
    normalize_printed_section_id,
    normalize_section_title,
    parse_gold_section_label,
    section_key_from_gold,
    section_key_from_result,
    sections_match,
)
from agentic_rag.kg.models import KGSectionResult


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "subset_test_en_cm_syn.json"
)


def make_result(
    *,
    document_id: str = "Cardiomyopathies_2023",
    section_uid: str = "Cardiomyopathies_2023::7.1.1.1",
    section_id: str = "7.1.1.1",
    printed_section_id: str | None = "7.1.1.1",
    title: str | None = "Diagnostic criteria",
) -> KGSectionResult:
    return KGSectionResult(
        section_uid=section_uid,
        document_id=document_id,
        section_id=section_id,
        printed_section_id=printed_section_id,
        title=title,
        text="Example section text.",
    )


def test_parse_gold_section_label() -> None:
    section_id, title = parse_gold_section_label(
        "7.1.1.1. Diagnostic criteria"
    )

    assert section_id == "7.1.1.1"
    assert title == "diagnostic criteria"


def test_gold_matches_retrieved_section() -> None:
    gold = section_key_from_gold(
        "Cardiomyopathies_2023",
        "7.1.1.1. Diagnostic criteria",
    )
    retrieved = section_key_from_result(make_result())

    assert gold == retrieved
    assert sections_match(
        "Cardiomyopathies_2023",
        "7.1.1.1. Diagnostic criteria",
        make_result(),
    )


def test_document_extension_and_case_are_ignored() -> None:
    assert normalize_document_id(
        " Cardiomyopathies_2023.MD "
    ) == "cardiomyopathies_2023"
    assert normalize_document_id(
        "Cardiomyopathies_2023.pdf"
    ) == "cardiomyopathies_2023"


def test_same_section_in_different_document_does_not_match() -> None:
    gold = section_key_from_gold(
        "Cardiomyopathies_2023",
        "7.1.1.1. Diagnostic criteria",
    )
    retrieved = section_key_from_result(
        make_result(
            document_id="Syncope_2018",
            section_uid="Syncope_2018::7.1.1.1",
        )
    )

    assert gold != retrieved


def test_different_printed_section_id_does_not_match() -> None:
    gold = section_key_from_gold(
        "Cardiomyopathies_2023",
        "7.1.1.1. Diagnostic criteria",
    )
    retrieved = section_key_from_result(
        make_result(
            section_uid="Cardiomyopathies_2023::7.1.1.2",
            section_id="7.1.1.2",
            printed_section_id="7.1.1.2",
        )
    )

    assert gold != retrieved


def test_different_title_does_not_match() -> None:
    gold = section_key_from_gold(
        "Cardiomyopathies_2023",
        "7.1.1.1. Diagnostic criteria",
    )
    retrieved = section_key_from_result(
        make_result(title="Management")
    )

    assert gold != retrieved


def test_typographic_variations_are_normalized() -> None:
    assert normalize_section_title(
        "  Diagnostic   criteria. "
    ) == "diagnostic criteria"
    assert normalize_section_title(
        "Pre–pregnancy"
    ) == "pre-pregnancy"
    assert normalize_section_title(
        "Patients’ assessment"
    ) == "patients' assessment"
    assert normalize_printed_section_id(" 7.1.1.1. ") == "7.1.1.1"


def test_result_requires_printed_section_id() -> None:
    with pytest.raises(ValueError, match="printed_section_id"):
        section_key_from_result(
            make_result(printed_section_id=None)
        )


def test_result_requires_title() -> None:
    with pytest.raises(ValueError, match="no title"):
        section_key_from_result(make_result(title=None))


def test_invalid_gold_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid gold section label"):
        parse_gold_section_label("Diagnostic criteria")


def test_deduplication_preserves_first_occurrence() -> None:
    first = SectionKey("doc", "1", "first")
    second = SectionKey("doc", "2", "second")

    assert deduplicate_section_keys(
        [first, first, second, first]
    ) == [first, second]


def test_all_subset_gold_labels_are_parseable() -> None:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    parsed_count = 0
    for question in payload["questions"]:
        for source in question["sources"]:
            for section_label in source["sections"]:
                key = section_key_from_gold(
                    source["document_id"],
                    section_label,
                )
                assert key.document_id
                assert key.printed_section_id
                assert key.title
                parsed_count += 1

    assert parsed_count > 0
