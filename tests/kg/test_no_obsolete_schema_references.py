from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KG_ROOT = ROOT / "src" / "agentic_rag" / "kg"

OBSOLETE_ENTITY_TYPES = (
    "risk_factor",
    "imaging_modality",
    "complication_or_comorbidity",
)


def test_updated_schema_file_is_gone() -> None:
    assert not (
        KG_ROOT / "updated_schema.py"
    ).exists()


def test_kg_source_has_no_obsolete_entity_type_tokens() -> None:
    offenders: list[str] = []

    for path in sorted(KG_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")

        found = [
            token
            for token in OBSOLETE_ENTITY_TYPES
            if token in text
        ]

        if found:
            offenders.append(
                f"{path.relative_to(ROOT)}: {', '.join(found)}"
            )

    assert not offenders, (
        "Obsolete entity-schema tokens remain:\n"
        + "\n".join(offenders)
    )


def test_router_prompts_do_not_describe_old_categories() -> None:
    prompt_paths = (
        ROOT
        / "src"
        / "agentic_rag"
        / "agent"
        / "prompts"
        / "kg_retrieval_router.yaml",
        ROOT
        / "src"
        / "agentic_rag"
        / "agent"
        / "prompts"
        / "kg_mentions_router.yaml",
    )

    obsolete_phrases = (
        "complications and risk factors",
        "imaging modalities",
    )

    offenders: list[str] = []

    for path in prompt_paths:
        text = path.read_text(encoding="utf-8").casefold()

        found = [
            phrase
            for phrase in obsolete_phrases
            if phrase in text
        ]

        if found:
            offenders.append(
                f"{path.relative_to(ROOT)}: {', '.join(found)}"
            )

    assert not offenders, (
        "Router prompts still describe obsolete categories:\n"
        + "\n".join(offenders)
    )
