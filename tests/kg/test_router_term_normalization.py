from agentic_rag.agent.output import KGMentionsPlan
from agentic_rag.kg.query_normalization import normalize_mentions_plan


def test_safe_v1_repairs_known_serialization_delimiter() -> None:
    plan = KGMentionsPlan(
        terms=["pulmonary valve disease」「RASopathy"],
        require_all=False,
    )
    result = normalize_mentions_plan(plan, mode="safe_v1")
    assert result.plan.terms == ["pulmonary valve disease", "RASopathy"]
    assert result.changed is True
    assert result.split_count == 1
    assert result.overflow_avoided is False


def test_safe_v1_preserves_clinical_semantics() -> None:
    plan = KGMentionsPlan(
        terms=["HER2-targeted therapy", "non-obstructive HCM", "do not use"],
        require_all=False,
    )
    result = normalize_mentions_plan(plan, mode="safe_v1")
    assert result.plan.terms == plan.terms
    assert result.changed is False


def test_safe_v1_normalizes_unicode_dash_and_whitespace_only() -> None:
    plan = KGMentionsPlan(terms=["  cardio\u2013oncology   service  "])
    result = normalize_mentions_plan(plan, mode="safe_v1")
    assert result.plan.terms == ["cardio-oncology service"]


def test_safe_v1_does_not_silently_exceed_five_term_contract() -> None:
    plan = KGMentionsPlan(terms=["a」「b", "c", "d", "e", "f"])
    result = normalize_mentions_plan(plan, mode="safe_v1")
    assert result.overflow_avoided is True
    assert result.plan.terms == ["a」「b", "c", "d", "e", "f"]
