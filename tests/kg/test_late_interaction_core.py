import numpy as np

from agentic_rag.kg.late_interaction import (
    concept_maxsim_score,
    rank_by_concept_maxsim,
)


def test_concept_maxsim_score_mean_of_per_term_maxima() -> None:
    term_map = {
        "alpha": np.asarray([1.0, 0.0], dtype=np.float32),
        "beta": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    concept_map = {
        "x": ("X", np.asarray([1.0, 0.0], dtype=np.float32)),
        "y": ("Y", np.asarray([0.0, 0.5], dtype=np.float32)),
    }
    score, evidence = concept_maxsim_score(
        ["alpha", "beta"], ["X", "Y"], term_map, concept_map
    )
    assert score == 0.75
    assert evidence[0]["best_concept"] == "X"
    assert evidence[1]["best_concept"] == "Y"


def test_concept_maxsim_empty_section_is_negative_one() -> None:
    score, evidence = concept_maxsim_score(
        ["alpha"],
        [],
        {"alpha": np.asarray([1.0], dtype=np.float32)},
        {},
    )
    assert score == -1.0
    assert evidence == [
        {"query_term": "alpha", "best_concept": None, "max_similarity": -1.0}
    ]


def test_rank_by_concept_maxsim_preserves_input_order_on_score_tie() -> None:
    universe = [{"uid": "b"}, {"uid": "a"}]
    ranked = rank_by_concept_maxsim(
        universe,
        {"a": 0.5, "b": 0.5},
        lambda row: row["uid"],
    )
    assert [row["uid"] for row in ranked] == ["b", "a"]
