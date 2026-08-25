from pathlib import Path


def test_pre_foundation_backups_exist_for_algorithmic_gate():
    paths = [
        Path("scripts/analyze_v11_specificity_ranking.py.before_kg_multidoc_ablation_foundation_v1"),
        Path("scripts/analyze_v13_r4a_structural_tail.py.before_kg_multidoc_ablation_foundation_v1"),
        Path("scripts/analyze_v14_r4c_best_channel.py.before_kg_multidoc_ablation_foundation_v1"),
    ]
    assert all(p.is_file() for p in paths)


def test_current_analyzers_exist_for_algorithmic_gate():
    paths = [
        Path("scripts/analyze_v11_specificity_ranking.py"),
        Path("scripts/analyze_v13_r4a_structural_tail.py"),
        Path("scripts/analyze_v14_r4c_best_channel.py"),
    ]
    assert all(p.is_file() for p in paths)


def test_algorithmic_gate_is_offline():
    text = Path("scripts/run_cm_algorithmic_equivalence_v1.py").read_text(encoding="utf-8")
    assert "run_connection_ablation_from_config.py" not in text
    assert "export_concept_specificity.py" not in text
    assert "export_v13_structural_snapshot.py" not in text
    assert "export_v14_r4c_snapshot.py" not in text
    assert "build_v14_li_embedding_artifact.py" not in text
    assert '"live_semantic_seeding": False' in text
