from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_connection_ablation_from_config.py"


def load_module():
    spec = importlib.util.spec_from_file_location("connection_runner_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_connection_runner_propagates_document_ids(tmp_path, monkeypatch):
    module = load_module()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "schema_version": "kg_connection_ablation_v1",
                "experiment": {
                    "run_id": "two_doc",
                    "dataset": "dataset.json",
                    "coverage_artifact": "coverage.json",
                    "mentions_plans_file": "plans.jsonl",
                    "candidate_k": 50,
                    "top_k": 20,
                    "output_root": "artifacts/kg_retrieval",
                    "modes": ["mentions_direct_balanced"],
                    "document_ids": [
                        "Cardiomyopathies_2023",
                        "Cardio-oncology_2022",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_call(cmd):
        captured["cmd"] = list(cmd)
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_connection_ablation_from_config.py", "--config", str(cfg)],
    )
    assert module.main() == 0
    cmd = captured["cmd"]
    pairs = list(zip(cmd, cmd[1:]))
    assert ("--document-id", "Cardiomyopathies_2023") in pairs
    assert ("--document-id", "Cardio-oncology_2022") in pairs
