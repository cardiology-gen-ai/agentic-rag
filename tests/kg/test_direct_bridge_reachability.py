from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_direct_bridge_reachability.py"
)
spec = importlib.util.spec_from_file_location(
    "analyze_direct_bridge_reachability",
    SCRIPT,
)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_classify_new_gold_partitions_union():
    direct = {("d", "1"), ("d", "2")}
    bridge = {("d", "2"), ("d", "3")}
    out = mod.classify_new_gold(direct, bridge)
    assert out["direct_only"] == {("d", "1")}
    assert out["bridge_only"] == {("d", "3")}
    assert out["overlap"] == {("d", "2")}
    assert out["union"] == {("d", "1"), ("d", "2"), ("d", "3")}


def test_main_configuration_is_frozen_source_aware():
    assert mod.DIRECT_PROFILE == "balanced"
    assert mod.DIRECT_SOURCES == ("SNOMEDCT_US", "NCI", "OMIM")
    assert mod.BRIDGE_SOURCE_PROFILES == {
        "SNOMEDCT_US": "strong",
        "NCI": "balanced",
        "OMIM": "balanced",
    }


def test_section_key_is_case_insensitive_and_trimmed():
    assert mod.section_key(" Doc ", " 7.1 ") == ("doc", "7.1")


def test_default_top_n_contains_controlled_budgets():
    assert mod.DEFAULT_TOP_N == (1, 3, 5, 10)
