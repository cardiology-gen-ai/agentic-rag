import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "builder", ROOT/"scripts"/"build_nci_nonhier_ablation_artifacts.py"
)
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


def _pair(i, *, new=False, broad=False, extra=False):
    a=f"C{i:03d}A"; z=f"C{i:03d}B"
    assertions=[
        {"source_cui":a,"target_cui":z,"relation_name":"forward","semantic_status":"valid_but_broad" if broad else "valid","provenance":"safe63" if not new else "v8_specific_novel"},
        {"source_cui":z,"target_cui":a,"relation_name":"reverse","semantic_status":"valid","provenance":"v8_inverse_completion" if not new else "v8_specific_novel"},
    ]
    if extra:
        assertions.append({"source_cui":a,"target_cui":z,"relation_name":"extra_semantics","semantic_status":"valid","provenance":"v8_specific_novel"})
    return {
        "pair_id":f"P{i:03d}",
        "future_policy":"RERANK_ONLY" if broad else "EXPAND_ELIGIBLE",
        "connectivity_new_vs_safe63":new,
        "directed_assertions":assertions,
    }


def test_builder_counts_and_collapses_assertion_multiplicity():
    pairs=[_pair(i,broad=(i==1),extra=(i==2)) for i in range(1,19)]
    pairs.append(_pair(19,new=True))
    v9={
        "schema_version":"nci_direct_final_artifact_v9",
        "semantic_pair_count":19,
        "new_pair_count_vs_safe63":1,
        "bidirectionally_source_supported_pair_count":19,
        "benchmark_data_used":False,
        "pairs":pairs,
    }
    out=b.build_artifacts(v9)
    assert out["N1"]["edge_count"]==18
    assert out["N2"]["edge_count"]==36
    assert out["N3"]["edge_count"]==38
    assert all(e["expansion_mode"]=="support_only" for e in out["N3"]["edges"])
    # Extra source assertion on same pair/direction remains one retrieval edge.
    p2=[e for e in out["N2"]["edges"] if e["pair_id"]=="P002"]
    assert len(p2)==2
    assert max(e["assertion_count_collapsed"] for e in p2)==2
