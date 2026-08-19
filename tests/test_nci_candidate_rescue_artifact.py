import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("b",ROOT/"scripts"/"build_nci_candidate_rescue_artifact.py")
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

def pair(i,broad=False,new=False,extra=False):
    a=f"C{i:03d}A"; z=f"C{i:03d}B"
    ass=[{"source_cui":a,"target_cui":z,"relation_name":"fwd","semantic_status":"valid_but_broad" if broad else "valid","provenance":"safe63" if not new else "v8_specific_novel"},{"source_cui":z,"target_cui":a,"relation_name":"rev","semantic_status":"valid","provenance":"v8_inverse_completion"}]
    if extra: ass.append({"source_cui":a,"target_cui":z,"relation_name":"extra","semantic_status":"valid","provenance":"v8_specific_novel"})
    return {"pair_id":f"P{i:03d}","future_policy":"RERANK_ONLY" if broad else "EXPAND_ELIGIBLE","connectivity_new_vs_safe63":new,"directed_assertions":ass}

def test_policy_maps_to_expand_vs_support_only_without_double_votes():
    pairs=[pair(i,broad=(i==1),extra=(i==2)) for i in range(1,19)]+[pair(19,new=True)]
    v9={"schema_version":"nci_direct_final_artifact_v9","semantic_pair_count":19,"bidirectionally_source_supported_pair_count":19,"benchmark_data_used":False,"pairs":pairs}
    out=b.build(v9)
    assert out["edge_count"]==38
    assert out["pair_policy_counts"]=={"RERANK_ONLY":1,"EXPAND_ELIGIBLE":18}
    assert out["directed_expansion_mode_counts"]=={"expand":36,"support_only":2}
    p1=[e for e in out["edges"] if e["pair_id"]=="P001"]
    assert len(p1)==2 and all(e["expansion_mode"]=="support_only" for e in p1)
    p2=[e for e in out["edges"] if e["pair_id"]=="P002"]
    assert len(p2)==2
    assert max(e["assertion_count_collapsed"] for e in p2)==2
