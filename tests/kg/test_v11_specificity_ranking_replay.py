from __future__ import annotations
import importlib.util
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[2]/'scripts'/'analyze_v11_specificity_ranking.py'
spec=importlib.util.spec_from_file_location('rankreplay',SCRIPT); mod=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod)
def cand(uid,diags,rank=1,document_id='doc-A'): return {'source_rank':rank,'source':'x','section':{'section_uid':uid,'document_id':document_id,'printed_section_id':uid,'represented_section_ids':[uid],'match_diagnostics':diags}}
def test_graph_evidence_uses_seed_cosine_fallback():
 d={'query_term':'q','concept_name':'target','similarity':None,'evidence_source':'ontology_bridge_artifact','lexical_weight':0.73}; assert mod.evidence_similarity(d)==0.73
def test_unique_specificity_deduplicates_concept_across_terms():
 c=cand('S',[{'query_term':'a','concept_name':'rare','similarity':0.9},{'query_term':'b','concept_name':'rare','similarity':0.8}]); f=mod.ranking_features(c,{'rare':2.5}); assert f['coverage']==2 and f['unique_concept_count']==1 and f['unique_specificity']==2.5
def test_r2a_prefers_more_specific_at_equal_coverage():
 a=cand('A',[{'query_term':'q','concept_name':'generic','similarity':0.95}],1); b=cand('B',[{'query_term':'q','concept_name':'rare','similarity':0.70}],2); r=mod.rank_variant([a,b],'r2a_u',{'generic':0.1,'rare':2.0}); assert [mod.uid(x) for x in r]==['B','A']
def test_best_channel_has_no_double_support_bonus():
 a=cand('A',[{'query_term':'q','concept_name':'a','similarity':1.0}],1); b=cand('B',[{'query_term':'q','concept_name':'b','similarity':1.0}],2); assert [mod.uid(x) for x in mod.best_channel([a,b],[b,a])]==['A','B']


def test_gold_identity_is_document_scoped():
 gold_row={'question_id':'q','gold_sections':[{'document_id':'doc-A','printed_section_id':'4.1'}]}
 wrong=cand('4.1',[],document_id='doc-B')
 right=cand('4.1',[],document_id='doc-A')
 assert mod.gold_ids(gold_row) == {'doc-a::4.1'}
 assert mod.covered_ids(wrong) == {'doc-b::4.1'}
 assert mod.covered_ids(right) == {'doc-a::4.1'}
 assert mod.metrics(gold_row,[wrong])['hit@1'] == 0.0
 assert mod.metrics(gold_row,[right])['hit@1'] == 1.0

def test_represented_sections_remain_document_scoped():
 candidate=cand('4',[],document_id='doc-A')
 candidate['section']['represented_section_ids']=['4.1','4.2']
 assert mod.covered_ids(candidate) == {'doc-a::4','doc-a::4.1','doc-a::4.2'}
