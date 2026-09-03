from __future__ import annotations

import importlib.util
from pathlib import Path

from agentic_rag.kg.models import KGSectionResult

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_bm25plus_fixed_pool.py"


def load_script():
    spec = importlib.util.spec_from_file_location("analyze_bm25plus_fixed_pool", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate(uid: str, title: str, text: str):
    doc, section_id = uid.split("::", 1)
    section = KGSectionResult(
        section_uid=uid,
        document_id=doc,
        section_id=section_id,
        printed_section_id=section_id,
        title=title,
        text=text,
    )
    return {
        "section": section.model_dump(mode="json"),
        "source": "mentions",
        "source_rank": 1,
        "metadata": {},
    }


def test_candidate_to_document_uses_kg_page_content():
    mod = load_script()
    item = candidate("Doc::1", "Hypertrophic cardiomyopathy", "Body text here.")
    document = mod.candidate_to_document(item, original_rank=1)
    assert document.metadata["record_id"] == "Doc::1"
    assert document.metadata["original_candidate_rank"] == 1
    assert "Hypertrophic cardiomyopathy" in document.page_content
    assert "Body text here." in document.page_content


def test_bm25plus_rerank_preserves_membership_and_is_deterministic():
    mod = load_script()
    items = [
        candidate("Doc::1", "General", "unrelated background"),
        candidate("Doc::2", "Cardiomyopathy", "hypertrophic cardiomyopathy diagnosis"),
        candidate("Doc::3", "Follow-up", "clinical follow-up"),
    ]
    first, _ = mod.rerank_pool(items, "hypertrophic cardiomyopathy")
    second, _ = mod.rerank_pool(items, "hypertrophic cardiomyopathy")
    first_uids = [mod.candidate_uid(x) for x in first]
    second_uids = [mod.candidate_uid(x) for x in second]
    assert set(first_uids) == {"Doc::1", "Doc::2", "Doc::3"}
    assert len(first_uids) == 3
    assert first_uids == second_uids
    assert first_uids[0] == "Doc::2"
