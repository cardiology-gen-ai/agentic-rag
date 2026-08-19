from agentic_rag.kg.pipeline import _validate_mode

def test_controlled_rescue_mode_is_registered():
    mode="mentions_nonhier_artifact_safe_strict_direct_first_rescue"
    assert _validate_mode(mode)==mode
