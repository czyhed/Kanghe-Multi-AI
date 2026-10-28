"""问诊置信度计算与收敛判断（纯函数，无外部依赖）"""
from src.agents.inquiry.confidence import apply_context_weights, check_convergence
from src.agents.inquiry.state import CandidateDisease, PatientContext


def _disease(name, base, symptoms):
    return CandidateDisease(name=name, base_confidence=base, all_symptoms=symptoms)


def test_medical_history_adds_015():
    cands = [_disease("感冒", 0.5, ["咳嗽"])]
    ctx = PatientContext(medical_history=["曾患感冒"])
    result = apply_context_weights(cands, ctx, denied_symptoms=[])
    assert result[0].confidence == 0.65


def test_long_term_memory_adds_010():
    cands = [_disease("感冒", 0.5, ["咳嗽"])]
    ctx = PatientContext(long_term_memories=["去年感冒过两次"])
    result = apply_context_weights(cands, ctx, denied_symptoms=[])
    assert result[0].confidence == 0.6


def test_denied_core_symptom_subtracts_020():
    # "咳嗽"只属于感冒，被否认则扣分
    cands = [_disease("感冒", 0.5, ["咳嗽", "发热"]), _disease("流感", 0.5, ["发热"])]
    result = apply_context_weights(cands, PatientContext(), denied_symptoms=["咳嗽"])
    by_name = {c.name: c.confidence for c in result}
    assert by_name["感冒"] == 0.3
    assert by_name["流感"] == 0.5


def test_denied_shared_symptom_is_not_penalized():
    # "发热"是两种病共有的，不算核心症状
    cands = [_disease("感冒", 0.5, ["发热"]), _disease("流感", 0.5, ["发热"])]
    result = apply_context_weights(cands, PatientContext(), denied_symptoms=["发热"])
    assert all(c.confidence == 0.5 for c in result)


def test_confidence_is_clamped_and_sorted():
    cands = [
        _disease("A", 0.95, ["x"]),
        _disease("B", 0.1, ["y"]),
    ]
    ctx = PatientContext(medical_history=["A"], long_term_memories=["A"])
    result = apply_context_weights(cands, ctx, denied_symptoms=["y"])
    assert result[0].name == "A" and result[0].confidence == 1.0
    assert result[1].name == "B" and result[1].confidence == 0.0


def test_converges_when_top1_reaches_070():
    cands = [CandidateDisease(name="A", confidence=0.70)]
    assert check_convergence(cands, current_round=1) == (True, False)


def test_converges_when_top1_leads_top2_by_030():
    cands = [
        CandidateDisease(name="A", confidence=0.60),
        CandidateDisease(name="B", confidence=0.30),
    ]
    assert check_convergence(cands, current_round=1) == (True, False)


def test_does_not_converge_when_close():
    cands = [
        CandidateDisease(name="A", confidence=0.50),
        CandidateDisease(name="B", confidence=0.40),
    ]
    assert check_convergence(cands, current_round=1) == (False, False)


def test_force_conclude_at_max_rounds():
    cands = [CandidateDisease(name="A", confidence=0.1)]
    assert check_convergence(cands, current_round=10, max_rounds=10) == (True, True)


def test_no_candidates_does_not_converge():
    assert check_convergence([], current_round=1) == (False, False)
