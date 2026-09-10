import os
import sys
import pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from modules.clinical.negation_detector import ClinicalNegationDetector


def test_negation_detector_detects_contraindication_mismatch():
    context = (
        "Osimertinib is contraindicated in patients with severe pre-existing interstitial lung disease. "
        "Fatal pulmonary toxicity has been documented in clinical trials."
    )
    answer = (
        "Osimertinib is recommended as preferred first-line therapy for all EGFR-mutant NSCLC patients."
    )

    has_contra, risk, reasons = ClinicalNegationDetector.detect_contradiction(answer, context)
    assert has_contra is True
    assert risk >= 0.40
    assert len(reasons) > 0
    assert any("contraindication" in r.lower() for r in reasons)


def test_negation_detector_allows_harmonious_recommendations():
    context = (
        "Osimertinib demonstrated superior progression-free survival in untreated EGFR-mutated advanced NSCLC. "
        "It is standard first-line management."
    )
    answer = (
        "Osimertinib is recommended as preferred first-line therapy based on superior progression-free survival."
    )

    has_contra, risk, reasons = ClinicalNegationDetector.detect_contradiction(answer, context)
    assert has_contra is False
    assert risk == 0.0
    assert len(reasons) == 0


def test_negation_detector_allows_answer_that_mentions_caveat():
    context = (
        "Osimertinib is contraindicated in active interstitial lung disease."
    )
    answer = (
        "Osimertinib is indicated for EGFR mutations, but is contraindicated in patients with interstitial lung disease."
    )

    has_contra, risk, _ = ClinicalNegationDetector.detect_contradiction(answer, context)
    assert has_contra is False


def test_evaluator_penalizes_contraindicated_answer(monkeypatch):
    import modules.agent.evaluator as ev_mod

    # Even with high grounding (0.80), contradiction must penalize score to <= 4
    monkeypatch.setattr(ev_mod, "compute_combined_grounding", lambda ans, ctx: (0.80, 0.80, 0.80))
    monkeypatch.setattr(
        ev_mod.SESSION,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("Local LLM offline"))
    )

    docs = [
        "Osimertinib is contraindicated in active interstitial lung disease due to fatal pulmonary risk."
    ]
    answer = (
        "Osimertinib is recommended as preferred first-line therapy for all lung cancer patients."
    )

    eval_res = ev_mod.evaluate_answer("What is osimertinib indication?", docs, answer)
    assert eval_res["contradiction_detected"] is True
    assert eval_res["needs_retry"] is True
    assert eval_res["hallucination_risk"] == "high"
    assert eval_res["score"] <= 4

