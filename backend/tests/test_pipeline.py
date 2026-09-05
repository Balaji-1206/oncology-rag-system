import os
import sys
import pytest

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import settings
from modules.optimization.response_optimizer import prune_redundancy
from modules.generator.medgemma import build_context
from modules.xai.explain import REASONING_MODEL
from modules.generator.medgemma import MODEL
from latency.profile_pipeline import PipelineProfiler


def test_effective_embedding_dimension_mrl():
    """Verify that MRL database resolves to 512 dimensions."""
    settings.update_settings({"active_database": "mrl", "enable_mrl": True})
    dim = settings.effective_embedding_dimension()
    assert dim == 512, f"Expected 512 for MRL database, got {dim}"


def test_effective_embedding_dimension_full():
    """Verify that Full database inspects metadata.json and resolves to 256 dimensions."""
    settings.update_settings({"active_database": "full", "enable_mrl": False})
    dim = settings.effective_embedding_dimension()
    assert dim == 256, f"Expected 256 for Full database from metadata, got {dim}"
    # Restore MRL default
    settings.update_settings({"active_database": "mrl", "enable_mrl": True})


def test_database_consistency_both_modes():
    """Verify validate_database_consistency passes for both mrl and full without index error."""
    from modules.retrieval.hybrid_retriever import validate_database_consistency

    # Test MRL mode
    settings.update_settings({"active_database": "mrl", "enable_mrl": True})
    meta_mrl = validate_database_consistency()
    assert meta_mrl["embedding_dimension"] == 512
    assert meta_mrl["mrl_enabled"] is True

    # Test Full mode
    settings.update_settings({"active_database": "full", "enable_mrl": False})
    meta_full = validate_database_consistency()
    assert meta_full["embedding_dimension"] == 256
    assert meta_full["mrl_enabled"] is False

    # Restore default
    settings.update_settings({"active_database": "mrl", "enable_mrl": True})


def test_response_optimizer_preserves_distinct_clinical_bullets():
    """Verify distinct clinical recommendations sharing first 6 words are NOT deleted."""
    clinical_text = (
        "* Stage IV non-small cell lung cancer patients with EGFR mutations should receive osimertinib as first-line therapy.\n"
        "* Stage IV non-small cell lung cancer patients with ALK rearrangements should receive alectinib as first-line therapy.\n"
        "* Stage IV non-small cell lung cancer patients with ROS1 fusions should receive entrectinib or crizotinib."
    )
    cleaned, duplicates_removed = prune_redundancy(clinical_text)
    assert duplicates_removed == 0, f"Expected 0 duplicates removed, got {duplicates_removed}"
    lines = [line for line in cleaned.split("\n") if line.strip()]
    assert len(lines) == 3, f"Expected 3 distinct lines preserved, got {len(lines)}"
    assert "osimertinib" in cleaned
    assert "alectinib" in cleaned
    assert "entrectinib" in cleaned


def test_response_optimizer_removes_genuine_duplicate_bullets():
    """Verify identical bullets with different markdown markers are correctly deduplicated."""
    redundant_text = (
        "- Stage IV non-small cell lung cancer patients with EGFR mutations should receive osimertinib.\n"
        "* Stage IV non-small cell lung cancer patients with EGFR mutations should receive osimertinib.\n"
        "1. Stage IV non-small cell lung cancer patients with EGFR mutations should receive osimertinib."
    )
    cleaned, duplicates_removed = prune_redundancy(redundant_text)
    assert duplicates_removed == 2, f"Expected 2 duplicate bullets removed, got {duplicates_removed}"
    lines = [line for line in cleaned.split("\n") if line.strip()]
    assert len(lines) == 1, f"Expected 1 unique line, got {len(lines)}"


def test_generator_context_expansion_beyond_1800_chars():
    """Verify build_context preserves full context budget up to 6000 chars without 1800-char truncation."""
    passages = [
        "Breast cancer oncology treatment guidelines recommend dual anti-HER2 blockade with trastuzumab and pertuzumab combined with taxane chemotherapy for patients presenting with metastatic progression. Clinical oncology trial evidence confirms significant overall survival benefits in cancer patients with confirmed HER2 overexpression across multiple cohorts.",
        "Lung cancer oncology patients exhibiting EGFR activating mutations should receive first-line osimertinib targeted therapy as the preferred standard of care. Central nervous system penetrance in lung cancer provides substantial intracranial disease control and delays central progression in treated patients.",
        "Colorectal cancer oncology patients harboring wild-type KRAS and NRAS genes respond favorably to anti-EGFR antibody cancer therapy with cetuximab or panitumumab. Primary tumor location in colon cancer correlates with superior response rates and extended survival outcomes in treated clinical patients.",
        "Cutaneous melanoma cancer bearing BRAF V600E mutations requires combined kinase inhibition cancer therapy with dabrafenib and trametinib for all eligible patients. Clinical oncology evidence indicates prolonged progression-free survival in advanced melanoma cancer patients compared to previous single-agent treatments.",
        "Leukemia cancer oncology patients featuring FLT3 mutations should receive targeted midostaurin cancer therapy combined with standard cytarabine and daunorubicin induction. Allogeneic stem cell transplantation remains the standard consolidation therapy for eligible leukemia cancer patients achieving complete remission.",
        "Prostate cancer oncology patients demonstrate objective therapeutic responses to next-generation androgen receptor cancer therapy including enzalutamide and abiraterone. Serum PSA kinetics in prostate cancer patients assess ongoing oncology clinical response and guide subsequent treatment sequencing.",
        "Ovarian cancer oncology patients with deleterious BRCA mutations derive profound progression-free survival benefit from maintenance PARP inhibitor therapy with olaparib. Hematologic monitoring in ovarian cancer oncology is warranted during active cancer treatment for all enrolled clinical patients."
    ]
    # Verify legacy 1800 ceiling chokes context
    c_1800 = build_context(passages, query="cancer oncology patients therapy treatment clinical", intent="exploratory", query_type="list", max_chars=1800)
    assert len(c_1800) == 1800, f"Expected exactly 1800 with 1800 cap, got {len(c_1800)}"

    # Verify remediated 6000 ceiling preserves full multi-chunk evidence beyond 1800
    c_6000 = build_context(passages, query="cancer oncology patients therapy treatment clinical", intent="exploratory", query_type="list", max_chars=6000)
    assert len(c_6000) > 1800, f"Context was choked at {len(c_6000)} chars (<= 1800); expected > 1800 chars"
    assert len(c_6000) >= 2200, f"Expected >= 2200 chars for 7 chunks, got {len(c_6000)}"
    assert "[Chunk 1]" in c_6000
    assert "[Chunk 7]" in c_6000


def test_xai_model_aligned_with_generator():
    """Verify XAI reasoning model matches the generator Ollama model to avoid GPU thrashing."""
    assert REASONING_MODEL == MODEL, (
        f"XAI model ({REASONING_MODEL}) does not match generator model ({MODEL})"
    )


def test_settings_coercion():
    """Verify settings boolean and string coercions work safely."""
    settings.update_settings({"enable_rag": "false", "enable_laqa": 0, "enable_mrl": "1"})
    assert settings.is_rag_enabled() is False
    assert settings.is_laqa_enabled() is False
    assert settings.is_mrl_enabled() is True

    # Restore defaults
    settings.update_settings({"enable_rag": True, "enable_laqa": True, "enable_mrl": True})
    assert settings.is_rag_enabled() is True
    assert settings.is_laqa_enabled() is True


def test_pipeline_profiler_record_stage_duration():
    """Verify PipelineProfiler can record discrete stage durations accurately."""
    profiler = PipelineProfiler(enabled=True, capture_system_stats=False)
    with profiler.question_scope(1, "Test Question"):
        profiler.record_stage_duration("Dense Embedding", 0.045)
        profiler.record_stage_duration("FAISS Search", 0.012)
        profiler.record_stage_duration("BM25 Retrieval", 0.008)
        profiler.record_stage_duration("Hybrid Fusion", 0.005)

    record = profiler.question_records[0]
    assert record.stage_times["Dense Embedding"] == pytest.approx(0.045, rel=1e-3)
    assert record.stage_times["FAISS Search"] == pytest.approx(0.012, rel=1e-3)
    assert record.stage_times["BM25 Retrieval"] == pytest.approx(0.008, rel=1e-3)
    assert record.stage_times["Hybrid Fusion"] == pytest.approx(0.005, rel=1e-3)
    # Ensure FAISS does not falsely absorb embedding or BM25
    assert record.stage_times["FAISS Search"] < record.stage_times["Dense Embedding"]
