import os
import sys
import pytest
import warnings

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import settings
from server import app
from modules.optimization.response_optimizer import optimize_response, prune_redundancy
from metrics import recall_at_k


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# =========================================================
# 1. API CONTRACT & ROUTE TESTING
# =========================================================
def test_api_health_contract(client):
    """GET /health must return 200 with service and uptime_minutes."""
    res = client.get('/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "Oncology Agentic RAG"
    assert "uptime_minutes" in data


def test_api_serve_frontend(client):
    """GET / and GET /app must serve the SPA frontend HTML."""
    res = client.get('/')
    assert res.status_code == 200
    assert b"Oncology Research Assistant" in res.data
    assert b"<!DOCTYPE html>" in res.data

    res_app = client.get('/app')
    assert res_app.status_code == 200
    assert b"Oncology Research Assistant" in res_app.data


def test_api_settings_get(client):
    """GET /settings must return public settings dictionary."""
    res = client.get('/settings')
    assert res.status_code == 200
    data = res.get_json()
    for key in ["enable_rag", "enable_laqa", "enable_mrl", "active_database"]:
        assert key in data


def test_api_settings_update_valid(client):
    """POST /settings/update with valid parameters must update and return new state."""
    res = client.post('/settings/update', json={
        "enable_laqa": False,
        "enable_mrl": True,
        "active_database": "mrl"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["enable_laqa"] is False
    assert data["enable_mrl"] is True
    assert data["active_database"] == "mrl"

    # Restore default
    res_restore = client.post('/settings/update', json={
        "enable_laqa": True,
        "enable_mrl": True,
        "active_database": "mrl"
    })
    assert res_restore.status_code == 200
    assert res_restore.get_json()["enable_laqa"] is True


def test_api_settings_update_invalid_inputs(client):
    """POST /settings/update must reject non-dict, unknown keys, and invalid databases."""
    # Non-dict JSON
    res1 = client.post('/settings/update', data="not json", content_type="application/json")
    assert res1.status_code == 400

    # Invalid active_database
    res2 = client.post('/settings/update', json={"active_database": "invalid_fake_db"})
    assert res2.status_code == 400
    assert "Invalid active_database" in res2.get_json()["error"]


def test_api_validate_index(client):
    """GET /system/validate-index must report active database consistency without crashing."""
    settings.update_settings({"active_database": "mrl", "enable_mrl": True})
    res = client.get('/system/validate-index')
    assert res.status_code == 200
    data = res.get_json()
    assert data["valid"] is True
    assert data["active_dimension"] == 512
    assert data["mrl_enabled"] is True


# =========================================================
# 2. QUERY INPUT SANITIZATION & EDGE CASES
# =========================================================
def test_api_query_validation_empty(client):
    """POST /query with empty string must return 400."""
    res = client.post('/query', json={"query": "   "})
    assert res.status_code == 400
    assert "Empty query" in res.get_json()["error"]


def test_api_query_validation_too_short(client):
    """POST /query with query < 3 characters must return 400."""
    res = client.post('/query', json={"query": "ab"})
    assert res.status_code == 400
    assert "Query too short" in res.get_json()["error"]


def test_api_query_validation_too_long(client):
    """POST /query with query > 1000 characters must return 400."""
    long_q = "cancer " * 200  # > 1000 chars
    res = client.post('/query', json={"query": long_q})
    assert res.status_code == 400
    assert "Query too long" in res.get_json()["error"]


def test_api_query_missing_payload(client):
    """POST /query with no JSON or missing 'query' key must return 400."""
    res1 = client.post('/query', json={})
    assert res1.status_code == 400

    res2 = client.post('/query', json={"text": "hello"})
    assert res2.status_code == 400


# =========================================================
# 3. RESPONSE OPTIMIZER RIGOROUS AUDIT
# =========================================================
def test_optimizer_duplicate_answer():
    """Verify completely duplicated paragraphs are deduplicated."""
    raw = (
        "Osimertinib is recommended for EGFR-mutant NSCLC.\n"
        "Osimertinib is recommended for EGFR-mutant NSCLC.\n"
        "Osimertinib is recommended for EGFR-mutant NSCLC."
    )
    result = optimize_response(raw)
    assert result["is_valid"] is True
    assert result["diagnostics"]["duplicate_lines_removed"] == 2
    assert result["optimized_answer"] == "Osimertinib is recommended for EGFR-mutant NSCLC."


def test_optimizer_partially_duplicated_answer():
    """Verify mixed unique and duplicated lines are filtered cleanly."""
    raw = (
        "- Trastuzumab is indicated for HER2-positive breast cancer.\n"
        "- Pertuzumab provides synergistic dual-blockade.\n"
        "- Trastuzumab is indicated for HER2-positive breast cancer."
    )
    result = optimize_response(raw)
    assert result["diagnostics"]["duplicate_lines_removed"] == 1
    assert "Trastuzumab is indicated for HER2-positive breast cancer" in result["optimized_answer"]
    assert "Pertuzumab provides synergistic dual-blockade" in result["optimized_answer"]


def test_optimizer_medically_similar_distinct_lines():
    """Verify lines that share medical words but convey distinct facts are preserved."""
    raw = (
        "• BRCA1 mutations confer elevated lifetime risk of breast and ovarian cancer.\n"
        "• BRCA2 mutations also increase risks of pancreatic and prostate cancer."
    )
    result = optimize_response(raw)
    assert result["diagnostics"]["duplicate_lines_removed"] == 0
    lines = result["optimized_answer"].split("\n")
    assert len(lines) == 2
    assert "BRCA1" in lines[0]
    assert "BRCA2" in lines[1]


def test_optimizer_empty_and_whitespace_safe():
    """Verify empty or whitespace-only answers return safe fallback without crashing."""
    res_empty = optimize_response("")
    assert res_empty["is_valid"] is False
    assert "Unable to generate a medical answer" in res_empty["optimized_answer"]

    res_spaces = optimize_response("     \n\t   ")
    assert res_spaces["is_valid"] is False
    assert "Unable to generate a medical answer" in res_spaces["optimized_answer"]


def test_optimizer_strips_internal_scratchpad():
    """Verify that <think> and scratchpad tags are cleanly excised."""
    raw = (
        "<think>The user is asking about melanoma staging. I should check Breslow depth.</think>\n"
        "Melanoma staging incorporates Breslow tumor thickness and ulceration status."
    )
    result = optimize_response(raw)
    assert "<think>" not in result["optimized_answer"]
    assert "Melanoma staging incorporates" in result["optimized_answer"]
    assert result["diagnostics"]["artifacts_stripped"] >= 1


# =========================================================
# 4. EVALUATION & RECALL METRIC AUDIT (NO CHEATING)
# =========================================================
def test_recall_at_k_rejects_identical_or_none_pool():
    """Verify Recall@K warns and returns 0.0 when candidate_pool <= chunks, preventing circular 1.0 identity."""
    chunks = ["chunk 1", "chunk 2"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Test candidate_pool is None
        score_none = recall_at_k(chunks, query="cancer", k=2, candidate_pool=None)
        assert score_none == 0.0
        assert len(w) >= 1
        assert "candidate_pool strictly larger than chunks" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Test candidate_pool == chunks
        score_same = recall_at_k(chunks, query="cancer", k=2, candidate_pool=chunks)
        assert score_same == 0.0
        assert len(w) >= 1
        assert "candidate_pool strictly larger than chunks" in str(w[-1].message)


# =========================================================
# 5. SECURITY & ERROR MASKING AUDIT
# =========================================================
def test_server_error_masks_internal_tracebacks(client, monkeypatch):
    """Verify that internal 500 exceptions return sanitized clinical message without python traceback."""
    def broken_handler(query):
        raise RuntimeError("CRITICAL INTERNAL FAILURE: database connection lost at /var/data/secret.db")

    import server
    monkeypatch.setattr(server, "handle_query", broken_handler)

    res = client.post('/query', json={"query": "What is breast cancer?"})
    assert res.status_code == 500
    data = res.get_json()
    assert "error" in data
    assert "An internal server error occurred" in data["error"]
    # Ensure no internal path or exception name leaked
    assert "CRITICAL INTERNAL FAILURE" not in str(data)
    assert "/var/data/secret.db" not in str(data)
    assert "Traceback" not in str(data)
