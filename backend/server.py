import json
import os
import time
import traceback
import requests
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

import settings
from app import handle_query
from modules.security.auth import require_auth, require_admin, get_auth_manager
from modules.security.guardrails import validate_query_security
from modules.security.phi_scrubber import scrub_phi
from modules.security.rate_limiter import get_rate_limiter
from modules.observability.logger import get_logger, set_request_id, get_request_id
from modules.observability.metrics_collector import get_metrics_collector

app = Flask(__name__)
logger = get_logger("oncology_server")

# Restrict CORS to known origins. Override via CORS_ORIGINS env var.
_cors_origins_env = os.environ.get("CORS_ORIGINS", "")
if _cors_origins_env:
    _allowed_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    _allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5000",
    ]

CORS(app, origins=_allowed_origins)
SERVER_START_TIME = time.time()


@app.before_request
def handle_before_request():
    """Sets request correlation ID and applies sliding-window rate limiting."""
    req_id = request.headers.get("X-Request-ID") or None
    set_request_id(req_id)

    # Apply rate limits on intensive endpoints
    if request.path in {"/query", "/query/stream", "/settings/update"}:
        allowed, remaining, retry_after = get_rate_limiter().is_allowed()
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for client on {request.path}",
                extra={"extra_fields": {"retry_after": retry_after}}
            )
            return jsonify({
                "error": "Too Many Requests",
                "message": f"Rate limit exceeded. Please retry in {retry_after} seconds.",
                "retry_after": retry_after
            }), 429


@app.after_request
def set_security_headers(response):
    """Attaches standard security and tracking headers to all HTTP responses."""
    response.headers["X-Request-ID"] = get_request_id()
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


def safe_response(result, phi_redactions=None):
    """Formats and sanitizes the pipeline output for REST JSON consumption."""
    explanation = result.get("explanation", {})
    evaluation = result.get("evaluation", {})
    metrics = result.get("metrics", {})
    query_analysis = result.get("query_analysis", {})

    return {
        "answer": result.get("answer", ""),
        "raw_answer": result.get("raw_answer", ""),
        "optimization_stats": result.get("optimization_stats", {}),
        "confidence": result.get("confidence", 0.5),
        "reasoning": explanation.get("reasoning", ""),
        "supporting_sentences": explanation.get("supporting_sentences", []),
        "grounded": explanation.get("grounded", False),
        "quality": explanation.get("quality", "Low"),
        "sources": result.get("sources", []),
        "source_texts": result.get("source_texts", []),
        "evaluation": {
            "score": evaluation.get("score", 0),
            "answer_relevance": evaluation.get("answer_relevance", 0),
            "grounding_score": evaluation.get("grounding_score", 0),
            "hallucination_risk": evaluation.get("hallucination_risk", "medium"),
            "retrieval_score": evaluation.get("retrieval_score", 0),
            "contradiction_detected": evaluation.get("contradiction_detected", False),
            "contradiction_reasons": evaluation.get("contradiction_reasons", [])
        },
        "query_analysis": {
            "intent": query_analysis.get("intent"),
            "query_type": query_analysis.get("query_type"),
            "expanded_query": query_analysis.get("expanded_query"),
            "keywords": query_analysis.get("keywords", []),
            "query_metadata": query_analysis.get("query_metadata", {})
        },
        "phi_redactions": phi_redactions or {},
        "metrics": metrics,
        "disclaimer": (
            "For research and educational purposes only. "
            "Not certified for direct clinical diagnosis or medical decision support."
        ),
        "settings": settings.public_settings()
    }


@app.route("/query", methods=["POST"])
@require_auth
def query():
    """Main endpoint for submitting oncology medical queries."""
    request_start = time.time()
    collector = get_metrics_collector()

    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing query or invalid JSON"}), 400

        raw_query = str(data.get("query", ""))

        # 1. Security Guardrails Validation
        is_valid, error_msg, sanitized_query = validate_query_security(raw_query)
        if not is_valid:
            collector.record_query(latency=0.01, success=False)
            return jsonify({"error": error_msg}), 400

        # 2. HIPAA PHI De-identification
        cleaned_query, phi_redactions = scrub_phi(sanitized_query)
        if phi_redactions:
            logger.info(
                "HIPAA PHI identifiers masked from query",
                extra={"extra_fields": {"redactions": phi_redactions}}
            )

        logger.info(f"Processing clinical query: {cleaned_query[:80]}...")

        # 3. Execute Pipeline
        result = handle_query(cleaned_query)
        response = safe_response(result, phi_redactions=phi_redactions)

        elapsed = round(time.time() - request_start, 2)
        response["api_metrics"] = {
            "request_time": elapsed,
            "server_uptime_minutes": round((time.time() - SERVER_START_TIME) / 60, 2),
            "request_id": get_request_id()
        }

        # 4. Record Telemetry Metrics
        eval_data = result.get("evaluation", {})
        collector.record_query(
            latency=elapsed,
            success=True,
            cache_hit=result.get("semantic_cache_hit", False),
            contradiction=eval_data.get("contradiction_detected", False),
            hallucination_risk=eval_data.get("hallucination_risk", "medium"),
            evaluator_mode=eval_data.get("evaluator_mode", "llm_evaluator")
        )

        logger.info(f"Request completed in {elapsed}s with score {eval_data.get('score', 'N/A')}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Pipeline failure: {e}\n{traceback.format_exc()}")
        collector.record_query(latency=round(time.time() - request_start, 2), success=False)

        return jsonify({
            "answer": "The oncology AI system encountered an internal error.",
            "confidence": 0.1,
            "reasoning": "Pipeline execution failed.",
            "supporting_sentences": [],
            "sources": [],
            "source_texts": [],
            "grounded": False,
            "quality": "Low",
            "evaluation": {
                "score": 1,
                "hallucination_risk": "medium"
            },
            "metrics": {},
            "error": "An internal server error occurred while processing the clinical query."
        }), 500


@app.route("/query/stream", methods=["POST"])
@require_auth
def query_stream():
    """
    Server-Sent Events (SSE) streaming endpoint for progressive real-time delivery.
    Emits events: query_analysis, retrieval, answer_chunk, evaluation, and done.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing query or invalid JSON"}), 400

    raw_query = str(data.get("query", ""))
    is_valid, error_msg, sanitized_query = validate_query_security(raw_query)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    cleaned_query, phi_redactions = scrub_phi(sanitized_query)

    def event_stream():
        yield f"event: status\ndata: {json.dumps({'message': 'Query sanitized and de-identified', 'phi_redacted': bool(phi_redactions)})}\n\n"
        try:
            result = handle_query(cleaned_query)
            payload = safe_response(result, phi_redactions=phi_redactions)

            # Stream Query Analysis
            yield f"event: query_analysis\ndata: {json.dumps(payload['query_analysis'])}\n\n"

            # Stream Sources
            yield f"event: sources\ndata: {json.dumps({'sources': payload['sources'], 'count': len(payload['sources'])})}\n\n"

            # Stream Answer in progressive paragraphs/chunks
            paragraphs = payload["answer"].split("\n")
            for para in paragraphs:
                if para.strip():
                    yield f"event: answer_chunk\ndata: {json.dumps({'chunk': para})}\n\n"

            # Stream XAI & Evaluation
            yield f"event: evaluation\ndata: {json.dumps(payload['evaluation'])}\n\n"
            yield f"event: xai\ndata: {json.dumps({'reasoning': payload['reasoning'], 'supporting_sentences': payload['supporting_sentences']})}\n\n"
            yield f"event: done\ndata: {json.dumps({'confidence': payload['confidence'], 'quality': payload['quality']})}\n\n"

        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/settings", methods=["GET"])
def get_runtime_settings():
    """Returns current active settings configuration."""
    return jsonify(settings.public_settings())


@app.route("/settings/update", methods=["POST"])
@require_admin
def update_runtime_settings():
    """Updates runtime pipeline settings (requires administrative authorization)."""
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Missing JSON body"}), 400

    allowed_keys = {
        "enable_rag", "enable_laqa", "enable_mrl",
        "active_database", "retrieval_relevance_threshold"
    }
    sanitized_data = {k: v for k, v in data.items() if k in allowed_keys}

    if "active_database" in sanitized_data:
        if sanitized_data["active_database"] not in {"mrl", "full", "dockling_mrl"}:
            return jsonify({"error": "Invalid active_database. Must be 'mrl' or 'full'"}), 400

    updated = settings.update_settings(sanitized_data)
    logger.info("Pipeline settings updated by administrator", extra={"extra_fields": {"settings": updated}})
    return jsonify({
        "enable_rag": updated["enable_rag"],
        "enable_laqa": updated["enable_laqa"],
        "enable_mrl": updated["enable_mrl"],
        "active_database": updated.get("active_database", "mrl")
    })


@app.route("/metrics", methods=["GET"])
def get_telemetry_metrics():
    """Returns operational and latency telemetry summary."""
    collector = get_metrics_collector()
    return jsonify(collector.get_summary())


@app.route("/health", methods=["GET"])
def health():
    """
    Deep health check verifying core subsystems:
    1. Server uptime
    2. Vector database index files
    3. Ollama local LLM connectivity
    """
    db_path = settings.get_database_path()
    faiss_present = os.path.exists(os.path.join(db_path, "faiss.index"))
    metadata_present = os.path.exists(os.path.join(db_path, "metadata.json"))

    # Probe Ollama service
    ollama_ok = False
    ollama_latency_ms = 0
    try:
        t0 = time.time()
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        ollama_latency_ms = round((time.time() - t0) * 1000, 1)
        ollama_ok = (r.status_code == 200)
    except Exception:
        ollama_ok = False

    subsystems = {
        "vector_database": {
            "path": db_path,
            "faiss_index": faiss_present,
            "metadata": metadata_present,
            "status": "up" if (faiss_present and metadata_present) else "down"
        },
        "generator_service": {
            "endpoint": "http://localhost:11434",
            "status": "up" if ollama_ok else "down",
            "latency_ms": ollama_latency_ms
        }
    }

    is_healthy = faiss_present and metadata_present
    status_str = "ok" if is_healthy else "unhealthy"
    deep_status = "healthy" if (is_healthy and ollama_ok) else ("degraded" if is_healthy else "unhealthy")
    http_code = 200 if is_healthy else 503

    return jsonify({
        "status": status_str,
        "deep_status": deep_status,
        "service": "Oncology Agentic RAG",
        "version": "2.1.0-production",
        "uptime_minutes": round((time.time() - SERVER_START_TIME) / 60, 2),
        "subsystems": subsystems,
        "auth_enabled": get_auth_manager().auth_enabled
    }), http_code


@app.route("/system/validate-index", methods=["GET"])
def validate_index():
    """Validates database consistency and FAISS vector dimension."""
    db_path = settings.get_database_path()
    metadata_path = f"{db_path}/metadata.json"
    faiss_path = f"{db_path}/faiss.index"

    status = {
        "active_database": db_path,
        "mrl_enabled": settings.is_mrl_enabled(),
        "active_dimension": settings.effective_embedding_dimension(),
        "valid": True,
        "errors": [],
        "metadata": None
    }

    try:
        if not os.path.exists(metadata_path):
            status["valid"] = False
            status["errors"].append(f"Metadata not found at {metadata_path}")
        else:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            status["metadata"] = metadata

            if metadata.get('mrl_enabled') != settings.is_mrl_enabled():
                status["valid"] = False
                status["errors"].append(
                    f"MRL mode mismatch: database={metadata.get('mrl_enabled')}, "
                    f"setting={settings.is_mrl_enabled()}"
                )

            if metadata.get('embedding_dimension') != status["active_dimension"]:
                status["valid"] = False
                status["errors"].append(
                    f"Dimension mismatch: database={metadata.get('embedding_dimension')}, "
                    f"expected={status['active_dimension']}"
                )

        if not os.path.exists(faiss_path):
            status["valid"] = False
            status["errors"].append(f"FAISS index not found at {faiss_path}")

    except Exception as e:
        status["valid"] = False
        status["errors"].append(f"Validation error: {str(e)}")

    http_status = 200 if status["valid"] else 503
    return jsonify(status), http_status


@app.route("/", methods=["GET"])
@app.route("/app", methods=["GET"])
def serve_app():
    """Serves the Single-Page Web Application frontend."""
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
    if os.path.exists(os.path.join(frontend_dir, "index.html")):
        return send_from_directory(frontend_dir, "index.html")
    return jsonify({
        "message": "Oncology Agentic RAG Production API running",
        "version": "2.1.0",
        "routes": [
            "/query", "/query/stream", "/health", "/metrics",
            "/settings", "/settings/update", "/system/validate-index"
        ]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Oncology AI Server on port {port}")
    _debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=_debug)
