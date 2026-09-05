import os
import time
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import settings
from app import handle_query

app = Flask(__name__)

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


def safe_response(result):
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
            "retrieval_score": evaluation.get("retrieval_score", 0)
        },
        "query_analysis": {
            "intent": query_analysis.get("intent"),
            "query_type": query_analysis.get("query_type"),
            "expanded_query": query_analysis.get("expanded_query"),
            "keywords": query_analysis.get("keywords", []),
            "query_metadata": query_analysis.get("query_metadata", {})
        },
        "metrics": metrics,
        "disclaimer": (
            "For research and educational purposes only. "
            "Not certified for direct clinical diagnosis or medical decision support."
        ),
        "settings": settings.public_settings()
    }


@app.route("/query", methods=["POST"])
def query():
    """Main endpoint for submitting oncology medical queries."""
    request_start = time.time()

    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing query or invalid JSON"}), 400

        user_query = str(data.get("query", "")).strip()

        if not user_query:
            return jsonify({"error": "Empty query"}), 400
        if len(user_query) < 3:
            return jsonify({"error": "Query too short"}), 400
        if len(user_query) > 1000:
            return jsonify({"error": "Query too long"}), 400

        print("\n" + "=" * 60)
        print(f"🩺 Incoming Query: {user_query}")

        result = handle_query(user_query)
        response = safe_response(result)

        response["api_metrics"] = {
            "request_time": round(time.time() - request_start, 2),
            "server_uptime_minutes": round((time.time() - SERVER_START_TIME) / 60, 2)
        }

        print(f"✅ Request completed in {response['api_metrics']['request_time']}s")
        return jsonify(response)

    except Exception as e:
        print("\n❌ SERVER ERROR")
        print(traceback.format_exc())

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


@app.route("/settings", methods=["GET"])
def get_runtime_settings():
    """Returns current active settings configuration."""
    return jsonify(settings.public_settings())


@app.route("/settings/update", methods=["POST"])
def update_runtime_settings():
    """Updates runtime pipeline settings (LAQA, MRL mode) with input validation."""
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
    return jsonify({
        "enable_rag": updated["enable_rag"],
        "enable_laqa": updated["enable_laqa"],
        "enable_mrl": updated["enable_mrl"],
        "active_database": updated.get("active_database", "mrl")
    })


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
                import json
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


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint returning server status and uptime."""
    uptime = round((time.time() - SERVER_START_TIME) / 60, 2)
    return jsonify({
        "status": "ok",
        "service": "Oncology Agentic RAG",
        "uptime_minutes": uptime
    })


@app.route("/", methods=["GET"])
@app.route("/app", methods=["GET"])
def serve_app():
    """Serves the Single-Page Web Application frontend."""
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
    if os.path.exists(os.path.join(frontend_dir, "index.html")):
        return send_from_directory(frontend_dir, "index.html")
    return jsonify({
        "message": "Oncology Agentic RAG API running",
        "routes": ["/query", "/health", "/settings", "/settings/update", "/system/validate-index"]
    })


if __name__ == "__main__":
    print("🚀 Oncology AI Server running on http://localhost:5000")
    _debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=_debug)
