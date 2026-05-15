from flask import Flask, request, jsonify
from flask_cors import CORS

import time
import traceback
import settings

from app import handle_query

# =========================================================
# 🔹 FLASK INIT
# =========================================================
app = Flask(__name__)

CORS(app)

SERVER_START_TIME = time.time()


# =========================================================
# 🔹 SAFE JSON RESPONSE
# =========================================================
def safe_response(result):

    explanation = result.get(
        "explanation",
        {}
    )

    evaluation = result.get(
        "evaluation",
        {}
    )

    metrics = result.get(
        "metrics",
        {}
    )

    query_analysis = result.get(
        "query_analysis",
        {}
    )

    return {

        # =====================================================
        # 🔹 CORE OUTPUT
        # =====================================================
        "answer": result.get(
            "answer",
            ""
        ),

        "confidence": result.get(
            "confidence",
            0.5
        ),

        # =====================================================
        # 🔹 EXPLAINABILITY
        # =====================================================
        "reasoning": explanation.get(
            "reasoning",
            ""
        ),

        "supporting_sentences": explanation.get(
            "supporting_sentences",
            []
        ),

        "grounded": explanation.get(
            "grounded",
            False
        ),

        "quality": explanation.get(
            "quality",
            "Low"
        ),

        # =====================================================
        # 🔹 SOURCES
        # =====================================================
        "sources": result.get(
            "sources",
            []
        ),

        "source_texts": result.get(
            "source_texts",
            []
        ),

        # =====================================================
        # 🔹 EVALUATION
        # =====================================================
        "evaluation": {

            "score": evaluation.get(
                "score",
                0
            ),

            "answer_relevance": evaluation.get(
                "answer_relevance",
                0
            ),

            "grounding_score": evaluation.get(
                "grounding_score",
                0
            ),

            "hallucination_risk": evaluation.get(
                "hallucination_risk",
                "medium"
            ),

            "retrieval_score": evaluation.get(
                "retrieval_score",
                0
            )
        },

        # =====================================================
        # 🔹 QUERY ANALYSIS
        # =====================================================
        "query_analysis": {

            "intent": query_analysis.get(
                "intent"
            ),

            "query_type": query_analysis.get(
                "query_type"
            ),

            "expanded_query": query_analysis.get(
                "expanded_query"
            ),

            "keywords": query_analysis.get(
                "keywords",
                []
            )
        },

        # =====================================================
        # 🔹 PIPELINE METRICS
        # =====================================================
        "metrics": metrics,

        "settings": settings.public_settings()
    }


# =========================================================
# 🔹 QUERY ENDPOINT
# =========================================================
@app.route(
    "/query",
    methods=["POST"]
)
def query():

    request_start = time.time()

    try:

        # =====================================================
        # 🔹 INPUT
        # =====================================================
        data = request.get_json()

        if not data:

            return jsonify({

                "error": "Missing JSON body"

            }), 400

        if "query" not in data:

            return jsonify({

                "error": "Missing query"

            }), 400

        user_query = str(
            data.get(
                "query",
                ""
            )
        ).strip()

        # =====================================================
        # 🔹 VALIDATION
        # =====================================================
        if not user_query:

            return jsonify({

                "error": "Empty query"

            }), 400

        if len(user_query) < 3:

            return jsonify({

                "error": "Query too short"

            }), 400

        if len(user_query) > 1000:

            return jsonify({

                "error": "Query too long"

            }), 400

        print("\n" + "=" * 60)

        print(
            f"🩺 Incoming Query: {user_query}"
        )

        # =====================================================
        # 🔹 PIPELINE
        # =====================================================
        result = handle_query(
            user_query
        )

        # =====================================================
        # 🔹 RESPONSE
        # =====================================================
        response = safe_response(
            result
        )

        # =====================================================
        # 🔹 API METRICS
        # =====================================================
        response["api_metrics"] = {

            "request_time": round(
                time.time() - request_start,
                2
            ),

            "server_uptime_minutes": round(
                (
                    time.time()
                    -
                    SERVER_START_TIME
                ) / 60,
                2
            )
        }

        print(
            f"✅ Request completed in "
            f"{response['api_metrics']['request_time']}s"
        )

        return jsonify(response)

    # =========================================================
    # 🔹 FAILURE
    # =========================================================
    except Exception as e:

        print("\n❌ SERVER ERROR")
        print(traceback.format_exc())

        return jsonify({

            "answer": (

                "The oncology AI system "
                "encountered an internal error."
            ),

            "confidence": 0.1,

            "reasoning": (

                "Pipeline execution failed."
            ),

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

            "error": str(e)
        }), 500


# =========================================================
# 🔹 SETTINGS ENDPOINTS
# =========================================================
@app.route(
    "/settings",
    methods=["GET"]
)
def get_runtime_settings():

    return jsonify(
        settings.public_settings()
    )


@app.route(
    "/settings/update",
    methods=["POST"]
)
def update_runtime_settings():

    data = request.get_json()

    if not isinstance(data, dict):

        return jsonify({

            "error": "Missing JSON body"

        }), 400

    updated = settings.update_settings(
        data
    )

    return jsonify({

        "enable_laqa": updated["enable_laqa"],

        "enable_mrl": updated["enable_mrl"]
    })


# =========================================================
# 🔹 SYSTEM VALIDATION ENDPOINT
# =========================================================
@app.route(
    "/system/validate-index",
    methods=["GET"]
)
def validate_index():
    """
    Validate database consistency and return status.
    """
    import os
    import json

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
        # Check metadata exists
        if not os.path.exists(metadata_path):
            status["valid"] = False
            status["errors"].append(f"Metadata not found at {metadata_path}")
        else:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            status["metadata"] = metadata

            # Check MRL mode consistency
            if metadata.get('mrl_enabled') != settings.is_mrl_enabled():
                status["valid"] = False
                status["errors"].append(
                    f"MRL mode mismatch: database={metadata.get('mrl_enabled')}, "
                    f"setting={settings.is_mrl_enabled()}"
                )

            # Check dimension consistency
            if metadata.get('embedding_dimension') != status["active_dimension"]:
                status["valid"] = False
                status["errors"].append(
                    f"Dimension mismatch: database={metadata.get('embedding_dimension')}, "
                    f"expected={status['active_dimension']}"
                )

        # Check FAISS index exists
        if not os.path.exists(faiss_path):
            status["valid"] = False
            status["errors"].append(f"FAISS index not found at {faiss_path}")

    except Exception as e:
        status["valid"] = False
        status["errors"].append(f"Validation error: {str(e)}")

    http_status = 200 if status["valid"] else 503

    return jsonify(status), http_status


# =========================================================
# 🔹 HEALTH CHECK
# =========================================================
@app.route(
    "/health",
    methods=["GET"]
)
def health():

    uptime = round(
        (
            time.time()
            -
            SERVER_START_TIME
        ) / 60,
        2
    )

    return jsonify({

        "status": "ok",

        "service": "Oncology Agentic RAG",

        "uptime_minutes": uptime
    })


# =========================================================
# 🔹 DEBUG ROUTE
# =========================================================
@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "message": (
            "Oncology Agentic RAG API running"
        ),

        "routes": [

            "/query",

            "/health",

            "/settings",

            "/settings/update"
        ]
    })


# =========================================================
# 🔹 RUN SERVER
# =========================================================
if __name__ == "__main__":

    print(
        "🚀 Oncology AI Server running "
        "on http://localhost:5000"
    )

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )
