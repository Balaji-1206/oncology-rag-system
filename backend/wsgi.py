"""Production WSGI entrypoint for Oncology Agentic RAG."""
import os
import sys

# Ensure backend directory is in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from server import app
from modules.observability.logger import get_logger

logger = get_logger("wsgi")
logger.info("Initializing Oncology Agentic RAG production WSGI application")

if __name__ == "__main__":
    # If run directly on Windows or without Gunicorn, use Waitress if available, else Flask
    try:
        from waitress import serve
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"Serving with Waitress production server on port {port}")
        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        port = int(os.environ.get("PORT", 5000))
        logger.warning(f"Waitress not installed; falling back to standard server on port {port}")
        app.run(host="0.0.0.0", port=port, debug=False)
