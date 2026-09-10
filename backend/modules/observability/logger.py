import json
import logging
import os
import sys
import time
import uuid
from threading import local
from typing import Any, Dict, Optional

_THREAD_LOCAL = local()


def get_request_id() -> str:
    """Returns the current thread's request correlation ID or generates one."""
    req_id = getattr(_THREAD_LOCAL, "request_id", None)
    if not req_id:
        req_id = str(uuid.uuid4())[:8]
        _THREAD_LOCAL.request_id = req_id
    return req_id


def set_request_id(request_id: Optional[str] = None) -> str:
    """Sets correlation ID for current request thread."""
    new_id = request_id or str(uuid.uuid4())[:8]
    _THREAD_LOCAL.request_id = new_id
    return new_id


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
            "module": record.module,
            "line": record.lineno,
        }

        # Include custom extra metadata if present
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_data.update(record.extra_fields)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(name: str = "oncology_rag") -> logging.Logger:
    """Configures and returns a structured JSON logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        log_level_env = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level_env, logging.INFO))
        logger.propagate = False

    return logger
