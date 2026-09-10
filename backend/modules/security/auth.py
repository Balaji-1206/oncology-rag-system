import os
from functools import wraps
from typing import Optional, Set, Tuple
from flask import request, jsonify


class AuthManager:
    """Manages API key authentication and role-based authorization."""

    def __init__(self) -> None:
        self.user_keys: Set[str] = set()
        self.admin_keys: Set[str] = set()
        self.auth_enabled: bool = False
        self.reload_keys()

    def reload_keys(self) -> None:
        """Loads valid API keys from environment configuration."""
        raw_user_keys = os.environ.get("API_KEYS", "")
        raw_admin_keys = os.environ.get("ADMIN_API_KEYS", "")
        auth_enabled_env = os.environ.get("AUTH_ENABLED", "").strip().lower()

        self.user_keys = {
            key.strip() for key in raw_user_keys.split(",") if key.strip()
        }
        self.admin_keys = {
            key.strip() for key in raw_admin_keys.split(",") if key.strip()
        }

        if auth_enabled_env in {"true", "1", "yes"}:
            self.auth_enabled = True
        elif auth_enabled_env in {"false", "0", "no"}:
            self.auth_enabled = False
        else:
            # Enabled if any keys are defined, disabled otherwise for dev convenience
            self.auth_enabled = bool(self.user_keys or self.admin_keys)

    def extract_key(self) -> Optional[str]:
        """Extracts API key from X-API-Key header or Bearer authorization header."""
        header_key = request.headers.get("X-API-Key")
        if header_key:
            return header_key.strip()

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[len("Bearer "):].strip()

        query_key = request.args.get("api_key")
        if query_key:
            return query_key.strip()

        return None

    def verify_key(self, api_key: Optional[str]) -> Tuple[bool, str]:
        """
        Verifies API key and determines assigned role.
        Returns (is_valid, role).
        """
        if not self.auth_enabled:
            return True, "admin"

        if not api_key:
            return False, "anonymous"

        if api_key in self.admin_keys:
            return True, "admin"

        if api_key in self.user_keys:
            return True, "user"

        return False, "anonymous"


_AUTH_MANAGER = AuthManager()


def require_auth(view_func):
    """Decorator requiring a valid user or admin API key."""
    @wraps(view_func)
    def decorated(*args, **kwargs):
        api_key = _AUTH_MANAGER.extract_key()
        is_valid, role = _AUTH_MANAGER.verify_key(api_key)

        if not is_valid:
            return jsonify({
                "error": "Unauthorized: valid API key required",
                "code": "AUTH_REQUIRED"
            }), 401

        request.user_role = role
        return view_func(*args, **kwargs)
    return decorated


def require_admin(view_func):
    """Decorator requiring administrative API privileges."""
    @wraps(view_func)
    def decorated(*args, **kwargs):
        api_key = _AUTH_MANAGER.extract_key()
        is_valid, role = _AUTH_MANAGER.verify_key(api_key)

        if not is_valid:
            return jsonify({
                "error": "Unauthorized: valid API key required",
                "code": "AUTH_REQUIRED"
            }), 401

        if role != "admin":
            return jsonify({
                "error": "Forbidden: administrative privileges required",
                "code": "FORBIDDEN"
            }), 403

        request.user_role = role
        return view_func(*args, **kwargs)
    return decorated


def get_auth_manager() -> AuthManager:
    """Returns singleton AuthManager instance."""
    return _AUTH_MANAGER
