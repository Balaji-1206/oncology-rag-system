import html
import re
from typing import Optional, Tuple


class SecurityGuardrails:
    """Security filters for input validation, prompt injection, and sanitization."""

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"reveal\s+(the\s+)?(system\s+prompt|hidden\s+instructions)", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(rules|guidelines)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(an\s+unrestricted|in\s+dan\s+mode)", re.IGNORECASE),
        re.compile(r"(system\s+prompt|developer\s+mode)\s*:\s*override", re.IGNORECASE),
        re.compile(r"output\s+initial\s+prompt", re.IGNORECASE),
        re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript\s*:", re.IGNORECASE),
    ]

    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 3

    @classmethod
    def validate_and_sanitize(cls, query: str) -> Tuple[bool, Optional[str], str]:
        """
        Validates and sanitizes a query string.
        Returns:
            (is_valid, error_message, sanitized_query)
        """
        if not isinstance(query, str):
            return False, "Query must be a text string", ""

        cleaned = query.strip()

        if not cleaned:
            return False, "Empty query: input cannot be blank", ""

        if len(cleaned) < cls.MIN_QUERY_LENGTH:
            return False, f"Query too short (minimum {cls.MIN_QUERY_LENGTH} characters)", ""

        if len(cleaned) > cls.MAX_QUERY_LENGTH:
            return False, f"Query too long: exceeds maximum length of {cls.MAX_QUERY_LENGTH} characters", ""

        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(cleaned):
                return False, "Security violation: potentially adversarial pattern or script detected", ""

        # Neutralize HTML entities to prevent stored/reflected XSS in dashboards
        escaped = html.escape(cleaned, quote=False)

        return True, None, escaped


def validate_query_security(query: str) -> Tuple[bool, Optional[str], str]:
    """Helper function to run security guardrail validation on query."""
    return SecurityGuardrails.validate_and_sanitize(query)
