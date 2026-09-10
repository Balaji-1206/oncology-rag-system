import re
from typing import Dict, List, Tuple


class PHIScrubber:
    """
    HIPAA-compliant clinical text de-identification engine.
    Detects and scrubs Protected Health Information (PHI) from patient queries.
    """

    PATTERNS: List[Tuple[str, re.Pattern, str]] = [
        (
            "SSN",
            re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
            "[REDACTED_SSN]"
        ),
        (
            "EMAIL",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "[REDACTED_EMAIL]"
        ),
        (
            "PHONE",
            re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
            "[REDACTED_PHONE]"
        ),
        (
            "MRN",
            re.compile(
                r"\b(?:MRN|mrn|medical record number|record id|patient id)[:\s#]*[A-Za-z0-9-]{4,15}\b",
                re.IGNORECASE
            ),
            "[REDACTED_MRN]"
        ),
        (
            "DOB",
            re.compile(
                r"\b(?:DOB|dob|date of birth|born)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                re.IGNORECASE
            ),
            "[REDACTED_DOB]"
        ),
        (
            "PATIENT_NAME_PREFIX",
            re.compile(
                r"\b(?:patient|pt|mr\.|mrs\.|ms\.)[:\s]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
                re.IGNORECASE
            ),
            "[REDACTED_PATIENT]"
        ),
        (
            "PATIENT_NAME_EXPLICIT",
            re.compile(
                r"\bname[:\s]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
                re.IGNORECASE
            ),
            "name: [REDACTED_PATIENT]"
        ),
    ]

    @classmethod
    def scrub(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Scrubs clinical identifiers from text.
        Returns:
            (sanitized_text, redaction_summary)
        """
        if not text:
            return "", {}

        sanitized = text
        redactions: Dict[str, int] = {}

        for phi_type, pattern, replacement in cls.PATTERNS:
            matches = pattern.findall(sanitized)
            if matches:
                redactions[phi_type] = len(matches)
                sanitized = pattern.sub(replacement, sanitized)

        return sanitized, redactions

    @classmethod
    def contains_phi(cls, text: str) -> bool:
        """Checks if text contains potential Protected Health Information."""
        if not text:
            return False
        return any(pattern.search(text) for _, pattern, _ in cls.PATTERNS)


def scrub_phi(text: str) -> Tuple[str, Dict[str, int]]:
    """Helper function to scrub PHI from clinical text."""
    return PHIScrubber.scrub(text)
