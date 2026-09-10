import re
from typing import Dict, List, Set, Tuple


class ClinicalNegationDetector:
    """
    Deterministic clinical negation and contraindication detector.
    Identifies clinical contradictions and negated assertions between
    generated answers and retrieved evidence chunks.
    """

    NEGATION_TRIGGERS: List[re.Pattern] = [
        re.compile(r"\b(?:contraindicated|contraindication|contraindications)\b", re.IGNORECASE),
        re.compile(r"\b(?:not recommended|do not administer|must not be used|avoid in)\b", re.IGNORECASE),
        re.compile(r"\b(?:black box warning|fatal risk|life-threatening toxicity)\b", re.IGNORECASE),
        re.compile(r"\b(?:failed to show|no significant benefit|no improvement|ineffective)\b", re.IGNORECASE),
        re.compile(r"\b(?:no evidence of|negative for|absence of|lacks)\b", re.IGNORECASE),
        re.compile(r"\b(?:refractory to|resistant to|progression despite)\b", re.IGNORECASE),
        re.compile(r"\b(?:discontinue immediately|treatment cessation)\b", re.IGNORECASE),
    ]

    ASSERTION_TRIGGERS: List[re.Pattern] = [
        re.compile(r"\b(?:is recommended|first-line therapy|standard of care|indicated for)\b", re.IGNORECASE),
        re.compile(r"\b(?:demonstrated superior|highly effective|significant improvement)\b", re.IGNORECASE),
        re.compile(r"\b(?:should be administered|preferred regimen|approved for)\b", re.IGNORECASE),
    ]

    @classmethod
    def extract_negations(cls, text: str) -> List[Dict[str, str]]:
        """Extracts negated phrases and their surrounding sentence context."""
        negations = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            sent_clean = sent.strip()
            for pattern in cls.NEGATION_TRIGGERS:
                match = pattern.search(sent_clean)
                if match:
                    negations.append({
                        "trigger": match.group(0),
                        "sentence": sent_clean
                    })
                    break

        return negations

    @classmethod
    def detect_contradiction(
        cls,
        answer: str,
        context: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Analyzes whether the generated answer contradicts the retrieved context.
        Returns:
            (has_contradiction, contradiction_risk, reasons)
        """
        if not answer or not context:
            return False, 0.0, []

        context_negations = cls.extract_negations(context)
        reasons = []

        if not context_negations:
            return False, 0.0, []

        answer_lower = answer.lower()
        contradiction_count = 0

        for item in context_negations:
            trigger = item["trigger"].lower()
            context_sent = item["sentence"]

            # Extract subject tokens near the negation trigger in context
            context_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", context_sent.lower()))
            answer_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", answer_lower))

            # Stopwords to filter out
            stopwords = {"with", "that", "this", "from", "patient", "patients", "treatment", "therapy", "clinical"}
            shared_entities = (context_words & answer_words) - stopwords

            # If the context explicitly negates or contraindicates an entity,
            # but the answer recommends it positively without mentioning the contraindication:
            if len(shared_entities) >= 1:
                # Check if answer contains positive recommendation terms for this entity
                has_positive_claim = any(p.search(answer) for p in cls.ASSERTION_TRIGGERS)
                has_caveat = any(p.search(answer) for p in cls.NEGATION_TRIGGERS)

                if has_positive_claim and not has_caveat:
                    contradiction_count += 1
                    reasons.append(
                        f"Context notes contraindication/negation for {list(shared_entities)[:2]} ('{trigger}'), "
                        f"but answer makes an unreserved positive recommendation."
                    )

        if contradiction_count > 0:
            risk = min(1.0, 0.40 + (contradiction_count * 0.30))
            return True, risk, reasons

        return False, 0.0, []


def detect_clinical_contradiction(answer: str, context: str) -> Tuple[bool, float, List[str]]:
    """Helper function to run clinical contradiction analysis."""
    return ClinicalNegationDetector.detect_contradiction(answer, context)
