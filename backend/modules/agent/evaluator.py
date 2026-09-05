import json
import logging
import re
import requests
import numpy as np
from sentence_transformers import util

import settings
from modules.embeddings.mrl_embeddings import get_mrl_embedding

logger = logging.getLogger(__name__)

SESSION = requests.Session()
PHI3MINI_URL = "http://localhost:11434/api/generate"
EVAL_MODEL = "phi3:mini"


def extract_json(text: str) -> dict:
    """Extracts JSON structure from text response."""
    text = text.replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found")
    return json.loads(match.group(0))


def tokenize(text: str) -> set:
    """Extracts set of clean token words."""
    return set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))


def grounding_score(answer: str, context: str) -> float:
    """Computes lexical token overlap between answer and context."""
    answer_words = tokenize(answer)
    context_words = tokenize(context)
    if not answer_words:
        return 0.0
    overlap = len(answer_words & context_words)
    return overlap / max(len(answer_words), 1)


def semantic_grounding_score(answer: str, context: str) -> float:
    """Computes semantic similarity embedding score between answer and context sentences."""
    if not answer.strip() or not context.strip():
        return 0.0

    try:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 30]
        if not sentences:
            sentences = [context[:1200]]

        sentences = sentences[:10]
        embeddings = get_mrl_embedding([answer[:1200]] + sentences, log=False)
        answer_embedding = embeddings[0]
        context_embeddings = embeddings[1:]

        similarities = util.cos_sim(answer_embedding, context_embeddings)[0]
        top_scores = similarities.topk(min(3, len(sentences))).values
        return float(np.mean(top_scores.cpu().numpy()))
    except Exception:
        return 0.0


def compute_combined_grounding(answer: str, context: str) -> tuple:
    """Combines lexical and semantic grounding scores."""
    lex_score = grounding_score(answer, context)
    sem_score = semantic_grounding_score(answer, context)

    if sem_score > 0:
        combined = (0.35 * lex_score) + (0.65 * sem_score)
    else:
        combined = lex_score

    return combined, lex_score, sem_score


def evaluate_answer(query: str, docs: list, answer: str, retrieval_score: float = 0.0, intent: str = "factual", query_type: str = "general") -> dict:
    """Evaluates generated answer for quality, grounding, relevance, and hallucination risk."""
    context = " ".join(docs)
    combined_grounding, lexical_grounding, semantic_grounding = compute_combined_grounding(answer, context)

    prompt = f"""
Evaluate this medical QA response against the provided context.

Query: {query}
Context: {context[:2500]}
Generated Answer: {answer}

Output JSON with exact fields:
{{
  "answered_question": true/false,
  "answer_relevance": 0.0-1.0,
  "missing_information": true/false,
  "hallucination_risk": "low"/"medium"/"high",
  "contradiction_risk": 0.0-1.0,
  "refusal_detected": true/false,
  "insufficient_structure": true/false,
  "missing_ranking": true/false,
  "has_coverage_gap": true/false,
  "score": 1-10,
  "confidence": 0.0-1.0,
  "reasoning": "short explanation"
}}
"""

    try:
        response = SESSION.post(
            PHI3MINI_URL,
            json={
                "model": EVAL_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 150}
            },
            timeout=25
        )
        res_text = response.json().get("response", "")
        data = extract_json(res_text)

        score = data.get("score", 5)
        confidence = data.get("confidence", 0.5)
        answer_relevance = data.get("answer_relevance", 0.5)
        missing_information = data.get("missing_information", False)
        hallucination_risk = data.get("hallucination_risk", "medium")
        contradiction = data.get("contradiction_risk", 0.0)
        refusal = data.get("refusal_detected", False)
        answered_question = data.get("answered_question", True)
        insufficient_structure = data.get("insufficient_structure", False)
        missing_ranking = data.get("missing_ranking", False)
        has_coverage_gap = data.get("has_coverage_gap", False)

        grounding = combined_grounding

        # Grounding-based adjustments
        if grounding < 0.20 and score > 5:
            score = max(score - 3, 3)
            hallucination_risk = "high"
        elif grounding < 0.40 and score > 7:
            score = max(score - 2, 5)

        # Retry logic
        retry = False
        if score < 7 or grounding < 0.25 or contradiction > 0.35 or missing_information or insufficient_structure or missing_ranking or has_coverage_gap:
            retry = True

        if refusal and grounding > 0.15:
            retry = True
            score = min(score, 4)

        # High Quality override
        if (score >= 8 and confidence >= 0.75 and answer_relevance >= 0.75 and grounding >= 0.55 and contradiction <= 0.25
                and not missing_information and not insufficient_structure and not missing_ranking and not has_coverage_gap):
            retry = False

        if confidence > 0.80 and (grounding < 0.70 or answer_relevance < 0.80 or contradiction > 0.20 or missing_information):
            confidence = 0.80

        _eval_retrieval_score = max(0.0, min(1.0, (0.55 * grounding) + (0.25 * answer_relevance) + (0.20 * confidence)))

        return {
            "score": score,
            "confidence": round(confidence, 2),
            "needs_retry": retry,
            "answered_question": answered_question,
            "answer_relevance": round(answer_relevance, 2),
            "hallucination_risk": hallucination_risk,
            "missing_information": missing_information,
            "grounding_score": round(grounding, 2),
            "retrieval_score": round(_eval_retrieval_score, 2),
            "lexical_grounding_score": round(lexical_grounding, 2),
            "semantic_grounding_score": round(semantic_grounding, 2),
            "contradiction_risk": round(contradiction, 2),
            "refusal_detected": refusal,
            "is_fallback": False,
            "evaluator_mode": "llm_evaluator"
        }

    except Exception as e:
        logger.warning(
            "Ollama evaluator call to %s failed (%s); switching to deterministic grounding fallback.",
            EVAL_MODEL,
            e
        )
        words = answer.strip().split()
        word_count = len(words)
        has_text = word_count >= 5
        grounding = combined_grounding

        # Calibrated evidence-based fallback scoring
        if grounding >= 0.70 and word_count >= 15:
            score = 8
            conf = min(0.78, round(grounding, 2))
            retry = False
            risk = "low"
            rel = round(grounding, 2)
        elif grounding >= 0.50 and word_count >= 10:
            score = 7
            conf = min(0.70, round(grounding, 2))
            retry = False
            risk = "low" if grounding >= 0.60 else "medium"
            rel = round(grounding, 2)
        elif grounding >= 0.35 and has_text:
            score = 6
            conf = 0.60
            retry = False
            risk = "medium"
            rel = round(grounding, 2)
        elif grounding >= 0.20 and has_text:
            score = 5
            conf = 0.48
            retry = True
            risk = "medium"
            rel = round(grounding, 2)
        elif grounding >= 0.10 and has_text:
            score = 4
            conf = 0.35
            retry = True
            risk = "high"
            rel = round(grounding, 2)
        else:
            score = 2
            conf = 0.20
            retry = True
            risk = "high"
            rel = 0.20

        eval_retrieval = max(0.0, min(1.0, (0.55 * grounding) + (0.25 * rel) + (0.20 * conf)))

        return {
            "score": score,
            "confidence": round(conf, 2),
            "needs_retry": retry,
            "answered_question": has_text,
            "answer_relevance": round(rel, 2),
            "hallucination_risk": risk,
            "missing_information": (score < 7),
            "grounding_score": round(grounding, 2),
            "retrieval_score": round(eval_retrieval, 2),
            "lexical_grounding_score": round(lexical_grounding, 2),
            "semantic_grounding_score": round(semantic_grounding, 2),
            "contradiction_risk": 0.05,
            "refusal_detected": False,
            "is_fallback": True,
            "evaluator_mode": "grounding_fallback",
            "fallback_reason": str(e)
        }
