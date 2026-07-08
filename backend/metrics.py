import numpy as np
import requests
import json
import re
import warnings
from functools import lru_cache
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

import settings
from modules.embeddings.mrl_embeddings import (
    get_mrl_embedding,
    prime_mrl_embedding_cache
)

OLLAMA_URL = "http://localhost:11434/api/generate"
EVAL_MODEL = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M "
#hf.co/QuantFactory/Llama3-Med42-8B-GGUF:Q4_K_M 

smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
    use_stemmer=True
)

_combined_llm_cache = {}


def _normalize_context_docs(contexts):

    if contexts is None:
        return []

    if isinstance(contexts, str):
        contexts = [contexts]

    normalized = []

    for context in contexts:

        if context is None:
            continue

        if isinstance(context, dict):
            context = (
                context.get("text")
                or context.get("content")
                or context.get("chunk")
                or ""
            )

        context = str(context).strip()

        if context:
            normalized.append(context)

    return normalized


@lru_cache(maxsize=20000)
def cached_lower(text):

    return text.lower()


@lru_cache(maxsize=20000)
def cached_split_tokens(text):

    return tuple(
        text.lower().split()
    )


def prepare_metric_embeddings(
    question,
    reference,
    prediction,
    contexts
):

    contexts = _normalize_context_docs(
        contexts
    )

    context_text = " ".join(
        contexts
    )

    texts = [
        question,
        reference,
        prediction,
        context_text
    ]

    texts.extend(
        contexts
    )

    prime_mrl_embedding_cache(
        [
            text
            for text in texts
            if text and text.strip()
        ]
    )


# -------------------------------
# 🔹 GENERATION METRICS
# -------------------------------
def compute_bleu_scores(ref, pred):
    try:
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        return {
            "bleu1": sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth),
            "bleu2": sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
            "bleu4": sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth),
        }
    except Exception:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

def compute_gleu_score(ref, pred):
    try:
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        if not pred_tokens:
            return {"gleu": 0.0}
        return {"gleu": sentence_gleu([ref_tokens], pred_tokens)}
    except Exception:
        return {"gleu": 0.0}

def compute_distinct_scores(pred):
    try:
        tokens = pred.lower().split()
        if len(tokens) < 2:
            return {"distinct1": 0.0, "distinct2": 0.0}

        unigrams = set(tokens)
        bigrams = set(zip(tokens[:-1], tokens[1:]))

        return {
            "distinct1": len(unigrams) / len(tokens),
            "distinct2": len(bigrams) / max(len(tokens) - 1, 1)
        }
    except Exception:
        return {"distinct1": 0.0, "distinct2": 0.0}

def compute_accuracy_f1(ref, pred):
    try:
        ref_tokens = set(ref.lower().split())
        pred_tokens = set(pred.lower().split())

        if not ref_tokens and not pred_tokens:
            return {"accuracy": 1.0, "f1": 1.0}
        if not ref_tokens or not pred_tokens:
            return {"accuracy": 0.0, "f1": 0.0}

        overlap = len(ref_tokens & pred_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)

        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        accuracy = float(ref.strip().lower() == pred.strip().lower())

        return {"accuracy": accuracy, "f1": f1}
    except Exception:
        return {"accuracy": 0.0, "f1": 0.0}

def compute_rouge_scores(ref, pred):
    try:
        scores = rouge.score(ref, pred)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
            "rougeLsum": scores["rougeLsum"].fmeasure
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

_bert_scorer = None

def compute_bertscore(refs, preds):
    global _bert_scorer
    try:
        if isinstance(refs, str):
            refs = [refs]
        if isinstance(preds, str):
            preds = [preds]
        if _bert_scorer is None:
            _bert_scorer = BERTScorer(lang="en")
        P, R, F1 = _bert_scorer.score(preds, refs)
        return float(F1.mean())
    except Exception:
        return 0.0

def compute_meteor_score(ref, pred):
    try:
        return {"meteor": meteor_score([word_tokenize(ref)], word_tokenize(pred))}
    except Exception:
        return {"meteor": 0.0}

def compute_sbert_similarity(ref, pred):
    if not ref.strip() or not pred.strip():
        return 0.0
    try:
        embeddings = get_mrl_embedding(
            [ref, pred],
            log=False
        )
        return float(util.cos_sim(embeddings[0], embeddings[1]).item())
    except Exception as e:
        print("SBERT similarity failed:", e)
        return 0.0

def combined_llm_evaluation(question, reference, prediction):

    cache_key = (
        question,
        reference,
        prediction
    )

    if cache_key in _combined_llm_cache:

        return _combined_llm_cache[cache_key]

    prompt = f"""
You are an evaluator for an oncology QA system.

Evaluate the prediction against the reference answer using BOTH:
1. A general LLM-as-judge score from 0.0 to 1.0.
2. The S.C.O.P.E framework from 1.0 to 5.0 for each metric:
- Safety
- Completeness
- Originality
- Precision
- Efficiency

Return ONLY valid JSON:
{{
  "llm_judge_score": 0.0,
  "llm_judge_reason": "short reason",
  "scope_safety": 0.0,
  "scope_completeness": 0.0,
  "scope_originality": 0.0,
  "scope_precision": 0.0,
  "scope_efficiency": 0.0
}}

QUESTION:
{question}

REFERENCE:
{reference[:1200]}

PREDICTION:
{prediction[:1200]}
"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": EVAL_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "top_p": 0.1,
                    "num_predict": 220,
                    "keep_alive": "20m"
                }
            },
            timeout=60
        )
        raw = response.json().get("response", "")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON judge output")
        parsed = json.loads(match.group(0))

        judge_score = float(parsed.get("llm_judge_score", 0))
        safety = float(parsed.get("scope_safety", 0))
        completeness = float(parsed.get("scope_completeness", 0))
        originality = float(parsed.get("scope_originality", 0))
        precision = float(parsed.get("scope_precision", 0))
        efficiency = float(parsed.get("scope_efficiency", 0))
        # Properly weighted SCOPE calculation.
        # Safety is most critical for medical QA.
        weighted = (
            0.35 * safety +
            0.25 * completeness +
            0.20 * precision +
            0.10 * efficiency +
            0.10 * originality
        )

        result = {
            "llm_judge_score": max(0, min(judge_score, 1)),
            "llm_judge_reason": str(parsed.get("llm_judge_reason", ""))[:240],
            "scope_safety": round(max(1.0, min(safety, 5.0)), 2),
            "scope_completeness": round(max(1.0, min(completeness, 5.0)), 2),
            "scope_originality": round(max(1.0, min(originality, 5.0)), 2),
            "scope_precision": round(max(1.0, min(precision, 5.0)), 2),
            "scope_efficiency": round(max(1.0, min(efficiency, 5.0)), 2),
            "scope_weighted_total": round(max(1.0, min(weighted, 5.0)), 2)
        }

        _combined_llm_cache[cache_key] = result

        return result

    except Exception as e:

        print("❌ LLM evaluation failed:", e)
        print("⚠️ Falling back to hybrid scoring...")

        try:
            bleu_score = compute_bleu_scores(
                reference[:1200],
                prediction[:1200]
            ).get("bleu4", 0.0)

            sbert_score = compute_sbert_similarity(
                reference[:1200],
                prediction[:1200]
            )

            faith_score = compute_faithfulness(
                prediction[:1200],
                [reference[:1200]]
            )

            hybrid_score = (
                0.4 * bleu_score +
                0.4 * sbert_score +
                0.2 * faith_score
            )

            scope_score = min(
                1.0 + (hybrid_score * 4.0),
                5.0
            )

            result = {
                "llm_judge_score": hybrid_score,
                "llm_judge_reason": "fallback_hybrid_score",
                "scope_safety": round(scope_score, 2),
                "scope_completeness": round(scope_score, 2),
                "scope_originality": round(max(1.0, scope_score * 0.8), 2),
                "scope_precision": round(scope_score, 2),
                "scope_efficiency": round(max(1.0, scope_score * 0.9), 2),
                "scope_weighted_total": round(scope_score, 2)
            }

            _combined_llm_cache[cache_key] = result
            return result

        except Exception as fallback_error:

            print("⚠️ Fallback also failed:", fallback_error)

            result = {
                "llm_judge_score": 0.0,
                "llm_judge_reason": "critical_failure_all_systems",
                "scope_safety": 1.0,
                "scope_completeness": 1.0,
                "scope_originality": 1.0,
                "scope_precision": 1.0,
                "scope_efficiency": 1.0,
                "scope_weighted_total": 1.0
            }

            _combined_llm_cache[cache_key] = result
            return result


def llm_as_judge_score(question, reference, prediction):

    result = combined_llm_evaluation(
        question,
        reference,
        prediction
    )

    return {
        "llm_judge_score": result.get(
            "llm_judge_score",
            0.0
        ),
        "llm_judge_reason": result.get(
            "llm_judge_reason",
            "judge_failed"
        )
    }


def scope_llm_judge(question, reference, prediction):

    result = combined_llm_evaluation(
        question,
        reference,
        prediction
    )

    return {
        "scope_safety": result.get("scope_safety", 0.0),
        "scope_completeness": result.get("scope_completeness", 0.0),
        "scope_originality": result.get("scope_originality", 0.0),
        "scope_precision": result.get("scope_precision", 0.0),
        "scope_efficiency": result.get("scope_efficiency", 0.0),
        "scope_weighted_total": result.get("scope_weighted_total", 0.0)
    }

# -------------------------------
# 🔹 RETRIEVAL METRICS
# -------------------------------
def _retrieval_threshold(threshold=None):

    if threshold is not None:
        return float(threshold)

    return settings.retrieval_relevance_threshold()


def semantic_relevance_scores(query, chunks):
    """
    Return cosine_similarity(embedding(query), embedding(chunk)) per chunk.
    Assumption: no human relevance labels exist, so semantic similarity is used
    as a weak relevance signal. Limitation: similarity can over-credit passages
    that are topically related but clinically incomplete or incorrect.
    """
    try:
        docs = _normalize_context_docs(
            chunks
        )

        if not query.strip() or not docs:
            return []

        embeddings = get_mrl_embedding(
            [query] + docs,
            log=False
        )

        q_emb = embeddings[0]
        doc_embs = embeddings[1:]

        return [
            float(util.cos_sim(q_emb, doc_emb).item())
            for doc_emb in doc_embs
        ]

    except Exception:
        return []


def semantic_relevance_labels(query, chunks, threshold=None):
    """
    relevant(chunk) = 1 when cosine_similarity(query, chunk) >= threshold,
    else 0. This avoids fabricating labels from retrieved IDs.
    """
    cutoff = _retrieval_threshold(
        threshold
    )

    return [
        1 if score >= cutoff else 0
        for score in semantic_relevance_scores(
            query,
            chunks
        )
    ]


def precision_at_k(chunks, query, k=5, threshold=None):
    """
    Precision@K = (# relevant documents in top K) / K.
    Uses semantic binary labels because this dataset has no judged relevant
    documents.
    """
    try:
        labels = semantic_relevance_labels(
            query,
            chunks,
            threshold
        )

        if not labels:
            return 0.0

        return sum(labels[:k]) / k

    except Exception:
        return 0.0


def recall_at_k(chunks, query, k=5, candidate_pool=None, threshold=None):
    """
    Recall@K = (# relevant documents retrieved in top K) /
    (# relevant documents available in candidate pool).
    Without manual labels or a fully judged corpus, candidate_pool is the
    evaluator-visible retrieved pool, so this is proxy recall.
    """
    try:
        retrieved_labels = semantic_relevance_labels(
            query,
            chunks,
            threshold
        )

        pool = candidate_pool if candidate_pool is not None else chunks

        pool_labels = semantic_relevance_labels(
            query,
            pool,
            threshold
        )

        relevant_available = sum(pool_labels)

        if relevant_available == 0:
            return 0.0

        return sum(retrieved_labels[:k]) / relevant_available

    except Exception:
        return 0.0


def mrr(chunks, query, threshold=None):
    """
    MRR = 1 / rank_of_first_relevant_document.
    Ranks are one-based and relevance comes from the semantic binary label.
    """
    try:
        labels = semantic_relevance_labels(
            query,
            chunks,
            threshold
        )

        for i, label in enumerate(labels):

            if label:
                return 1.0 / (i + 1)

        return 0.0

    except Exception:
        return 0.0


def hit_rate(chunks, query, k=5, threshold=None):
    """
    HitRate@K = 1 if at least one relevant document exists in top K else 0.
    """
    try:
        labels = semantic_relevance_labels(
            query,
            chunks,
            threshold
        )

        return float(
            int(
                any(labels[:k])
            )
        )

    except Exception:
        return 0.0


def ndcg(chunks, query, k=5, threshold=None):
    """
    DCG@K = sum((2^rel_i - 1) / log2(i + 1)).
    NDCG@K = DCG@K / IDCG@K.
    rel_i is the semantic binary relevance label at one-based rank i.
    """
    try:
        labels = semantic_relevance_labels(
            query,
            chunks,
            threshold
        )

        if not labels:
            return 0.0

        dcg = 0.0

        for i, rel in enumerate(labels[:k], start=1):
            dcg += (
                (2 ** rel - 1)
                / np.log2(i + 1)
            )

        ideal_labels = sorted(
            labels,
            reverse=True
        )[:k]

        ideal = sum(
            (2 ** rel - 1) / np.log2(i + 1)
            for i, rel in enumerate(
                ideal_labels,
                start=1
            )
        )

        return dcg / ideal if ideal > 0 else 0.0

    except Exception:
        return 0.0

# -------------------------------
# 🔹 FAITHFULNESS & RELEVANCE
# -------------------------------
def compute_faithfulness(answer, contexts):
    """
    Compute semantic faithfulness: how well the answer is grounded in contexts.
    Uses embedding similarity instead of word overlap.
    Returns: 0.0 to 1.0
    """
    try:
        context_docs = _normalize_context_docs(
            contexts
        )

        if not answer.strip():
            return 0.0

        if not context_docs:
            warnings.warn(
                "Faithfulness skipped: empty context received.",
                RuntimeWarning
            )
            print(
                "Faithfulness skipped: empty context received."
            )
            return 0.0

        answer_emb = get_mrl_embedding(
            [answer],
            log=False
        )[0]

        context_embs = get_mrl_embedding(
            context_docs,
            log=False
        )

        if len(context_embs) == 0:
            warnings.warn(
                "Faithfulness skipped: empty context embeddings.",
                RuntimeWarning
            )
            print(
                "Faithfulness skipped: empty context embeddings."
            )
            return 0.0

        max_similarity = max(
            float(util.cos_sim(answer_emb, ctx_emb).item())
            for ctx_emb in context_embs
        )

        return max(0.0, min(max_similarity, 1.0))

    except Exception:
        return 0.0

def answer_relevance(answer, question):
    """
    Compute how well the answer addresses the question.
    Uses semantic similarity instead of word matching.
    Returns: 0.0 to 1.0
    """
    try:
        if not answer.strip() or not question.strip():
            return 0.0

        q_emb = get_mrl_embedding(
            [question],
            log=False
        )[0]

        a_emb = get_mrl_embedding(
            [answer],
            log=False
        )

        similarity = float(
            util.cos_sim(q_emb, a_emb).item()
        )

        return max(0.0, min(similarity, 1.0))

    except Exception:
        return 0.0

def context_relevance(contexts, question):
    """
    Compute how relevant the retrieved contexts are to the question.
    Uses semantic similarity instead of word overlap.
    Returns: 0.0 to 1.0
    """
    try:
        context_docs = _normalize_context_docs(
            contexts
        )

        if not question.strip():
            return 0.0

        if not context_docs:
            warnings.warn(
                "Context relevancy skipped: empty context received.",
                RuntimeWarning
            )
            print(
                "Context relevancy skipped: empty context received."
            )
            return 0.0

        q_emb = get_mrl_embedding(
            [question],
            log=False
        )[0]

        context_embs = get_mrl_embedding(
            context_docs,
            log=False
        )

        if len(context_embs) == 0:
            warnings.warn(
                "Context relevancy skipped: empty context embeddings.",
                RuntimeWarning
            )
            print(
                "Context relevancy skipped: empty context embeddings."
            )
            return 0.0

        similarities = [
            float(util.cos_sim(q_emb, ctx_emb).item())
            for ctx_emb in context_embs
        ]

        avg_relevance = (
            sum(similarities) / len(similarities)
        ) if similarities else 0.0

        return max(0.0, min(avg_relevance, 1.0))

    except Exception:
        return 0.0
