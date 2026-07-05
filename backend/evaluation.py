import argparse
import contextlib
import io
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from tqdm import tqdm

import settings
import app as app_module
from app import handle_query
from modules.agent import agent_controller as agent_controller_module

from metrics import *


# =========================================================
# 🔹 LOAD DATASET
# =========================================================
def load_dataset(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =========================================================
# 🔹 SAFE AVG
# =========================================================
def avg(results, key):

    vals = [

        r.get(key, 0)

        for r in results
    ]

    if not vals:
        return 0

    return float(np.mean(vals))


def _report_metric(results, key, direct_mode=False, digits=4):

    if direct_mode:
        return "N/A"

    return f"{avg(results, key):.{digits}f}"


# =========================================================
# 🔹 QUERY TYPE
# =========================================================
def detect_query_type(q):

    q = q.lower()

    if "what is" in q:
        return "definition"

    if "symptom" in q:
        return "symptoms"

    if q.startswith(
        ("can", "does", "is", "are")
    ):
        return "yesno"

    if (
        "compare" in q
        or "difference" in q
    ):
        return "comparison"

    return "general"


# =========================================================
# 🔹 REFUSAL
# =========================================================
def refusal_detected(answer):

    patterns = [

        "not enough information",

        "insufficient information",

        "cannot determine",
    ]

    answer = answer.lower()

    return any(
        p in answer
        for p in patterns
    )


# =========================================================
# 🔹 GENERIC HELPERS
# =========================================================
def _coerce_float(value):

    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float, np.number)):
        if math.isfinite(float(value)):
            return float(value)
        return None

    if isinstance(value, str):
        text = value.strip().replace(",", "")

        if text.endswith("%"):
            text = text[:-1].strip()

        try:
            parsed = float(text)
        except ValueError:
            return None

        if math.isfinite(parsed):
            return parsed

    return None


def _coerce_bool(value):

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float, np.number)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes", "y"}:
            return True

        if normalized in {"false", "0", "no", "n", "none", "null"}:
            return False

    return None


def _load_json_payload(path):

    with open(path, "r", encoding="utf-8") as handle:

        try:
            return json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)
            records = []

            for line in handle:
                line = line.strip()

                if not line:
                    continue

                records.append(json.loads(line))

            return records


def _normalize_records(payload):

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):

        for key in ("results", "records", "question_records", "data"):
            value = payload.get(key)

            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        return [payload]

    return []


def load_records_from_path(path):

    return _normalize_records(
        _load_json_payload(path)
    )


def _results_directory():

    return os.path.join(
        os.path.dirname(__file__),
        "results"
    )


def _experiment_result_filename():

    if not settings.is_rag_enabled():

        return "direct_llm.json"

    laqa_state = "on" if settings.is_laqa_enabled() else "off"
    mrl_state = "on" if settings.is_mrl_enabled() else "off"

    return f"laqa_{laqa_state}_mrl_{mrl_state}.json"


def _serialize_experiment_record(record):

    return {
        "question_id": record.get("question_id", 0),
        "precision_at_5": record.get("precision", 0),
        "recall_at_5": record.get("recall", 0),
        "mrr": record.get("mrr", 0),
        "ndcg_at_5": record.get("ndcg", 0),
        "hit_rate": record.get("hit_rate", 0),
        "bleu4": record.get("bleu4", 0),
        "rougeL": record.get("rougeL", 0),
        "meteor": record.get("meteor", 0),
        "bertscore": record.get("bertscore", 0),
        "sbert_similarity": record.get("sbert_similarity", 0),
        "faithfulness": record.get("faithfulness", 0),
        "grounding_score": record.get("grounding_score", 0),
        "answer_relevancy": record.get("answer_rel", 0),
        "context_relevancy": record.get("context_rel", 0),
        "hallucinated": record.get("hallucinated", 0),
        "retrieval_latency": record.get("retrieval_latency", 0),
        "agent_latency": record.get("agent_communication_latency", 0),
        "total_latency": record.get("total_response_time", 0),
    }


def _save_experiment_results(results):

    output_dir = _results_directory()
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        _experiment_result_filename()
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            [
                _serialize_experiment_record(record)
                for record in results
            ],
            handle,
            indent=2,
            ensure_ascii=False
        )

    return output_path


def _first_numeric(source, keys):

    if not isinstance(source, dict):
        return None

    for key in keys:

        if key in source:
            value = _coerce_float(source.get(key))

            if value is not None:
                return value

    for nested_key in ("metrics", "profile", "latency", "latencies", "timings", "performance"):

        nested = source.get(nested_key)

        if isinstance(nested, dict):
            value = _first_numeric(nested, keys)

            if value is not None:
                return value

    return None


LATENCY_FIELD_ALIASES = {
    "retrieval_latency": (
        "retrieval_latency",
        "retrieval_time",
        "retrieval_ms",
        "retrieval_latency_ms",
        "rag_retrieval_latency",
        "search_latency",
    ),
    "agent_communication_latency": (
        "agent_communication_latency",
        "agent_comm_latency",
        "communication_latency",
        "agent_latency",
        "comm_latency",
    ),
    "total_response_time": (
        "total_response_time",
        "total_time",
        "response_time",
        "elapsed_time",
        "latency",
    ),
}


def extract_latency_snapshot(record, metrics=None):

    merged = {}

    if isinstance(metrics, dict):
        merged.update(metrics)

    if isinstance(record, dict):
        merged.update(record)

    retrieval_latency = _first_numeric(
        merged,
        (
            "retrieval_latency",
            "retrieval_time",
            "retrieval_ms",
            "retrieval_latency_ms",
            "rag_retrieval_latency",
            "search_latency",
        )
    )

    agent_latency = _first_numeric(
        merged,
        (
            "agent_communication_latency",
            "agent_comm_latency",
            "communication_latency",
            "agent_latency",
            "comm_latency",
        )
    )

    total_response_time = _first_numeric(
        merged,
        (
            "total_response_time",
            "response_time",
            "total_time",
            "elapsed_time",
            "latency",
        )
    )

    return {
        "retrieval_latency": retrieval_latency,
        "agent_communication_latency": agent_latency,
        "total_response_time": total_response_time,
    }


def _numeric_series(results, key):

    series = []

    for item in results:

        if not isinstance(item, dict):
            continue

        value = _first_numeric(
            item,
            LATENCY_FIELD_ALIASES.get(
                key,
                (key,)
            )
        )

        if value is None:
            continue

        series.append(value)

    return series


def _summarize_series(values):

    clean_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]

    if not clean_values:
        return None

    arr = np.asarray(clean_values, dtype=float)
    mean_value = float(np.mean(arr))
    median_value = float(np.median(arr))
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    std_value = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    if arr.size > 1:
        sem = stats.sem(arr, nan_policy="omit")

        if sem is not None and math.isfinite(float(sem)):
            critical = float(stats.t.ppf(0.975, df=arr.size - 1))
            margin = critical * float(sem)
            ci_low = mean_value - margin
            ci_high = mean_value + margin
        else:
            ci_low = mean_value
            ci_high = mean_value
    else:
        ci_low = mean_value
        ci_high = mean_value

    return {
        "count": int(arr.size),
        "mean": mean_value,
        "median": median_value,
        "min": min_value,
        "max": max_value,
        "std": std_value,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_half_width": float(abs(ci_high - mean_value)),
    }


def _is_hallucinated(record):

    if not isinstance(record, dict):
        return False

    if "hallucinated" in record:
        value = _coerce_bool(record.get("hallucinated"))

        if value is not None:
            return value

    for key in ("hallucination_risk", "hallucination_state", "risk"):
        value = record.get(key)

        if value is None:
            continue

        value_text = str(value).strip().lower()

        if value_text in {"high", "hallucinated", "unsafe", "yes", "true", "1"}:
            return True

        if value_text in {"low", "safe", "false", "0", "no"}:
            return False

    if "hallucination_low" in record:
        safe_value = _coerce_bool(record.get("hallucination_low"))

        if safe_value is not None:
            return not safe_value

    return False


def hallucination_rate(results):

    total_questions = len(results)

    if total_questions == 0:
        return 0.0

    hallucinated = sum(
        1 for record in results
        if _is_hallucinated(record)
    )

    return (hallucinated / total_questions) * 100.0


def build_latency_stats(results, field_name):

    return _summarize_series(
        _numeric_series(results, field_name)
    )


def _run_query_with_latency_measurements(user_query):

    total_start = time.perf_counter()
    retrieval_latency = 0.0
    agent_latency = 0.0

    original_handle_agent_decision = app_module.agent_decision
    original_hybrid_search = agent_controller_module.hybrid_search

    def timed_agent_decision(*args, **kwargs):

        nonlocal agent_latency

        agent_start = time.perf_counter()

        try:
            return original_handle_agent_decision(*args, **kwargs)
        finally:
            agent_latency += time.perf_counter() - agent_start

    def timed_hybrid_search(*args, **kwargs):

        nonlocal retrieval_latency

        retrieval_start = time.perf_counter()

        try:
            return original_hybrid_search(*args, **kwargs)
        finally:
            retrieval_latency += time.perf_counter() - retrieval_start

    app_module.agent_decision = timed_agent_decision
    agent_controller_module.hybrid_search = timed_hybrid_search

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = handle_query(user_query)
    finally:
        app_module.agent_decision = original_handle_agent_decision
        agent_controller_module.hybrid_search = original_hybrid_search

    total_response_time = time.perf_counter() - total_start

    agent_communication_latency = max(
        0.0,
        agent_latency - retrieval_latency
    )

    return result, {
        "retrieval_latency": retrieval_latency,
        "agent_communication_latency": agent_communication_latency,
        "total_response_time": total_response_time,
    }


def _format_number(value, digits=4):

    if value is None:
        return "N/A"

    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"

    return f"{value:.{digits}f}"


def _print_latency_block(title, stats_dict):

    print(
        f"\n{title}"
    )

    print(
        "-" * 52
    )

    if not stats_dict:

        print(
            "No latency data available."
        )

        return

    print(
        f"Mean               : {_format_number(stats_dict['mean'])}"
    )

    print(
        f"Median             : {_format_number(stats_dict['median'])}"
    )

    print(
        f"Min                : {_format_number(stats_dict['min'])}"
    )

    print(
        f"Max                : {_format_number(stats_dict['max'])}"
    )

    print(
        f"Std                : {_format_number(stats_dict['std'])}"
    )

    print(
        f"P50                : {_format_number(stats_dict['p50'])}"
    )

    print(
        f"P90                : {_format_number(stats_dict['p90'])}"
    )

    print(
        f"P95                : {_format_number(stats_dict['p95'])}"
    )

    print(
        f"P99                : {_format_number(stats_dict['p99'])}"
    )

    print(
        f"95% CI             : {_format_number(stats_dict['mean'])} ± {_format_number(stats_dict['ci_half_width'])}"
        f" ({_format_number(stats_dict['ci_low'])}, {_format_number(stats_dict['ci_high'])})"
    )


# =========================================================
# 🔹 MAIN EVALUATION
# =========================================================
def evaluate(dataset_path):

    data = load_dataset(
        dataset_path
    )

    results = []

    predictions = []

    references = []

    print("\n🚀 Running Evaluation...\n")

    total_questions = min(
        2,
        len(data)
    )

    progress_bar = tqdm(
        data[:2],
        desc="🧪 Evaluating",
        ncols=120
    )

    for idx, item in enumerate(
        progress_bar,
        start=1
    ):

        direct_mode = not settings.is_rag_enabled()

        # =====================================================
        # 🔹 QUESTION HEADER
        # =====================================================
        q = item["q"]

        print(
            f"\n{'='*80}"
        )

        print(
            f"🧪 Q{idx}/{total_questions}"
        )

        print(
            f"❓ {q}"
        )

        print(
            f"{'='*80}\n"
        )

        gt = item["a"]

        query_type = detect_query_type(
            q
        )

        # =====================================================
        # 🔹 PIPELINE
        # =====================================================
        result, latency_snapshot = _run_query_with_latency_measurements(
            q
        )

        pred = result.get(
            "answer",
            ""
        )

        contexts = result.get(
            "source_texts",
            []
        )

        context_docs = result.get(
            "context_docs",
            []
        ) or contexts

        print(
            "EVALUATOR RECEIVED DOCS:",
            len(context_docs)
        )

        confidence = result.get(
            "confidence",
            0.5
        )

        explanation = result.get(
            "explanation",
            {}
        )

        evaluation = result.get(
            "evaluation",
            {}
        )

        # =====================================================
        # 🔹 GENERATION METRICS
        # =====================================================
        prepare_metric_embeddings(
            q,
            gt,
            pred,
            context_docs
        )

        print(
            "FAITHFULNESS DOC COUNT:",
            len(context_docs)
        )

        print(
            "CONTEXT RELEVANCY DOC COUNT:",
            len(context_docs)
        )

        question_bertscore = compute_bertscore(
            [gt],
            [pred]
        )

        with ThreadPoolExecutor(
            max_workers=8
        ) as executor:

            futures = {

                "bleu": executor.submit(
                    compute_bleu_scores,
                    gt,
                    pred
                ),

                "gleu": executor.submit(
                    compute_gleu_score,
                    gt,
                    pred
                ),

                "meteor": executor.submit(
                    compute_meteor_score,
                    gt,
                    pred
                ),

                "distinct": executor.submit(
                    compute_distinct_scores,
                    pred
                ),

                "accuracy_f1": executor.submit(
                    compute_accuracy_f1,
                    gt,
                    pred
                ),

                "rouge": executor.submit(
                    compute_rouge_scores,
                    gt,
                    pred
                ),

                "sbert_similarity": executor.submit(
                    compute_sbert_similarity,
                    gt,
                    pred
                ),

                "llm_metrics": executor.submit(
                    combined_llm_evaluation,
                    q,
                    gt,
                    pred
                ),

                "faithfulness": executor.submit(
                    compute_faithfulness,
                    pred,
                    context_docs
                ),

                "context_rel": executor.submit(
                    context_relevance,
                    context_docs,
                    q
                ),

                "answer_rel": executor.submit(
                    answer_relevance,
                    pred,
                    q
                ),

                "precision": executor.submit(
                    precision_at_k,
                    context_docs,
                    q
                ),

                "recall": executor.submit(
                    recall_at_k,
                    context_docs,
                    q,
                    5,
                    context_docs
                ),

                "mrr": executor.submit(
                    mrr,
                    context_docs,
                    q
                ),

                "ndcg": executor.submit(
                    ndcg,
                    context_docs,
                    q
                ),

                "hit_rate": executor.submit(
                    hit_rate,
                    context_docs,
                    q
                )
            }

            bleu = futures["bleu"].result()
            gleu = futures["gleu"].result()
            meteor = futures["meteor"].result()
            distinct = futures["distinct"].result()
            accuracy_f1 = futures["accuracy_f1"].result()
            rouge = futures["rouge"].result()

            sbert_similarity = {

                "sbert_similarity": futures[
                    "sbert_similarity"
                ].result()
            }

            llm_all = futures["llm_metrics"].result()

            llm_judge = {

                "llm_judge_score": llm_all.get(
                    "llm_judge_score",
                    0.0
                ),

                "llm_judge_reason": llm_all.get(
                    "llm_judge_reason",
                    "judge_failed"
                )
            }

            scope = {

                "scope_safety": llm_all.get(
                    "scope_safety",
                    0.0
                ),

                "scope_completeness": llm_all.get(
                    "scope_completeness",
                    0.0
                ),

                "scope_originality": llm_all.get(
                    "scope_originality",
                    0.0
                ),

                "scope_precision": llm_all.get(
                    "scope_precision",
                    0.0
                ),

                "scope_efficiency": llm_all.get(
                    "scope_efficiency",
                    0.0
                ),

                "scope_weighted_total": llm_all.get(
                    "scope_weighted_total",
                    0.0
                )
            }

            faithfulness = futures["faithfulness"].result()

            context_rel = futures["context_rel"].result()
            answer_rel = futures["answer_rel"].result()

        # =====================================================
        # 🔹 RETRIEVAL
        # =====================================================
        retrieval = {

            "precision": futures["precision"].result(),

            "recall": futures["recall"].result(),

            "mrr": futures["mrr"].result(),

            "ndcg": futures["ndcg"].result(),

            "hit_rate": futures["hit_rate"].result()
        }

        # =====================================================
        # 🔹 MEDICAL RAG METRICS
        # =====================================================
        medical_metrics = {

            "faithfulness": faithfulness,

            "context_rel": context_rel,

            "answer_rel": answer_rel,

            "confidence": confidence,

            "grounding_score": evaluation.get(
                "grounding_score",
                0
            ),

            "retrieval_score": evaluation.get(
                "retrieval_score",
                0
            ),

            "hallucination_low": int(
                evaluation.get(
                    "hallucination_risk",
                    "medium"
                ) == "low"
            ),

            "hallucination_risk": evaluation.get(
                "hallucination_risk",
                "medium"
            ),

            "hallucinated": int(
                str(
                    evaluation.get(
                        "hallucination_risk",
                        "medium"
                    )
                ).lower() == "high"
            ),

            "safe_refusal": int(
                refusal_detected(pred)
            ),

            "grounded_answer": int(
                explanation.get(
                    "grounded",
                    False
                )
            ),

            "answer_quality": evaluation.get(
                "score",
                0
            ),

            "agent_iters": evaluation.get(
                "agent_iterations",
                1
            ),

            "rerank_score": evaluation.get(
                "reranker_confidence",
                0
            ),

            "query_type": query_type
        }

        # =====================================================
        # 🔹 FINAL RESULT
        # =====================================================
        res = {

            **bleu,

            **gleu,

            **meteor,

            **scope,

            **distinct,

            **accuracy_f1,

            **rouge,

            **sbert_similarity,

            **llm_judge,

            **retrieval,

            **medical_metrics,

            **latency_snapshot
        }

        res["retrieval_diagnostics"] = evaluation.get(
            "retrieval_diagnostics",
            []
        )

        res["question_id"] = idx
        res["question"] = q
        res["ground_truth_answer"] = gt
        res["generated_answer"] = pred
        res["hallucination_safe"] = res.get(
            "hallucination_low",
            0
        )
        res["context_relevancy"] = context_rel
        res["answer_relevancy"] = answer_rel
        res["retrieval_precision"] = retrieval.get(
            "precision",
            0
        )
        res["precision_at_5"] = retrieval.get(
            "precision",
            0
        )
        res["question_metadata"] = item.get(
            "query_metadata",
            {}
        )
        res["metrics"] = evaluation
        res["bertscore"] = question_bertscore

        results.append(res)

        predictions.append(pred)

        references.append(gt)

        # =====================================================
        # 🔹 STRUCTURED SUMMARY
        # =====================================================
        print(
            f"✅ Completed Q{idx}"
        )

        if direct_mode:

            print(
                "📊 Grounding : N/A"
            )

            print(
                "📊 Retrieval : N/A"
            )

        else:

            print(
                f"📊 Grounding : "
                f"{evaluation.get('grounding_score',0):.2f}"
            )

            print(
                f"📊 Retrieval : "
                f"{evaluation.get('retrieval_score',0):.2f}"
            )

        print(
            f"📊 Confidence: "
            f"{confidence:.2f}"
        )

        print(
            f"📊 Quality   : "
            f"{evaluation.get('score',0)}"
        )

    # =========================================================
    # 🔹 BERT SCORE
    # =========================================================
    bert = compute_bertscore(
        references,
        predictions
    )

    _save_experiment_results(
        results
    )

    return results, bert


# =========================================================
# 🔹 REPORT
# =========================================================
def print_report(results, bert):

    direct_mode = not settings.is_rag_enabled()

    print("\n" + "=" * 80)

    print(
        "ONCOLOGY RAG - COMPLETE EVALUATION REPORT"
    )

    print(
        "Agentic Oncology RAG"
    )

    print(
        f"Direct LLM mode   : {direct_mode}"
    )

    print(
        f"LAQA enabled        : {settings.is_laqa_enabled()}"
    )

    print(
        f"MRL enabled         : {settings.is_mrl_enabled()}"
    )

    print("=" * 80)

    # =====================================================
    # 🔹 BASIC
    # =====================================================
    print(
        f"\nQuestions evaluated : {len(results)}"
    )

    print(
        f"Avg confidence     : "
        f"{avg(results,'confidence'):.4f}"
    )

    print(
        f"Avg agent iters   : "
        f"{avg(results,'agent_iters'):.2f}"
    )

    print(
        f"Avg rerank score  : "
        f"{_report_metric(results, 'rerank_score', direct_mode)}"
    )

    # =====================================================
    # 🔹 RETRIEVAL
    # =====================================================
    print(
        "\n-- Retrieval Quality (k=5) "
        "---------------------------------------------"
    )

    if direct_mode:

        print(
            "Precision@5        : N/A"
        )

        print(
            "Recall@5           : N/A"
        )

        print(
            "MRR                : N/A"
        )

        print(
            "NDCG@5             : N/A"
        )

        print(
            "Hit-Rate@5         : N/A"
        )

    else:

        print(
            f"Precision@5        : "
            f"{avg(results,'precision'):.4f}"
        )

        print(
            f"Recall@5           : "
            f"{avg(results,'recall'):.4f}"
        )

        print(
            f"MRR                : "
            f"{avg(results,'mrr'):.4f}"
        )

        print(
            f"NDCG@5             : "
            f"{avg(results,'ndcg'):.4f}"
        )

        print(
            f"Hit-Rate@5         : "
            f"{avg(results,'hit_rate'):.4f}"
        )

    # =====================================================
    # 🔹 GENERATION
    # =====================================================
    print(
        "\n-- Generation Lexical "
        "-----------------------------------------------"
    )

    print(
        f"BLEU-1             : "
        f"{avg(results,'bleu1'):.4f}"
    )

    print(
        f"BLEU-2             : "
        f"{avg(results,'bleu2'):.4f}"
    )

    print(
        f"BLEU-4             : "
        f"{avg(results,'bleu4'):.4f}"
    )

    print(
        f"GLEU               : "
        f"{avg(results,'gleu'):.4f}"
    )

    print(
        f"METEOR             : "
        f"{avg(results,'meteor'):.4f}"
    )

    print(
        f"Accuracy           : "
        f"{avg(results,'accuracy'):.4f}"
    )

    print(
        f"F1                 : "
        f"{avg(results,'f1'):.4f}"
    )

    print(
        f"ROUGE-1            : "
        f"{avg(results,'rouge1'):.4f}"
    )

    print(
        f"ROUGE-2            : "
        f"{avg(results,'rouge2'):.4f}"
    )

    print(
        f"ROUGE-L            : "
        f"{avg(results,'rougeL'):.4f}"
    )

    print(
        f"ROUGE-Lsum         : "
        f"{avg(results,'rougeLsum'):.4f}"
    )

    # =====================================================
    # 🔹 SEMANTIC
    # =====================================================
    print(
        "\n-- Generation Semantic "
        "----------------------------------------------"
    )

    print(
        f"BERTScore F1       : "
        f"{bert:.4f}"
    )

    print(
        f"SBERT similarity   : "
        f"{avg(results,'sbert_similarity'):.4f}"
    )

    print(
        f"DISTINCT-1         : "
        f"{avg(results,'distinct1'):.4f}"
    )

    print(
        f"DISTINCT-2         : "
        f"{avg(results,'distinct2'):.4f}"
    )

    # =====================================================
    # 🔹 MEDICAL RAG
    # =====================================================
    print(
        "\n-- Faithfulness & Safety "
        "-----------------------------------------"
    )

    if direct_mode:

        print(
            "Faithfulness       : N/A"
        )

        print(
            "Context Relevancy  : N/A"
        )

        print(
            f"Answer relevancy   : "
            f"{avg(results,'answer_rel'):.4f}"
        )

        print(
            "Grounding Score    : N/A"
        )

        print(
            "Retrieval Score    : N/A"
        )

    else:

        print(
            f"Faithfulness       : "
            f"{avg(results,'faithfulness'):.4f}"
        )

        print(
            f"Context Relevancy  : "
            f"{avg(results,'context_rel'):.4f}"
        )

        print(
            f"Answer relevancy   : "
            f"{avg(results,'answer_rel'):.4f}"
        )

        print(
            f"Grounding Score    : "
            f"{avg(results,'grounding_score'):.4f}"
        )

        print(
            f"Retrieval Score    : "
            f"{avg(results,'retrieval_score'):.4f}"
        )

    print(
        f"Hallucination Safe : "
        f"{avg(results,'hallucination_low'):.4f}"
    )

    print(
        f"Grounded Answers   : "
        f"{avg(results,'grounded_answer'):.4f}"
    )

    print(
        f"Safe Refusal Rate  : "
        f"{avg(results,'safe_refusal'):.4f}"
    )

    print(
        f"Answer Quality     : "
        f"{avg(results,'answer_quality'):.4f}"
    )

    print(
        f"LLM Judge Score    : "
        f"{avg(results,'llm_judge_score'):.4f}"
    )

    # =====================================================
    # 🔹 SCOPE
    # =====================================================
    print(
        "\n-- S.C.O.P.E LLM-as-judge (/5.0) "
        "------------------------------------------------"
    )

    print(
        f"S  Safety         : "
        f"{avg(results,'scope_safety'):.2f}"
    )

    print(
        f"C  Completeness   : "
        f"{avg(results,'scope_completeness'):.2f}"
    )

    print(
        f"O  Originality    : "
        f"{avg(results,'scope_originality'):.2f}"
    )

    print(
        f"P  Precision      : "
        f"{avg(results,'scope_precision'):.2f}"
    )

    print(
        f"E  Efficiency     : "
        f"{avg(results,'scope_efficiency'):.2f}"
    )

    weighted = avg(
        results,
        "scope_weighted_total"
    )

    scores = [

        r.get(
            "scope_weighted_total",
            0
        )

        for r in results
    ]

    std = np.std(scores)

    print(
        f"Weighted Total    : "
        f"{weighted:.2f}/5.00 "
        f"(std={std:.2f})"
    )

    print("=" * 80)

    # =====================================================
    # 🔹 HALLUCINATION STATISTICS
    # =====================================================
    print(
        "\n=========================================================="
    )

    print(
        "Hallucination Statistics"
    )

    print(
        "=========================================================="
    )

    print(
        f"Hallucination Safe : {avg(results,'hallucination_low'):.4f}"
    )

    print(
        f"Hallucination Rate : {hallucination_rate(results):.2f}%"
    )

    # =====================================================
    # 🔹 LATENCY STATISTICS
    # =====================================================
    print(
        "\n=========================================================="
    )

    print(
        "Latency Statistics"
    )

    print(
        "=========================================================="
    )

    _print_latency_block(
        "Retrieval Latency",
        build_latency_stats(results, "retrieval_latency")
    )

    _print_latency_block(
        "Agent Communication Latency",
        build_latency_stats(results, "agent_communication_latency")
    )

    _print_latency_block(
        "Total Response Time",
        build_latency_stats(results, "total_response_time")
    )


# =========================================================
# 🔹 RUN
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Oncology Agentic RAG evaluation and latency statistics."
    )

    parser.add_argument(
        "--dataset",
        default="backend/cleaned_output.json",
        help="Path to the evaluation dataset JSON file."
    )

    args = parser.parse_args()

    results, bert = evaluate(
        args.dataset
    )

    print_report(
        results,
        bert
    )
