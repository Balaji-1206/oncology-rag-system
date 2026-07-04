import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


DEFAULT_EXPERIMENT_A = os.path.join("backend", "results", "laqa_on_mrl_on.json")
DEFAULT_EXPERIMENT_B = os.path.join("backend", "results", "laqa_off_mrl_on.json")
DEFAULT_METRIC = "grounding_score"


def load_json_records(path: str) -> List[Dict[str, Any]]:
    """Load an experiment file and normalize it into a list of record dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in file: {path}") from exc

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("results", "records", "question_records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]

    return []


def coerce_float(value: Any) -> Optional[float]:
    """Best-effort float conversion that tolerates strings and booleans."""
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float, np.number)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            numeric = float(text)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) else None

    return None


def extract_question_id(record: Dict[str, Any]) -> Optional[int]:
    """Extract a comparable question id from a saved evaluation record."""
    for key in ("question_id", "qid", "id", "questionId"):
        if key in record:
            numeric = coerce_float(record.get(key))
            if numeric is not None:
                return int(numeric)
    return None


def _candidate_containers(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    containers = [record]
    for key in ("metrics", "evaluation", "scope", "latency", "performance", "medical_metrics"):
        nested = record.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    return containers


def extract_metric_value(record: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Extract a numeric metric from the record or its known nested containers."""
    aliases = {
        "grounding_score": ("grounding_score",),
        "faithfulness": ("faithfulness",),
        "context_rel": ("context_rel", "context_relevancy"),
        "answer_rel": ("answer_rel", "answer_relevancy"),
        "retrieval_score": ("retrieval_score",),
        "llm_judge_score": ("llm_judge_score",),
        "confidence": ("confidence",),
        "retrieval_latency": ("retrieval_latency",),
        "agent_communication_latency": ("agent_communication_latency",),
        "total_response_time": ("total_response_time",),
    }

    keys = aliases.get(metric_name, (metric_name,))

    for container in _candidate_containers(record):
        for key in keys:
            if key in container:
                numeric = coerce_float(container.get(key))
                if numeric is not None:
                    return numeric

    return None


def index_records_by_question_id(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for record in records:
        question_id = extract_question_id(record)
        if question_id is None:
            continue
        indexed[question_id] = record
    return indexed


def collect_paired_series(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    metric_name: str,
) -> Tuple[List[int], List[float], List[float], int]:
    """Align records by Question ID and extract comparable numeric series."""
    indexed_a = index_records_by_question_id(records_a)
    indexed_b = index_records_by_question_id(records_b)

    common_ids = sorted(set(indexed_a) & set(indexed_b))
    if not common_ids:
        raise ValueError(
            "No shared Question IDs were found between the two experiment files."
        )

    series_a: List[float] = []
    series_b: List[float] = []
    skipped_missing_metric = 0

    for question_id in common_ids:
        value_a = extract_metric_value(indexed_a[question_id], metric_name)
        value_b = extract_metric_value(indexed_b[question_id], metric_name)

        if value_a is None or value_b is None:
            skipped_missing_metric += 1
            continue

        series_a.append(value_a)
        series_b.append(value_b)

    if not series_a or not series_b:
        raise ValueError(
            f"No comparable values found for metric '{metric_name}'."
        )

    return common_ids, series_a, series_b, skipped_missing_metric


def paired_statistics(series_a: List[float], series_b: List[float]) -> Dict[str, float]:
    """Compute paired t-test statistics and summary values."""
    paired_length = min(len(series_a), len(series_b))
    if paired_length == 0:
        raise ValueError("No comparable paired observations were found.")

    arr_a = np.asarray(series_a[:paired_length], dtype=float)
    arr_b = np.asarray(series_b[:paired_length], dtype=float)
    diffs = arr_a - arr_b

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if paired_length > 1 else 0.0

    if np.allclose(diffs, 0.0):
        return {
            "n": paired_length,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "significant": False,
            "cohens_d": 0.0,
        }

    if paired_length > 1 and math.isclose(std_diff, 0.0, abs_tol=1e-12):
        t_statistic = float("inf") if mean_diff > 0 else float("-inf")
        return {
            "n": paired_length,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "t_statistic": t_statistic,
            "p_value": 0.0,
            "ci_low": mean_diff,
            "ci_high": mean_diff,
            "significant": True,
            "cohens_d": float("inf") if mean_diff != 0 else 0.0,
        }

    t_statistic, p_value = stats.ttest_rel(arr_a, arr_b, nan_policy="omit")
    if not math.isfinite(float(t_statistic)):
        t_statistic = 0.0
    if not math.isfinite(float(p_value)):
        p_value = 1.0

    sem = stats.sem(diffs, nan_policy="omit")
    if sem is not None and math.isfinite(float(sem)) and paired_length > 1:
        critical = float(stats.t.ppf(0.975, df=paired_length - 1))
        margin = critical * float(sem)
        ci_low = mean_diff - margin
        ci_high = mean_diff + margin
    else:
        ci_low = mean_diff
        ci_high = mean_diff

    cohens_d = 0.0 if std_diff == 0 else mean_diff / std_diff

    return {
        "n": paired_length,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool(p_value < 0.05),
        "cohens_d": float(cohens_d),
    }


def run_paired_t_test(path_a: str, path_b: str, metric_name: str) -> Dict[str, Any]:
    records_a = load_json_records(path_a)
    records_b = load_json_records(path_b)

    if not records_a:
        raise ValueError(f"No records found in experiment A file: {path_a}")
    if not records_b:
        raise ValueError(f"No records found in experiment B file: {path_b}")

    common_ids, series_a, series_b, skipped_missing_metric = collect_paired_series(
        records_a,
        records_b,
        metric_name,
    )

    stats_result = paired_statistics(series_a, series_b)
    stats_result.update(
        {
            "questions_compared": len(series_a),
            "shared_question_ids": common_ids,
            "skipped_missing_metric": skipped_missing_metric,
            "metric": metric_name,
            "experiment_a": path_a,
            "experiment_b": path_b,
        }
    )
    return stats_result


def _format_metric_name(metric_name: str) -> str:
    pretty_map = {
        "grounding_score": "Grounding Score",
        "faithfulness": "Faithfulness",
        "context_rel": "Context Relevancy",
        "answer_rel": "Answer Relevancy",
        "retrieval_score": "Retrieval Score",
        "llm_judge_score": "LLM Judge Score",
        "confidence": "Confidence",
        "retrieval_latency": "Retrieval Latency",
        "agent_communication_latency": "Agent Communication Latency",
        "total_response_time": "Total Response Time",
    }

    return pretty_map.get(
        metric_name,
        metric_name.replace("_", " ").title()
    )


def _experiment_title(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0].lower()

    title_map = {
        "laqa_on_mrl_on": "LAQA + MRL",
        "laqa_off_mrl_on": "Without LAQA",
        "laqa_on_mrl_off": "LAQA without MRL",
        "laqa_off_mrl_off": "Without LAQA / MRL Off",
    }

    return title_map.get(
        stem,
        os.path.basename(path)
    )


def _effect_size_label(value: float) -> str:
    magnitude = abs(value)

    if magnitude < 0.20:
        return "Small"

    if magnitude < 0.80:
        return "Medium"

    return "Large"


def _print_kv(label: str, value: str, width: int = 40) -> None:
    print(f"{label:<{width}} : {value}")


def print_report(result: Dict[str, Any]) -> None:
    print("==================================================")
    print("PAIRED T-TEST REPORT")
    print("==================================================")
    print("")
    _print_kv("Experiment A", _experiment_title(result["experiment_a"]))
    _print_kv("File", result["experiment_a"])
    print("")
    _print_kv("Experiment B", _experiment_title(result["experiment_b"]))
    _print_kv("File", result["experiment_b"])
    print("")
    print("--------------------------------------------------")
    print("Metric Compared")
    print("--------------------------------------------------")
    print("")
    print(_format_metric_name(result["metric"]))
    print("")
    _print_kv("Questions Compared", f"{result['questions_compared']}")
    print("")
    print("--------------------------------------------------")
    print("Descriptive Statistics")
    print("--------------------------------------------------")
    print("")
    _print_kv(
        f"Mean {_format_metric_name(result['metric'])} (Experiment A)",
        f"{result['mean_a']:.4f}"
    )
    _print_kv(
        f"Mean {_format_metric_name(result['metric'])} (Experiment B)",
        f"{result['mean_b']:.4f}"
    )
    _print_kv("Mean Difference (A - B)", f"{result['mean_diff']:.4f}")
    _print_kv("Standard Deviation of Differences", f"{result['std_diff']:.4f}")
    print("")
    print("--------------------------------------------------")
    print("Paired t-test")
    print("--------------------------------------------------")
    print("")
    _print_kv("t-statistic", f"{result['t_statistic']:.4f}")
    _print_kv("p-value", f"{result['p_value']:.4f}")
    _print_kv(
        "Statistically Significant",
        "YES (p < 0.05)" if result["significant"] else "NO"
    )
    print("")
    print("--------------------------------------------------")
    print("95% Confidence Interval")
    print("--------------------------------------------------")
    print("")
    print("Confidence Interval (A - B)")
    print("")
    _print_kv("Lower Bound", f"{result['ci_low']:.4f}")
    _print_kv("Upper Bound", f"{result['ci_high']:.4f}")
    print("")
    print("Interpretation")
    print("")
    print(
        "We are 95% confident that the true mean difference\n"
        "between Experiment A and Experiment B lies within\n"
        "this interval."
    )
    print("")
    print("--------------------------------------------------")
    print("Effect Size")
    print("--------------------------------------------------")
    print("")
    if math.isfinite(result["cohens_d"]):
        _print_kv("Cohen's d", f"{result['cohens_d']:.4f}")
        print("")
        print("Interpretation")
        print("")
        print("Small Effect     : |d| < 0.20")
        print("Medium Effect    : |d| ≈ 0.50")
        print("Large Effect     : |d| ≥ 0.80")
        print("")
        observed = _effect_size_label(result["cohens_d"])
        _print_kv("Observed Effect", observed)
    else:
        _print_kv("Cohen's d", "INF")
        print("")
        print("Interpretation")
        print("")
        print("Small Effect     : |d| < 0.20")
        print("Medium Effect    : |d| ≈ 0.50")
        print("Large Effect     : |d| ≥ 0.80")
        print("")
        _print_kv("Observed Effect", "Large")
    print("")
    print("==================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired t-test comparison for Oncology Agentic RAG experiments."
    )
    parser.add_argument(
        "--a",
        default=DEFAULT_EXPERIMENT_A,
        help="Path to experiment A JSON file."
    )
    parser.add_argument(
        "--b",
        default=DEFAULT_EXPERIMENT_B,
        help="Path to experiment B JSON file."
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Metric name to compare."
    )

    args = parser.parse_args()

    try:
        result = run_paired_t_test(args.a, args.b, args.metric)
        print_report(result)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
