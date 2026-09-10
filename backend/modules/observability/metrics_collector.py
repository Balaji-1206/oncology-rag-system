import time
import numpy as np
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List


class MetricsCollector:
    """Thread-safe telemetry metrics collector for production RAG pipeline."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.contradictions_detected = 0
        self.latencies: List[float] = []
        self.hallucination_counts: Dict[str, int] = defaultdict(int)
        self.evaluator_modes: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()

    def record_query(
        self,
        latency: float,
        success: bool = True,
        cache_hit: bool = False,
        contradiction: bool = False,
        hallucination_risk: str = "medium",
        evaluator_mode: str = "llm_evaluator"
    ) -> None:
        """Records telemetry for an executed clinical query."""
        with self.lock:
            self.total_queries += 1
            if success:
                self.successful_queries += 1
            else:
                self.failed_queries += 1

            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

            if contradiction:
                self.contradictions_detected += 1

            self.latencies.append(latency)
            if len(self.latencies) > 2000:
                self.latencies = self.latencies[-1000:]

            self.hallucination_counts[hallucination_risk] += 1
            self.evaluator_modes[evaluator_mode] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Calculates and returns metrics summary."""
        with self.lock:
            uptime = round(time.time() - self.start_time, 2)
            total = max(1, self.total_queries)

            if self.latencies:
                lat_array = np.array(self.latencies)
                p50 = round(float(np.percentile(lat_array, 50)), 3)
                p95 = round(float(np.percentile(lat_array, 95)), 3)
                p99 = round(float(np.percentile(lat_array, 99)), 3)
                mean_lat = round(float(np.mean(lat_array)), 3)
            else:
                p50 = p95 = p99 = mean_lat = 0.0

            cache_ratio = round(self.cache_hits / total, 3)
            error_rate = round(self.failed_queries / total, 3)

            return {
                "uptime_seconds": uptime,
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "failed_queries": self.failed_queries,
                "error_rate": error_rate,
                "cache_hit_ratio": cache_ratio,
                "contradictions_detected": self.contradictions_detected,
                "latency_metrics": {
                    "p50_seconds": p50,
                    "p95_seconds": p95,
                    "p99_seconds": p99,
                    "mean_seconds": mean_lat
                },
                "hallucination_risk_distribution": dict(self.hallucination_counts),
                "evaluator_mode_distribution": dict(self.evaluator_modes)
            }


_GLOBAL_METRICS = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Returns singleton MetricsCollector."""
    return _GLOBAL_METRICS
