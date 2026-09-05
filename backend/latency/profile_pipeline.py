"""
Standalone profiling/debug utility for full pipeline latency analysis.

This file is intentionally profiling-only:
- no retrieval logic changes
- no evaluation logic changes
- no scoring changes
- no architecture changes

Usage pattern:
1) Create a PipelineProfiler().
2) Wrap existing stage calls with profiler.profile_stage("Stage Name").
3) Run questions through profile_pipeline(...) or profile_stage_map(...).
4) Review per-question report and final summary.

Optional hooks are provided for:
- repeated embedding/Ollama call tracking
- cache hit/miss tracking
- model load timing
- GPU/CPU snapshots if available
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore


PROFILE_STAGES: Tuple[str, ...] = (
    "LAQA",
    "Query Expansion",
    "Dense Embedding",
    "FAISS Search",
    "BM25 Retrieval",
    "Hybrid Fusion",
    "Reranker",
    "Context Building",
    "MedGemma Generation",
    "Evaluator LLM",
    "BLEU/ROUGE/METEOR",
    "SBERT/BERTScore",
    "Grounding Metric",
    "Faithfulness Metric",
    "Relevance Metric",
    "S.C.O.P.E Evaluation",
    "XAI Layer",
)

DISPLAY_NAMES: Dict[str, str] = {
    "LAQA": "LAQA",
    "Query Expansion": "Query Expansion",
    "Dense Embedding": "Dense Embedding",
    "FAISS Search": "FAISS Search",
    "BM25 Retrieval": "BM25 Retrieval",
    "Hybrid Fusion": "Hybrid Fusion",
    "Reranker": "Reranker",
    "Context Building": "Context Building",
    "MedGemma Generation": "MedGemma Generation",
    "Evaluator LLM": "Evaluator LLM",
    "BLEU/ROUGE/METEOR": "BLEU/ROUGE/METEOR",
    "SBERT/BERTScore": "SBERT/BERTScore",
    "Grounding Metric": "Grounding Metric",
    "Faithfulness Metric": "Faithfulness Metric",
    "Relevance Metric": "Relevance Metric",
    "S.C.O.P.E Evaluation": "S.C.O.P.E Evaluation",
    "XAI Layer": "XAI Layer",
}

RETRIEVAL_STAGES: Tuple[str, ...] = (
    "Query Expansion",
    "Dense Embedding",
    "FAISS Search",
    "BM25 Retrieval",
    "Hybrid Fusion",
    "Reranker",
    "Context Building",
)

EMBEDDING_STAGES: Tuple[str, ...] = (
    "Dense Embedding",
    "SBERT/BERTScore",
)

OLLAMA_STAGES: Tuple[str, ...] = (
    "MedGemma Generation",
    "Evaluator LLM",
)

LEXICAL_METRIC_STAGES: Tuple[str, ...] = (
    "BLEU/ROUGE/METEOR",
)

SEMANTIC_METRIC_STAGES: Tuple[str, ...] = (
    "SBERT/BERTScore",
)

EVALUATION_STAGES: Tuple[str, ...] = (
    "Evaluator LLM",
    "Grounding Metric",
    "Faithfulness Metric",
    "Relevance Metric",
    "S.C.O.P.E Evaluation",
)

XAI_STAGES: Tuple[str, ...] = (
    "XAI Layer",
)


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _fmt_seconds(value: float) -> str:
    if value is None or not math.isfinite(value):
        value = 0.0
    return f"{value:.2f}s"


def _short_hash(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, (str, bytes)):
        text = repr(text)
    if isinstance(text, str):
        text = text.encode("utf-8", errors="ignore")
    return hashlib.sha256(text).hexdigest()[:16]


def _sum_times(record: "QuestionRecord", stage_names: Iterable[str]) -> float:
    return sum(record.stage_times.get(stage, 0.0) for stage in stage_names)


def _label_width() -> int:
    return max(len(v) for v in DISPLAY_NAMES.values()) + 2


def _safe_div(n: float, d: float) -> float:
    return (n / d) if d else 0.0


@dataclass
class StageStat:
    name: str
    total_time: float = 0.0
    calls: int = 0
    min_time: float = field(default_factory=lambda: float("inf"))
    max_time: float = 0.0

    def add(self, elapsed: float) -> None:
        self.total_time += elapsed
        self.calls += 1
        if elapsed < self.min_time:
            self.min_time = elapsed
        if elapsed > self.max_time:
            self.max_time = elapsed

    @property
    def avg_time(self) -> float:
        return self.total_time / self.calls if self.calls else 0.0


@dataclass
class QuestionRecord:
    question_id: int
    question_text: str = ""
    start_perf: float = 0.0
    end_perf: float = 0.0
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    stage_calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bottleneck_stage: str = ""
    bottleneck_share: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    embedding_calls: int = 0
    ollama_calls: int = 0
    evaluator_calls: int = 0
    repeated_embedding_calls: int = 0
    repeated_ollama_calls: int = 0
    model_load_time: float = 0.0
    model_load_events: int = 0
    system_start: Dict[str, Any] = field(default_factory=dict)
    system_end: Dict[str, Any] = field(default_factory=dict)
    embedding_signatures: set = field(default_factory=set, repr=False)
    ollama_signatures: set = field(default_factory=set, repr=False)
    cache_names: set = field(default_factory=set, repr=False)
    exceptions: List[str] = field(default_factory=list)

    def finalize(self) -> None:
        self.total_time = max(0.0, self.end_perf - self.start_perf)
        if self.total_time > 0.0 and self.stage_times:
            stage_name, stage_time = max(self.stage_times.items(), key=lambda item: item[1])
            self.bottleneck_stage = stage_name
            self.bottleneck_share = _safe_div(stage_time, self.total_time) * 100.0
        else:
            self.bottleneck_stage = ""
            self.bottleneck_share = 0.0


class PipelineProfiler:
    """
    Lightweight runtime profiler for per-question and per-stage analysis.

    The profiler is intentionally generic:
    - Use profile_stage(...) around any code block.
    - Use profile_callable(...) to measure a function.
    - Use note_embedding_call(...) / note_ollama_call(...) for repeated-call tracking.
    - Use record_cache_hit/miss(...) for cache efficiency.
    - Use model_load(...) for model initialization timing.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        capture_system_stats: bool = True,
        stream: Any = None,
    ) -> None:
        self.enabled = enabled
        self.capture_system_stats = capture_system_stats
        self.stream = stream if stream is not None else sys.stdout

        self.stage_stats: Dict[str, StageStat] = {name: StageStat(name=name) for name in PROFILE_STAGES}
        self.question_records: List[QuestionRecord] = []

        self._current_question: Optional[QuestionRecord] = None
        self._current_stage_name: Optional[str] = None

        self._cache_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})
        self._global_cache_hits: int = 0
        self._global_cache_misses: int = 0

        self._global_embedding_calls: int = 0
        self._global_ollama_calls: int = 0
        self._global_evaluator_calls: int = 0
        self._global_repeated_embedding_calls: int = 0
        self._global_repeated_ollama_calls: int = 0

        self._model_load_times: Dict[str, float] = defaultdict(float)
        self._model_load_events: Dict[str, int] = defaultdict(int)

    @contextlib.contextmanager
    def question_scope(self, question_id: int, question_text: str = ""):
        record = QuestionRecord(question_id=question_id, question_text=question_text)
        record.start_perf = time.perf_counter()
        if self.capture_system_stats:
            record.system_start = self._snapshot_system()
        self._current_question = record
        try:
            yield record
        except Exception as exc:
            record.exceptions.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record.end_perf = time.perf_counter()
            if self.capture_system_stats:
                record.system_end = self._snapshot_system()
            record.finalize()
            self.question_records.append(record)
            self._current_question = None
            self._current_stage_name = None

    @contextlib.contextmanager
    def profile_stage(self, stage_name: str):
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        previous_stage = self._current_stage_name
        self._current_stage_name = stage_name
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._add_stage_time(stage_name, elapsed)
            self._current_stage_name = previous_stage

    def timed(self, stage_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.profile_stage(stage_name):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    def profile_callable(self, stage_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self.profile_stage(stage_name):
            return fn(*args, **kwargs)

    def record_stage_duration(self, stage_name: str, elapsed: float) -> None:
        """Explicitly records elapsed time (in seconds) for an individually timed stage."""
        self._add_stage_time(stage_name, max(0.0, float(elapsed)))

    @contextlib.contextmanager
    def model_load(self, model_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record_model_load(model_name, elapsed)

    def record_model_load(self, model_name: str, elapsed: float) -> None:
        self._model_load_times[model_name] += elapsed
        self._model_load_events[model_name] += 1
        if self._current_question is not None:
            self._current_question.model_load_time += elapsed
            self._current_question.model_load_events += 1

    def record_cache_hit(self, cache_name: str = "default") -> None:
        self._cache_totals[cache_name]["hits"] += 1
        self._global_cache_hits += 1
        if self._current_question is not None:
            self._current_question.cache_hits += 1
            self._current_question.cache_names.add(cache_name)

    def record_cache_miss(self, cache_name: str = "default") -> None:
        self._cache_totals[cache_name]["misses"] += 1
        self._global_cache_misses += 1
        if self._current_question is not None:
            self._current_question.cache_misses += 1
            self._current_question.cache_names.add(cache_name)

    def note_embedding_call(self, payload: Any = None) -> bool:
        """
        Records an embedding call and returns True if the payload appears repeated.
        """
        self._global_embedding_calls += 1
        repeated = False
        signature = _short_hash(payload)
        if self._current_question is not None:
            self._current_question.embedding_calls += 1
            if signature and signature in self._current_question.embedding_signatures:
                self._current_question.repeated_embedding_calls += 1
                self._global_repeated_embedding_calls += 1
                repeated = True
            if signature:
                self._current_question.embedding_signatures.add(signature)
        return repeated

    def note_ollama_call(self, payload: Any = None) -> bool:
        """
        Records an Ollama call and returns True if the payload appears repeated.
        """
        self._global_ollama_calls += 1
        repeated = False
        signature = _short_hash(payload)
        if self._current_question is not None:
            self._current_question.ollama_calls += 1
            if self._current_stage_name == "Evaluator LLM":
                self._current_question.evaluator_calls += 1
                self._global_evaluator_calls += 1
            if signature and signature in self._current_question.ollama_signatures:
                self._current_question.repeated_ollama_calls += 1
                self._global_repeated_ollama_calls += 1
                repeated = True
            if signature:
                self._current_question.ollama_signatures.add(signature)
        return repeated

    def note_evaluator_call(self, payload: Any = None) -> bool:
        """
        Explicit evaluator call marker for non-Ollama evaluator implementations.
        """
        self._global_evaluator_calls += 1
        if self._current_question is not None:
            self._current_question.evaluator_calls += 1
        return self.note_ollama_call(payload)

    def _add_stage_time(self, stage_name: str, elapsed: float) -> None:
        if stage_name not in self.stage_stats:
            self.stage_stats[stage_name] = StageStat(name=stage_name)
        self.stage_stats[stage_name].add(elapsed)

        if self._current_question is not None:
            self._current_question.stage_times[stage_name] += elapsed
            self._current_question.stage_calls[stage_name] += 1

    def _snapshot_system(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "perf_counter": time.perf_counter(),
            "process_cpu_time": time.process_time(),
        }

        if psutil is not None:
            try:
                proc = psutil.Process(os.getpid())
                mem = proc.memory_info()
                snapshot["rss_mb"] = mem.rss / (1024 * 1024)
                snapshot["vms_mb"] = getattr(mem, "vms", 0) / (1024 * 1024)
                snapshot["threads"] = proc.num_threads()
            except Exception:
                pass

        if torch is not None:
            try:
                snapshot["torch_cuda_available"] = bool(torch.cuda.is_available())
                if torch.cuda.is_available():
                    snapshot["cuda_device_count"] = int(torch.cuda.device_count())
                    snapshot["cuda_current_device"] = int(torch.cuda.current_device())
                    snapshot["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                    snapshot["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            except Exception:
                pass

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_util = []
                gpu_mem_util = []
                gpu_mem_used_mb = []
                for idx in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_util.append(int(util.gpu))
                    gpu_mem_util.append(int(util.memory))
                    gpu_mem_used_mb.append(mem.used / (1024 * 1024))
                snapshot["gpu_util_percent"] = gpu_util
                snapshot["gpu_mem_util_percent"] = gpu_mem_util
                snapshot["gpu_mem_used_mb"] = gpu_mem_used_mb
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            except Exception:
                pass

        return snapshot

    def _stage_label(self, stage_name: str) -> str:
        return DISPLAY_NAMES.get(stage_name, stage_name)

    def _stage_time(self, record: QuestionRecord, stage_name: str) -> float:
        return record.stage_times.get(stage_name, 0.0)

    def print_question_report(self, record: QuestionRecord) -> None:
        width = _label_width()
        print("", file=self.stream)
        print("=" * 48, file=self.stream)
        print(f"QUESTION {record.question_id}", file=self.stream)
        print("=" * 10, file=self.stream)
        print("", file=self.stream)

        for stage_name in PROFILE_STAGES:
            label = self._stage_label(stage_name)
            value = self._stage_time(record, stage_name)
            print(f"{label:<{width}}: {_fmt_seconds(value)}", file=self.stream)

        print("", file=self.stream)
        print(f"{'TOTAL':<{width}}: {_fmt_seconds(record.total_time)}", file=self.stream)
        print("", file=self.stream)

        if record.bottleneck_stage:
            print("BOTTLENECK:", file=self.stream)
            print(
                f"{record.bottleneck_stage} ({record.bottleneck_share:.1f}%)",
                file=self.stream,
            )
        else:
            print("BOTTLENECK:", file=self.stream)
            print("None", file=self.stream)

        if record.exceptions:
            print("", file=self.stream)
            print("EXCEPTIONS:", file=self.stream)
            for item in record.exceptions:
                print(item, file=self.stream)

    def print_summary(self) -> None:
        print("", file=self.stream)
        print("=" * 48, file=self.stream)
        print("FINAL SUMMARY", file=self.stream)
        print("=" * 48, file=self.stream)
        print("", file=self.stream)

        if not self.question_records:
            print("No questions profiled.", file=self.stream)
            return

        totals = [r.total_time for r in self.question_records]
        print(f"Questions profiled     : {len(self.question_records)}", file=self.stream)
        print(f"Average total time     : {_fmt_seconds(_safe_mean(totals))}", file=self.stream)
        print(f"Total runtime          : {_fmt_seconds(sum(totals))}", file=self.stream)

        slowest_stage = max(self.stage_stats.values(), key=lambda s: s.total_time, default=None)
        if slowest_stage is not None:
            print(
                f"Slowest stage          : {slowest_stage.name} ({_fmt_seconds(slowest_stage.avg_time)} avg)",
                file=self.stream,
            )

        total_ollama = sum(
            _sum_times(r, OLLAMA_STAGES) for r in self.question_records
        )
        total_embedding = sum(
            _sum_times(r, EMBEDDING_STAGES) for r in self.question_records
        )
        total_retrieval = sum(
            _sum_times(r, RETRIEVAL_STAGES) for r in self.question_records
        )
        total_evaluator = sum(
            _sum_times(r, ("Evaluator LLM",)) for r in self.question_records
        )
        total_semantic_metrics = sum(
            _sum_times(r, SEMANTIC_METRIC_STAGES) for r in self.question_records
        )
        total_lexical_metrics = sum(
            _sum_times(r, LEXICAL_METRIC_STAGES) for r in self.question_records
        )
        total_metric_time = sum(
            _sum_times(r, ("BLEU/ROUGE/METEOR", "SBERT/BERTScore", "Grounding Metric", "Faithfulness Metric", "Relevance Metric", "S.C.O.P.E Evaluation"))
            for r in self.question_records
        )

        print(f"Total Ollama time      : {_fmt_seconds(total_ollama)}", file=self.stream)
        print(f"Total embedding time   : {_fmt_seconds(total_embedding)}", file=self.stream)
        print(f"Total retrieval time   : {_fmt_seconds(total_retrieval)}", file=self.stream)
        print(f"Total evaluator time   : {_fmt_seconds(total_evaluator)}", file=self.stream)
        print(f"Semantic metric time   : {_fmt_seconds(total_semantic_metrics)}", file=self.stream)
        print(f"Lexical metric time    : {_fmt_seconds(total_lexical_metrics)}", file=self.stream)
        print(f"Total metric time      : {_fmt_seconds(total_metric_time)}", file=self.stream)

        cache_hits = self._global_cache_hits
        cache_misses = self._global_cache_misses
        cache_total = cache_hits + cache_misses
        cache_efficiency = _safe_div(cache_hits, cache_total) * 100.0
        print(f"Cache efficiency       : {cache_efficiency:.1f}% ({cache_hits} hits / {cache_misses} misses)", file=self.stream)

        print(f"Embedding calls        : {self._global_embedding_calls}", file=self.stream)
        print(f"Repeated embedding calls: {self._global_repeated_embedding_calls}", file=self.stream)
        print(f"Ollama calls           : {self._global_ollama_calls}", file=self.stream)
        print(f"Repeated Ollama calls  : {self._global_repeated_ollama_calls}", file=self.stream)
        print(f"Evaluator calls        : {self._global_evaluator_calls}", file=self.stream)

        if self._model_load_times:
            print("", file=self.stream)
            print("MODEL LOAD TIMES", file=self.stream)
            for model_name, elapsed in sorted(self._model_load_times.items(), key=lambda item: item[1], reverse=True):
                events = self._model_load_events.get(model_name, 0)
                print(f"{model_name:<24}: {_fmt_seconds(elapsed)} ({events} events)", file=self.stream)

        print("", file=self.stream)
        print("AVERAGE STAGE TIMES", file=self.stream)
        stage_avgs = sorted(self.stage_stats.values(), key=lambda s: s.avg_time, reverse=True)
        for stat in stage_avgs:
            if stat.calls == 0:
                continue
            print(f"{stat.name:<24}: {_fmt_seconds(stat.avg_time)}", file=self.stream)

        print("", file=self.stream)
        print("TOP BOTTLENECKS", file=self.stream)
        bottlenecks = sorted(self.stage_stats.values(), key=lambda s: s.total_time, reverse=True)
        for idx, stat in enumerate(bottlenecks[:5], start=1):
            if stat.calls == 0:
                continue
            share = _safe_div(stat.total_time, sum(totals)) * 100.0
            print(f"{idx}. {stat.name:<22} {_fmt_seconds(stat.total_time)} total ({share:.1f}%)", file=self.stream)

        print("", file=self.stream)
        print("ESTIMATED OPTIMIZATION PRIORITIES", file=self.stream)
        priorities = [s.name for s in bottlenecks if s.calls > 0][:3]
        if priorities:
            for idx, name in enumerate(priorities, start=1):
                print(f"{idx}. {name}", file=self.stream)
        else:
            print("None", file=self.stream)

        print("", file=self.stream)
        if self.question_records:
            print("CPU/GPU SNAPSHOTS", file=self.stream)
            latest = self.question_records[-1]
            if latest.system_start or latest.system_end:
                if "rss_mb" in latest.system_end:
                    print(f"Process RSS (end)      : {latest.system_end['rss_mb']:.2f} MB", file=self.stream)
                if "process_cpu_time" in latest.system_end and "process_cpu_time" in latest.system_start:
                    cpu_delta = latest.system_end["process_cpu_time"] - latest.system_start["process_cpu_time"]
                    print(f"Process CPU time       : {_fmt_seconds(cpu_delta)}", file=self.stream)
                if "cuda_allocated_mb" in latest.system_end:
                    print(
                        f"CUDA allocated (end)   : {latest.system_end['cuda_allocated_mb']:.2f} MB",
                        file=self.stream,
                    )
                if "cuda_reserved_mb" in latest.system_end:
                    print(
                        f"CUDA reserved (end)    : {latest.system_end['cuda_reserved_mb']:.2f} MB",
                        file=self.stream,
                    )
                if "gpu_util_percent" in latest.system_end:
                    print(f"GPU util (%)           : {latest.system_end['gpu_util_percent']}", file=self.stream)

        if self._cache_totals:
            print("", file=self.stream)
            print("CACHE BREAKDOWN", file=self.stream)
            for cache_name, counts in sorted(self._cache_totals.items(), key=lambda item: (item[1]["hits"] + item[1]["misses"]), reverse=True):
                hits = counts["hits"]
                misses = counts["misses"]
                total = hits + misses
                eff = _safe_div(hits, total) * 100.0
                print(f"{cache_name:<24}: {eff:.1f}% ({hits} hits / {misses} misses)", file=self.stream)

    def get_stage_totals(self) -> Dict[str, float]:
        return {name: stat.total_time for name, stat in self.stage_stats.items()}

    def get_question_records(self) -> List[QuestionRecord]:
        return list(self.question_records)


def profile_stage_map(
    question: Any,
    stage_callables: Mapping[str, Callable[[MutableMapping[str, Any]], Any]],
    *,
    question_id: int = 1,
    profiler: Optional[PipelineProfiler] = None,
    print_report: bool = True,
) -> Tuple[Dict[str, Any], PipelineProfiler]:
    """
    Convenience helper for sequential stage profiling.

    stage_callables values should be callables that accept a mutable context dict.
    The context carries:
      - question
      - question_id
      - outputs
      - previous stage outputs
    """
    profiler = profiler or PipelineProfiler()
    outputs: Dict[str, Any] = {}
    context: Dict[str, Any] = {
        "question": question,
        "question_id": question_id,
        "outputs": outputs,
    }

    with profiler.question_scope(question_id=question_id, question_text=str(question)):
        for stage_name in PROFILE_STAGES:
            fn = stage_callables.get(stage_name)
            if not callable(fn):
                continue
            with profiler.profile_stage(stage_name):
                result = fn(context)
            outputs[stage_name] = result
            context[stage_name] = result

    if print_report:
        profiler.print_question_report(profiler.question_records[-1])

    return outputs, profiler


def profile_pipeline(
    questions: Sequence[Any],
    question_runner: Callable[[Any, PipelineProfiler, int], Any],
    *,
    profiler: Optional[PipelineProfiler] = None,
    print_each_question: bool = True,
    print_summary: bool = True,
) -> Tuple[List[QuestionRecord], PipelineProfiler]:
    """
    Generic pipeline profiler.

    question_runner(question, profiler, question_id) should execute the existing
    pipeline and wrap each relevant stage with profiler.profile_stage(...).

    This wrapper does not change pipeline behavior; it only measures timings.
    """
    profiler = profiler or PipelineProfiler()

    for idx, question in enumerate(questions, start=1):
        with profiler.question_scope(idx, str(question)):
            question_runner(question, profiler, idx)
        if print_each_question:
            profiler.print_question_report(profiler.question_records[-1])

    if print_summary:
        profiler.print_summary()

    return profiler.get_question_records(), profiler


def build_stage_context(
    question: Any,
    *,
    question_id: int = 1,
    previous_outputs: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Minimal helper for building a mutable context dict without altering any logic.
    """
    ctx: Dict[str, Any] = {
        "question": question,
        "question_id": question_id,
        "outputs": previous_outputs if previous_outputs is not None else {},
    }
    return ctx


__all__ = [
    "PIPELINE_STAGES",
    "PROFILE_STAGES",
    "DISPLAY_NAMES",
    "PipelineProfiler",
    "QuestionRecord",
    "StageStat",
    "profile_stage_map",
    "profile_pipeline",
    "build_stage_context",
]