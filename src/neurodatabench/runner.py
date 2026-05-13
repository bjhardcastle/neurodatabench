"""Call-first runner for executing one NeuroDataBench implementation."""

from __future__ import annotations

import dataclasses
import csv
import importlib.metadata
import importlib.resources
import json
import logging
import platform
import socket
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

import numpy as np
import pydantic
import pydantic_settings
import psutil

import neurodatabench.benchmarks
import neurodatabench.models
import neurodatabench.plots

logger = logging.getLogger(__name__)
_MISSING_ANSWER = object()
_UNKNOWN_ANSWER = object()
_REQUIREMENTS_ARTIFACT_NAME = "requirements.txt"


class BenchmarkValidationError(RuntimeError):
    """Raised when a benchmark run completes with incorrect answers."""


class _RunConfig(pydantic_settings.BaseSettings):
    """Resolved runner configuration from call defaults, environment, and CLI.

    profile_interval_seconds is derived from profile_interval_ms and controls
    how often the background resource profiler samples during a run. Environment
    variables use the NDB_ prefix, for example NDB_BENCHMARK.
    """

    model_config = pydantic_settings.SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_",
    )

    benchmark: str | None = None
    out: Path | None = None
    profile_interval_ms: int = pydantic.Field(default=250, gt=0)
    log_level: str = "INFO"

    @pydantic.field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Normalize and validate a Python logging level name."""
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"Unsupported log level: {value}")
        return normalized

    @property
    def profile_interval_seconds(self) -> float:
        """Return the profiler sampling interval in seconds."""
        return self.profile_interval_ms / 1000


class _Profiler:
    """Background sampler for process and system resource usage."""

    def __init__(self, interval_seconds: float) -> None:
        """Create a profiler with a sampling interval in seconds."""
        self.interval_seconds = interval_seconds
        self.samples: list[neurodatabench.models.JsonObject] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self._net_start: Any = None
        self._net_end: Any = None
        self._disk_start: Any = None
        self._disk_end: Any = None
        self._baseline_process_rss_bytes: int | None = None
        self._baseline_process_plus_children_rss_bytes: int | None = None

    def start(self) -> None:
        """Start sampling in a background thread."""
        logger.debug("Starting profiler.")
        self._net_start = psutil.net_io_counters()
        self._disk_start = psutil.disk_io_counters()
        self._record_memory_baseline()
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and record final counters."""
        logger.debug("Stopping profiler.")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.samples.append(self._sample())
        self._net_end = psutil.net_io_counters()
        self._disk_end = psutil.disk_io_counters()

    def summary(self) -> neurodatabench.models.JsonObject:
        """Return a compact summary of collected samples."""
        process_rss_values = [
            int(sample["process"]["rss_bytes"])
            for sample in self.samples
            if isinstance(sample.get("process"), dict)
        ]
        total_rss_values = [
            int(sample["process"]["rss_bytes"]) + int(sample["children"]["rss_bytes"])
            for sample in self.samples
            if isinstance(sample.get("process"), dict)
            and isinstance(sample.get("children"), dict)
        ]
        peak_process_rss_bytes = max(process_rss_values, default=None)
        peak_total_rss_bytes = max(total_rss_values, default=None)
        return {
            "sample_interval_seconds": self.interval_seconds,
            "num_samples": len(self.samples),
            "baseline_process_rss_bytes": self._baseline_process_rss_bytes,
            "baseline_process_plus_children_rss_bytes": (
                self._baseline_process_plus_children_rss_bytes
            ),
            "peak_process_rss_bytes": peak_process_rss_bytes,
            "peak_process_plus_children_rss_bytes": peak_total_rss_bytes,
            "peak_process_rss_delta_bytes": _memory_delta(
                peak_process_rss_bytes,
                self._baseline_process_rss_bytes,
            ),
            "peak_process_plus_children_rss_delta_bytes": _memory_delta(
                peak_total_rss_bytes,
                self._baseline_process_plus_children_rss_bytes,
            ),
            "network_delta": _counter_delta(self._net_start, self._net_end),
            "disk_delta": _counter_delta(self._disk_start, self._disk_end),
            "network_scope": "system_delta",
            "disk_scope": "system_delta",
        }

    def _record_memory_baseline(self) -> None:
        """Record the process RSS baseline before measured setup starts."""
        process_rss = self._process.memory_info().rss
        child_rss = _child_rss_bytes(self._process)
        self._baseline_process_rss_bytes = process_rss
        self._baseline_process_plus_children_rss_bytes = process_rss + child_rss

    def _run(self) -> None:
        """Collect samples until stopped."""
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval_seconds)

    def _sample(self) -> neurodatabench.models.JsonObject:
        """Collect one profiler sample."""
        memory = self._process.memory_info()
        child_rss = 0
        child_cpu = 0.0
        for child in self._process.children(recursive=True):
            try:
                child_rss += child.memory_info().rss
                child_cpu += child.cpu_percent(interval=None)
            except psutil.Error:
                logger.debug("Skipping vanished child process during profiling.")
        virtual_memory = psutil.virtual_memory()
        return {
            "time_ns": time.time_ns(),
            "process": {
                "cpu_percent": self._process.cpu_percent(interval=None),
                "rss_bytes": memory.rss,
                "vms_bytes": memory.vms,
                "num_threads": self._process.num_threads(),
            },
            "children": {
                "cpu_percent": child_cpu,
                "rss_bytes": child_rss,
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_total_bytes": virtual_memory.total,
                "memory_available_bytes": virtual_memory.available,
                "memory_percent": virtual_memory.percent,
            },
            "io": {
                "net": psutil.net_io_counters()._asdict(),
                "disk": psutil.disk_io_counters()._asdict(),
            },
        }


def main(
    *,
    setup: Callable[[neurodatabench.models.RunContext], None],
    submit_answers: Callable[[neurodatabench.models.RunContext], None],
    implementation_id: str,
    implementation_local_cache: Literal["cold", "warm", False] | None = None,
    implementation_remote_cache: bool = False,
    benchmark: str | Path | None = None,
    out: str | Path | None = None,
    log_level: str | None = None,
    clear_cache: Callable[[neurodatabench.models.RunContext], None] | None = None,
    teardown: Callable[[neurodatabench.models.RunContext], None] | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    """Run a benchmark implementation and write result artifacts."""
    logger.debug("Resolving runner configuration.")
    implementation = neurodatabench.models.Implementation(
        id=implementation_id,
        local_cache=implementation_local_cache,
        remote_cache=implementation_remote_cache,
    )
    config = _resolve_config(
        default_benchmark=benchmark,
        default_out=out,
        default_log_level=log_level,
        implementation_id=implementation.id,
        argv=argv,
    )
    _configure_logging(config.log_level)
    benchmark_source = config.benchmark
    out_dir = config.out
    assert benchmark_source is not None
    assert out_dir is not None

    raw_benchmark, loaded_benchmark = _load_benchmark(benchmark_source)
    submitted_answers: list[dict[str, Any]] = []
    run_start_ns: int | None = None
    run_start_wall_time_ns = 0

    def submit_answer(
        question_id: str,
        answer: neurodatabench.models.JsonValue,
    ) -> None:
        """Capture one submitted answer with wall-clock time."""
        expected = _expected_answer(loaded_benchmark, question_id)
        reused_expected_object = isinstance(expected, (dict, list)) and answer is expected
        submitted_at = datetime.now(timezone.utc).isoformat()
        submitted_elapsed_seconds = (
            None
            if run_start_ns is None
            else (perf_counter_ns() - run_start_ns) / 1_000_000_000
        )
        logger.debug("Recording submitted answer for question %s.", question_id)
        if reused_expected_object:
            logger.warning(
                "Submitted answer for %s reuses the expected answer object.",
                question_id,
            )
        submitted_answers.append(
            {
                "question_id": question_id,
                "answer": answer,
                "submitted_at": submitted_at,
                "submitted_elapsed_seconds": submitted_elapsed_seconds,
                "reused_expected_object": reused_expected_object,
            }
        )

    context = neurodatabench.models.RunContext(
        benchmark=loaded_benchmark,
        submit_answer=submit_answer,
    )
    profiler = _Profiler(interval_seconds=config.profile_interval_seconds)
    setup_duration_ns = 0
    submit_answers_duration_ns = 0
    total_duration_ns = 0

    if clear_cache is not None:
        logger.debug("Running untimed clear_cache.")
        clear_cache(context)

    profiler.start()
    try:
        run_start_wall_time_ns = time.time_ns()
        total_start_ns = perf_counter_ns()
        run_start_ns = total_start_ns
        setup_start_ns = perf_counter_ns()
        setup(context)
        setup_duration_ns = perf_counter_ns() - setup_start_ns

        submit_answers_start_ns = perf_counter_ns()
        submit_answers(context)
        submit_answers_duration_ns = perf_counter_ns() - submit_answers_start_ns
        total_duration_ns = perf_counter_ns() - total_start_ns
    finally:
        profiler.stop()
        if teardown is not None:
            logger.debug("Running untimed teardown.")
            teardown(context)

    timings = neurodatabench.models.RunTimings(
        setup_duration_ns=setup_duration_ns,
        submit_answers_duration_ns=submit_answers_duration_ns,
        total_duration_ns=total_duration_ns,
        phase_timings=_run_phase_timings(
            setup_duration_ns=setup_duration_ns,
            submit_answers_duration_ns=submit_answers_duration_ns,
            total_duration_ns=total_duration_ns,
        ),
        answer_submissions=[
            neurodatabench.models.AnswerSubmissionTiming(
                question_id=str(submitted_answer["question_id"]),
                submitted_at=str(submitted_answer["submitted_at"]),
                submitted_elapsed_seconds=(
                    float(submitted_answer["submitted_elapsed_seconds"])
                    if submitted_answer["submitted_elapsed_seconds"] is not None
                    else None
                ),
            )
            for submitted_answer in submitted_answers
        ],
    )
    _validate_answers(loaded_benchmark, submitted_answers)
    metadata = _run_metadata(
        implementation=implementation,
        benchmark_source=benchmark_source,
        benchmark=loaded_benchmark,
    )
    _write_result_artifacts(
        out_dir=out_dir,
        raw_benchmark=raw_benchmark,
        metadata=metadata,
        timings=timings,
        validation={"correct": True},
        profile_samples=profiler.samples,
        profile_summary=profiler.summary(),
        run_start_wall_time_ns=run_start_wall_time_ns,
    )
    logger.info(
        "Benchmark run completed in %.3f s (setup %.3f s, submit_answers %.3f s).",
        total_duration_ns / 1_000_000_000,
        setup_duration_ns / 1_000_000_000,
        submit_answers_duration_ns / 1_000_000_000,
    )


def _resolve_config(
    *,
    default_benchmark: str | Path | None,
    default_out: str | Path | None,
    default_log_level: str | None,
    implementation_id: str,
    argv: Sequence[str] | None,
) -> _RunConfig:
    """Resolve call defaults with Pydantic Settings overrides."""
    settings_kwargs: dict[str, object] = {}
    if default_benchmark is not None:
        settings_kwargs["benchmark"] = str(default_benchmark)
    if default_out is not None:
        settings_kwargs["out"] = Path(default_out)
    if default_log_level is not None:
        settings_kwargs["log_level"] = default_log_level
    config = _RunConfig(
        **settings_kwargs,
        _cli_parse_args=tuple(argv) if argv is not None else None,
    )
    if config.benchmark is None:
        raise ValueError("benchmark must be provided to main() or --benchmark")
    if config.out is None:
        config.out = _default_output_dir(
            implementation_id=implementation_id,
            benchmark=config.benchmark,
        )
    return config


def _default_output_dir(*, implementation_id: str, benchmark: str) -> Path:
    """Return the default result directory for an implementation/benchmark run."""
    benchmark_name = _benchmark_name_for_path(benchmark)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / f"{implementation_id}_{benchmark_name}_{timestamp}"


def _benchmark_name_for_path(benchmark: str) -> str:
    """Return a filesystem-safe benchmark name for output path defaults."""
    path = Path(benchmark)
    if path.suffix == ".json":
        return path.stem
    return path.name


def _configure_logging(log_level: str) -> None:
    """Configure package logging for a benchmark run."""
    level = logging.getLevelNamesMapping()[log_level]
    logging.basicConfig(level=level)
    logging.getLogger("neurodatabench").setLevel(level)


def _load_benchmark(
    benchmark: str,
) -> tuple[neurodatabench.models.JsonObject, neurodatabench.models.Benchmark]:
    """Load a benchmark from a filesystem path or packaged benchmark name."""
    logger.debug("Loading benchmark %s.", benchmark)
    path = Path(benchmark)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        resource_name = path.name if path.suffix == ".json" else f"{benchmark}.json"
        resource = importlib.resources.files(neurodatabench.benchmarks).joinpath(
            resource_name
        )
        if not resource.is_file():
            raise FileNotFoundError(f"Benchmark not found: {benchmark}")
        raw = json.loads(resource.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Benchmark JSON must be an object.")
    loaded = neurodatabench.models.Benchmark.model_validate(raw)
    return raw, loaded


def _expected_answer(
    benchmark: neurodatabench.models.Benchmark,
    question_id: str,
) -> neurodatabench.models.JsonValue | None:
    """Return the expected answer for a question ID, if present."""
    for question in benchmark.questions:
        if question.id == question_id:
            return question.answer
    return None


def _validate_answers(
    benchmark: neurodatabench.models.Benchmark,
    submitted_answers: list[dict[str, Any]],
) -> None:
    """Raise as soon as any submitted answer is invalid."""
    by_question_id: dict[str, list[dict[str, Any]]] = {}
    for submitted_answer in submitted_answers:
        question_id = str(submitted_answer["question_id"])
        by_question_id.setdefault(question_id, []).append(submitted_answer)

    for question in benchmark.questions:
        matches = by_question_id.get(question.id, [])
        if not matches:
            raise BenchmarkValidationError(
                _validation_error_message(
                    question_id=question.id,
                    reason="missing_answer",
                    actual_answer=question.answer,
                    submitted_answer=_MISSING_ANSWER,
                )
            )
        if len(matches) > 1:
            raise BenchmarkValidationError(
                _validation_error_message(
                    question_id=question.id,
                    reason="duplicate_answer",
                    actual_answer=question.answer,
                    submitted_answer=[match["answer"] for match in matches],
                )
            )

        submitted_answer = matches[0]
        mismatch_reason = _compare_json_value(
            question.answer,
            submitted_answer["answer"],
        )
        if mismatch_reason is not None:
            raise BenchmarkValidationError(
                _validation_error_message(
                    question_id=question.id,
                    reason=mismatch_reason,
                    actual_answer=question.answer,
                    submitted_answer=submitted_answer["answer"],
                )
            )

    known_question_ids = {question.id for question in benchmark.questions}
    for submitted_answer in submitted_answers:
        question_id = str(submitted_answer["question_id"])
        if question_id not in known_question_ids:
            raise BenchmarkValidationError(
                _validation_error_message(
                    question_id=question_id,
                    reason="unknown_question_id",
                    actual_answer=_UNKNOWN_ANSWER,
                    submitted_answer=submitted_answer["answer"],
                )
            )


def _run_phase_timings(
    *,
    setup_duration_ns: int,
    submit_answers_duration_ns: int,
    total_duration_ns: int,
) -> list[neurodatabench.models.RunPhaseTiming]:
    """Return measured run phases with explicit start and stop times."""
    setup_stop_seconds = setup_duration_ns / 1_000_000_000
    submit_answers_duration_seconds = submit_answers_duration_ns / 1_000_000_000
    submit_answers_stop_seconds = setup_stop_seconds + submit_answers_duration_seconds
    total_stop_seconds = total_duration_ns / 1_000_000_000
    return [
        neurodatabench.models.RunPhaseTiming(
            phase="setup",
            start_seconds=0.0,
            stop_seconds=setup_stop_seconds,
            duration_seconds=setup_stop_seconds,
        ),
        neurodatabench.models.RunPhaseTiming(
            phase="submit_answers",
            start_seconds=setup_stop_seconds,
            stop_seconds=submit_answers_stop_seconds,
            duration_seconds=submit_answers_duration_seconds,
        ),
        neurodatabench.models.RunPhaseTiming(
            phase="total",
            start_seconds=0.0,
            stop_seconds=total_stop_seconds,
            duration_seconds=total_stop_seconds,
        ),
    ]


def _validation_error_message(
    *,
    question_id: str,
    reason: str,
    actual_answer: Any,
    submitted_answer: Any,
) -> str:
    """Return a validation error message with answer values."""
    return (
        f"Benchmark answers failed validation: {question_id}: {reason}; "
        f"submitted_answer={_format_answer_for_error(submitted_answer)}; "
        f"actual_answer={_format_answer_for_error(actual_answer)}"
    )


def _format_answer_for_error(answer: Any) -> str:
    """Return a compact representation of an answer for validation errors."""
    if answer is _MISSING_ANSWER:
        return "<missing>"
    if answer is _UNKNOWN_ANSWER:
        return "<unknown_question_id>"
    try:
        return json.dumps(_jsonable(answer), sort_keys=True)
    except TypeError:
        return repr(answer)


def _memory_delta(peak_bytes: int | None, baseline_bytes: int | None) -> int | None:
    """Return a non-negative memory delta from a raw peak and baseline."""
    if peak_bytes is None or baseline_bytes is None:
        return None
    return max(peak_bytes - baseline_bytes, 0)


def _compare_json_value(
    expected: neurodatabench.models.JsonValue,
    actual: neurodatabench.models.JsonValue,
) -> str | None:
    """Return a mismatch reason for JSON-compatible values, or None when equal."""
    if isinstance(expected, bool):
        return None if actual is expected else "exact_bool"
    if isinstance(expected, int) and not isinstance(expected, bool):
        if (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        ):
            return None
        return "exact_int"
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return "expected_float"
        return None if bool(np.isclose(float(actual), expected)) else "float_close"
    if isinstance(expected, str):
        return None if actual == expected else "exact_str"
    if expected is None:
        return None if actual is None else "exact_null"
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return "expected_list"
        if len(expected) != len(actual):
            return "list_length_mismatch"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            mismatch_reason = _compare_json_value(expected_item, actual_item)
            if mismatch_reason is not None:
                return f"list_item_{index}_{mismatch_reason}"
        return None
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return "expected_object"
        if set(expected) != set(actual):
            return "object_keys_mismatch"
        for key in expected:
            mismatch_reason = _compare_json_value(expected[key], actual[key])
            if mismatch_reason is not None:
                return f"object_value_{key}_{mismatch_reason}"
        return None
    return "unsupported_expected_type"


def _run_metadata(
    *,
    implementation: neurodatabench.models.Implementation,
    benchmark_source: str,
    benchmark: neurodatabench.models.Benchmark,
) -> neurodatabench.models.JsonObject:
    """Collect run metadata."""
    return {
        "datetime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "benchmark_harness_version": _package_version("neurodatabench"),
        "implementation": _jsonable(implementation),
        "benchmark": {
            "id": benchmark.id,
            "source": benchmark_source,
            "nwb_format": benchmark.nwb_format,
            "nwb_paths": benchmark.nwb_paths,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "process": {
            "pid": psutil.Process().pid,
            "cwd": str(Path.cwd()),
            "argv": sys.argv,
        },
        "requirements": _REQUIREMENTS_ARTIFACT_NAME,
    }


def _write_result_artifacts(
    *,
    out_dir: Path,
    raw_benchmark: neurodatabench.models.JsonObject,
    metadata: neurodatabench.models.JsonObject,
    timings: neurodatabench.models.RunTimings,
    validation: neurodatabench.models.JsonObject,
    profile_samples: list[neurodatabench.models.JsonObject],
    profile_summary: neurodatabench.models.JsonObject,
    run_start_wall_time_ns: int,
) -> None:
    """Write benchmark result artifacts."""
    logger.debug("Writing result artifacts to %s.", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "benchmark.json", raw_benchmark)
    _write_json(out_dir / "run_metadata.json", metadata)
    _write_json(out_dir / "timings.json", timings)
    _write_json(out_dir / "validation.json", validation)
    _write_json(out_dir / "profile_summary.json", profile_summary)
    _write_jsonl(out_dir / "profile_samples.jsonl", profile_samples)
    neurodatabench.plots._write_result_dashboard(
        out_dir=out_dir,
        metadata=metadata,
        timings=timings,
        profile_samples=profile_samples,
        profile_summary=profile_summary,
        run_start_wall_time_ns=run_start_wall_time_ns,
    )
    (out_dir / _REQUIREMENTS_ARTIFACT_NAME).write_text(
        _environment_requirements_text(),
        encoding="utf-8",
    )
    _write_bundle(out_dir)
    _update_results_leaderboard(out_dir)


def _update_results_leaderboard(out_dir: Path) -> None:
    """Refresh aggregate leaderboard artifacts for default-style results dirs."""
    results_dir = out_dir.parent
    if results_dir.name != "results":
        logger.debug("Skipping leaderboard update outside a results directory.")
        return

    rows = _leaderboard_rows(results_dir)
    if not rows:
        logger.debug("Skipping leaderboard update because no complete runs were found.")
        return

    _write_json(results_dir / "leaderboard.json", rows)
    _write_leaderboard_csv(results_dir / "leaderboard.csv", rows)
    neurodatabench.plots._write_leaderboard_plot(
        results_dir=results_dir,
        rows=rows,
    )


def _leaderboard_rows(results_dir: Path) -> list[neurodatabench.models.JsonObject]:
    """Return leaderboard rows discovered under a results directory."""
    rows = [
        row
        for run_dir in sorted(results_dir.iterdir())
        if run_dir.is_dir()
        for row in [_leaderboard_row(run_dir)]
        if row is not None
    ]
    rows.sort(
        key=lambda row: (
            str(row["benchmark_id"]),
            float(row["total_seconds"]),
            str(row["implementation_id"]),
            str(row["datetime_utc"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
        row["leaderboard_label"] = (
            f"{index}. {row['implementation_id']} "
            f"({row['benchmark_id']}, {row['datetime_utc']})"
        )
    return rows


def _leaderboard_row(run_dir: Path) -> neurodatabench.models.JsonObject | None:
    """Return one leaderboard row for a complete successful run directory."""
    metadata = _read_json_object(run_dir / "run_metadata.json")
    timings = _read_json_object(run_dir / "timings.json")
    validation = _read_json_object(run_dir / "validation.json")
    profile_summary = _read_json_object(run_dir / "profile_summary.json")
    if (
        metadata is None
        or timings is None
        or validation is None
        or profile_summary is None
        or validation.get("correct") is not True
    ):
        return None

    total_duration_ns = _json_number_at(timings, ("total_duration_ns",))
    setup_duration_ns = _json_number_at(timings, ("setup_duration_ns",))
    submit_answers_duration_ns = _json_number_at(
        timings,
        ("submit_answers_duration_ns",),
    )
    if (
        total_duration_ns is None
        or setup_duration_ns is None
        or submit_answers_duration_ns is None
    ):
        return None

    implementation = metadata.get("implementation")
    benchmark = metadata.get("benchmark")
    if not isinstance(implementation, dict) or not isinstance(benchmark, dict):
        return None

    peak_rss_delta_bytes = _json_number_at(
        profile_summary,
        ("peak_process_plus_children_rss_delta_bytes",),
    )
    peak_rss_bytes = _json_number_at(
        profile_summary,
        ("peak_process_plus_children_rss_bytes",),
    )
    comparison_rss_bytes = (
        peak_rss_bytes if peak_rss_delta_bytes is None else peak_rss_delta_bytes
    )
    network_received_bytes = _json_number_at(
        profile_summary,
        ("network_delta", "bytes_recv"),
    )
    return {
        "rank": 0,
        "result_dir": run_dir.name,
        "leaderboard_label": run_dir.name,
        "datetime_utc": str(metadata.get("datetime_utc", "")),
        "implementation_id": str(implementation.get("id", "unknown")),
        "benchmark_id": str(benchmark.get("id", "unknown")),
        "nwb_format": str(benchmark.get("nwb_format", "unknown")),
        "local_cache": str(implementation.get("local_cache", "")),
        "remote_cache": str(implementation.get("remote_cache", "")),
        "total_seconds": total_duration_ns / 1_000_000_000,
        "setup_seconds": setup_duration_ns / 1_000_000_000,
        "submit_answers_seconds": submit_answers_duration_ns / 1_000_000_000,
        "peak_rss_delta_mib": (
            None
            if comparison_rss_bytes is None
            else comparison_rss_bytes / 1_048_576
        ),
        "peak_rss_mib": (
            None if peak_rss_bytes is None else peak_rss_bytes / 1_048_576
        ),
        "network_received_mib": (
            None
            if network_received_bytes is None
            else network_received_bytes / 1_048_576
        ),
    }


def _write_leaderboard_csv(
    path: Path,
    rows: list[neurodatabench.models.JsonObject],
) -> None:
    """Write leaderboard rows as a CSV table."""
    fieldnames = [
        "rank",
        "implementation_id",
        "benchmark_id",
        "nwb_format",
        "local_cache",
        "remote_cache",
        "total_seconds",
        "setup_seconds",
        "submit_answers_seconds",
        "peak_rss_delta_mib",
        "peak_rss_mib",
        "network_received_mib",
        "datetime_utc",
        "result_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json_object(path: Path) -> neurodatabench.models.JsonObject | None:
    """Read a JSON object from disk, or return None if unavailable."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        logger.debug("Skipping unreadable JSON artifact at %s.", path)
        return None
    if not isinstance(value, dict):
        logger.debug("Skipping non-object JSON artifact at %s.", path)
        return None
    return value


def _json_number_at(value: object, keys: Sequence[str]) -> float | None:
    """Return a nested JSON number, if present."""
    current: object = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def _write_json(path: Path, value: object) -> None:
    """Write a JSON artifact."""
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[neurodatabench.models.JsonObject],
) -> None:
    """Write newline-delimited JSON rows."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _write_bundle(out_dir: Path) -> None:
    """Write a zip bundle containing result artifacts."""
    bundle_path = out_dir / "results_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out_dir.iterdir()):
            if path == bundle_path or not path.is_file():
                continue
            bundle.write(path, arcname=path.name)


def _jsonable(value: object) -> object:
    """Convert dataclasses and paths into JSON-compatible values."""
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _environment_requirements_text() -> str:
    """Return stable requirements-style pins from the current Python environment."""
    rows: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name is None:
            continue
        rows.add(f"{name}=={version}")
    return "\n".join(sorted(rows, key=str.lower)) + "\n"


def _package_version(package_name: str) -> str:
    """Return an installed package version or unknown."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _counter_delta(start: Any, end: Any) -> dict[str, int] | None:
    """Return deltas between two psutil counter snapshots."""
    if start is None or end is None:
        return None
    return {field: getattr(end, field) - getattr(start, field) for field in start._fields}


def _child_rss_bytes(process: psutil.Process) -> int:
    """Return total RSS bytes for currently live child processes."""
    child_rss = 0
    for child in process.children(recursive=True):
        try:
            child_rss += child.memory_info().rss
        except psutil.Error:
            logger.debug("Skipping vanished child process during baseline profiling.")
    return child_rss
