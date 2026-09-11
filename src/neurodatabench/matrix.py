"""Reusable orchestration for running benchmark implementation matrices."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import neurodatabench.models

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class MatrixRun:
    """One implementation and environment combination in a benchmark matrix."""

    label: str
    implementation: str
    benchmark: str
    implementation_id: str
    object_store_backend: str | None = None
    local_cache: neurodatabench.models.LocalCacheState | None = None
    dependencies: tuple[str, ...] = ()
    environment: dict[str, str] = dataclasses.field(default_factory=dict)


def select_runs(
    runs: Iterable[MatrixRun],
    *,
    only: Sequence[str] = (),
    skip: Sequence[str] = (),
) -> list[MatrixRun]:
    """Filter matrix runs by label substrings while preserving order."""
    return [
        run
        for run in runs
        if (not only or any(token in run.label for token in only))
        and not any(token in run.label for token in skip)
    ]


def run_matrix(
    runs: Sequence[MatrixRun],
    *,
    repo_root: Path,
    output_root: Path = Path("results"),
    status_path: Path | None = None,
    dry_run: bool = False,
    keep_going: bool = True,
    profile_interval_ms: int | None = None,
    timeout_seconds: float | None = None,
    no_timeout: bool = False,
    log_level: str = "INFO",
) -> int:
    """Run a matrix sequentially and return a process-style status code."""
    if not runs:
        logger.info("No matrix runs selected.")
        return 0
    repo_root = repo_root.resolve()
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    if status_path is not None and not status_path.is_absolute():
        status_path = repo_root / status_path
    resolved_status_path = (
        output_root / "matrix_status.jsonl" if status_path is None else status_path
    )
    if not dry_run:
        resolved_status_path.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    logger.info("Starting %d matrix run(s).", len(runs))
    for index, run in enumerate(runs, start=1):
        logger.info("Starting %d/%d: %s", index, len(runs), run.label)
        run_output_dir = _run_output_dir(run, output_root)
        command = _command_for(
            run,
            run_output_dir=run_output_dir,
            profile_interval_ms=profile_interval_ms,
            timeout_seconds=timeout_seconds,
            no_timeout=no_timeout,
            log_level=log_level,
        )
        started = time.monotonic()
        if dry_run:
            logger.info("DRY RUN: %s", " ".join(command))
            result: dict[str, float | int] = {
                "returncode": 0,
                "elapsed_seconds": 0.0,
            }
        else:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=_environment_for(run, output_root=output_root),
                check=False,
            )
            result = {
                "returncode": completed.returncode,
                "elapsed_seconds": time.monotonic() - started,
            }
            _write_status(resolved_status_path, run, command, result)

        if result["returncode"] != 0:
            failures += 1
            logger.error("Failed %s with exit code %s.", run.label, result["returncode"])
            if not keep_going:
                return int(result["returncode"])
        else:
            logger.info("Finished %s in %.3f seconds.", run.label, result["elapsed_seconds"])

    if failures:
        logger.error("Matrix finished with %d failed run(s).", failures)
        return 1
    logger.info("Matrix finished successfully.")
    return 0


def _command_for(
    run: MatrixRun,
    *,
    run_output_dir: Path,
    profile_interval_ms: int | None,
    timeout_seconds: float | None,
    no_timeout: bool,
    log_level: str,
) -> list[str]:
    """Build the supervised command for one matrix run."""
    child_command = ["uv", "run"]
    for dependency in run.dependencies:
        child_command.extend(("--with", dependency))
    child_command.extend((run.implementation, "--log-level", log_level))
    if profile_interval_ms is not None:
        child_command.extend(("--profile-interval-ms", str(profile_interval_ms)))
    child_command.extend(("--out", str(run_output_dir)))

    command = [
        sys.executable,
        "-m",
        "neurodatabench.runner",
        "supervise",
        "--benchmark",
        run.benchmark,
    ]
    if timeout_seconds is not None:
        command.extend(("--timeout-seconds", str(timeout_seconds)))
    if no_timeout:
        command.append("--no-timeout")
    command.extend(("--timeout-profile-out", str(run_output_dir), "--"))
    command.extend(child_command)
    return command


def _run_output_dir(run: MatrixRun, output_root: Path) -> Path:
    """Return a timestamped output directory for one matrix run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return output_root / f"{run.implementation_id}_{run.benchmark}_{timestamp}"


def _environment_for(run: MatrixRun, *, output_root: Path) -> dict[str, str]:
    """Build a clean implementation environment for one matrix run."""
    environment = os.environ.copy()
    for name in (
        "NDB_BENCHMARK",
        "NDB_IMPLEMENTATION_ID",
        "NDB_OBJECT_STORE_BACKEND",
        "NDB_LOCAL_CACHE",
        "NDB_LAZYNWB_CACHE_PATH",
        "LAZYNWB_USE_OBSTORE",
        "LAZYNWB_USE_REMFILE",
    ):
        environment.pop(name, None)
    environment.update(run.environment)
    environment["AWS_REGION"] = environment.get("AWS_REGION", "us-west-2")
    environment["NDB_BENCHMARK"] = run.benchmark
    environment["NDB_IMPLEMENTATION_ID"] = run.implementation_id
    if run.object_store_backend is not None:
        environment["NDB_OBJECT_STORE_BACKEND"] = run.object_store_backend
    if run.local_cache is not None:
        environment["NDB_LOCAL_CACHE"] = run.local_cache
    if Path(run.implementation).name == "lazynwb_template.py":
        cache_id = run.implementation_id.removesuffix("_cold").removesuffix("_warm")
        environment["NDB_LAZYNWB_CACHE_PATH"] = str(
            output_root / "matrix_caches" / f"{cache_id}.sqlite"
        )
    return environment


def _write_status(
    status_path: Path,
    run: MatrixRun,
    command: list[str],
    result: dict[str, float | int],
) -> None:
    """Append one JSON Lines status record."""
    record = {"run": dataclasses.asdict(run), "command": command, **result}
    with status_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")
