#!/usr/bin/env python3
"""Run the NeuroDataBench example helper matrix across storage combinations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARKS_BY_FORMAT = {
    "hdf5": "dynamic_routing_hdf5_v0",
    "zarr": "dynamic_routing_zarr_v0",
}
_REMOTE_BACKENDS = ("remfile", "s3fs", "ros", "obstore")


@dataclass(frozen=True)
class MatrixRun:
    """One benchmark helper invocation in the comparison matrix."""

    label: str
    helper: str
    benchmark: str
    implementation_id: str
    object_store_backend: str
    nwb_format: Literal["hdf5", "zarr"]
    local_cache: Literal["cold", "warm"] | None = None
    zarr_major_version: Literal["2", "3"] | None = None
    extra_uv_args: tuple[str, ...] = ()


def main() -> int:
    """Run matrix entries selected by command-line filters."""
    args = _parse_args()
    _configure_logging(args.log_level)
    output_root = _output_root(args)
    runs = _selected_runs(_matrix(), only=args.only, skip=args.skip)
    if args.limit is not None:
        runs = runs[: args.limit]
    if not runs:
        logger.info("No matrix runs selected.")
        return 0

    status_path = _status_path(args, output_root=output_root)
    if status_path is not None and not args.dry_run:
        status_path.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    logger.info("Starting %d matrix run(s).", len(runs))
    for index, run in enumerate(runs, start=1):
        logger.info("Starting %d/%d: %s", index, len(runs), run.label)
        command = _command_for(run, args, output_root=output_root)
        started = time.monotonic()
        if args.dry_run:
            logger.info("DRY RUN: %s", " ".join(command))
            result = {"returncode": 0, "elapsed_seconds": 0.0}
        else:
            completed = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                env=_environment_for(run, output_root=output_root),
                check=False,
            )
            result = {
                "returncode": completed.returncode,
                "elapsed_seconds": time.monotonic() - started,
            }
            _write_status(status_path, run, command, result)
        if result["returncode"] != 0:
            failures += 1
            logger.error("Failed %s with exit code %s.", run.label, result["returncode"])
            if not args.keep_going:
                return int(result["returncode"])
        else:
            logger.info("Finished %s in %.3f seconds.", run.label, result["elapsed_seconds"])

    if failures:
        logger.error("Matrix finished with %d failed run(s).", failures)
        return 1
    logger.info("Matrix finished successfully.")
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the matrix runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after failed runs.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Substring filter for run labels. Can be repeated.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Substring filter for run labels to skip. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N selected entries.")
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Root directory for per-run result directories and matrix-managed "
            "artifacts such as status JSONL and lazynwb caches."
        ),
    )
    parser.add_argument(
        "--status-jsonl",
        type=Path,
        help="Path for per-run status records. Defaults to <out>/matrix_status.jsonl.",
    )
    parser.add_argument("--profile-interval-ms", type=int, help="Forwarded runner profile interval.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Supervisor timeout override for each matrix subprocess.",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Disable supervisor timeout enforcement.",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level for this matrix script and helpers.")
    return parser.parse_args()


def _output_root(args: argparse.Namespace) -> Path:
    """Return the matrix storage root."""
    if args.out is not None:
        return args.out
    return Path("results")


def _status_path(args: argparse.Namespace, *, output_root: Path) -> Path | None:
    """Return the JSON Lines status path for matrix runs."""
    if args.status_jsonl is not None:
        return args.status_jsonl
    return output_root / "matrix_status.jsonl"


def _configure_logging(log_level: str) -> None:
    """Configure matrix runner logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _matrix() -> list[MatrixRun]:
    """Build the default benchmark helper matrix."""
    runs: list[MatrixRun] = []
    for nwb_format, benchmark in _BENCHMARKS_BY_FORMAT.items():
        for backend in ("obstore", "remfile", "s3fs"):
            runs.append(
                MatrixRun(
                    label=f"lazynwb-pre1-{nwb_format}-{backend}",
                    helper="examples/lazynwb_v0.py",
                    benchmark=benchmark,
                    implementation_id=f"lazynwb_pre1_{backend}_{nwb_format}",
                    object_store_backend=backend,
                    nwb_format=nwb_format,  # type: ignore[arg-type]
                    local_cache="cold",
                )
            )
        for local_cache in ("cold", "warm"):
            runs.append(
                MatrixRun(
                    label=f"lazynwb-1.0.0dev5-{nwb_format}-obstore-{local_cache}",
                    helper="examples/lazynwb_v1dev.py",
                    benchmark=benchmark,
                    implementation_id=f"lazynwb_1dev5_{nwb_format}_{local_cache}",
                    object_store_backend="obstore",
                    nwb_format=nwb_format,  # type: ignore[arg-type]
                    local_cache=local_cache,
                )
            )

    for backend in _REMOTE_BACKENDS:
        runs.append(
            MatrixRun(
                label=f"direct-h5py-hdf5-{backend}",
                helper="examples/direct_h5py_template.py",
                benchmark=_BENCHMARKS_BY_FORMAT["hdf5"],
                implementation_id=f"direct_h5py_{backend}",
                object_store_backend=backend,
                nwb_format="hdf5",
            )
        )

    for zarr_major in ("2", "3"):
        zarr_pin = "zarr<3" if zarr_major == "2" else "zarr>=3,<4"
        for backend in ("s3fs", "obstore"):
            runs.append(
                MatrixRun(
                    label=f"direct-zarr-v{zarr_major}-{backend}",
                    helper="examples/direct_zarr_template.py",
                    benchmark=_BENCHMARKS_BY_FORMAT["zarr"],
                    implementation_id=f"direct_zarr_v{zarr_major}_{backend}",
                    object_store_backend=backend,
                    nwb_format="zarr",
                    local_cache="cold",
                    zarr_major_version=zarr_major,  # type: ignore[arg-type]
                    extra_uv_args=("--with", zarr_pin),
                )
            )

    for backend in ("s3fs", "obstore"):
        runs.append(
            MatrixRun(
                label=f"pynwb-zarr-v2-{backend}",
                helper="examples/pynwb_zarr_template.py",
                benchmark=_BENCHMARKS_BY_FORMAT["zarr"],
                implementation_id=f"pynwb_hdmf_zarr_direct_{backend}",
                object_store_backend=backend,
                nwb_format="zarr",
                local_cache="cold",
                zarr_major_version="2",
                extra_uv_args=("--with", "zarr<3"),
            )
        )

    for backend in _REMOTE_BACKENDS:
        runs.append(
            MatrixRun(
                label=f"pynwb-hdf5-{backend}",
                helper="examples/pynwb_hdf5_template.py",
                benchmark=_BENCHMARKS_BY_FORMAT["hdf5"],
                implementation_id=f"pynwb_hdf5_nwbfile_{backend}",
                object_store_backend=backend,
                nwb_format="hdf5",
            )
        )
    return runs


def _selected_runs(runs: list[MatrixRun], *, only: list[str], skip: list[str]) -> list[MatrixRun]:
    """Filter matrix runs by label substrings."""
    selected = [
        run
        for run in runs
        if (not only or any(token in run.label for token in only))
        and not any(token in run.label for token in skip)
    ]
    return selected


def _command_for(
    run: MatrixRun,
    args: argparse.Namespace,
    *,
    output_root: Path,
) -> list[str]:
    """Build the supervised uv command for a matrix run."""
    child_command = [
        "uv",
        "run",
        *run.extra_uv_args,
        run.helper,
        "--log-level",
        args.log_level,
    ]
    if args.profile_interval_ms is not None:
        child_command.extend(["--profile-interval-ms", str(args.profile_interval_ms)])
    if args.out is not None:
        child_command.extend(["--out", str(_run_output_dir(run, output_root))])
    command = [
        sys.executable,
        "-m",
        "neurodatabench.runner",
        "supervise",
        "--benchmark",
        run.benchmark,
    ]
    if args.timeout_seconds is not None:
        command.extend(["--timeout-seconds", str(args.timeout_seconds)])
    if args.no_timeout:
        command.append("--no-timeout")
    command.append("--")
    command.extend(child_command)
    return command


def _run_output_dir(run: MatrixRun, output_root: Path) -> Path:
    """Return the concrete output directory for one matrix run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"{run.implementation_id}_{run.benchmark}_{timestamp}"


def _environment_for(run: MatrixRun, *, output_root: Path) -> dict[str, str]:
    """Build the process environment for a matrix helper."""
    env = os.environ.copy()
    env["AWS_REGION"] = env.get("AWS_REGION", "us-west-2")
    env["NDB_BENCHMARK"] = run.benchmark
    env["NDB_IMPLEMENTATION_ID"] = run.implementation_id
    env["NDB_OBJECT_STORE_BACKEND"] = run.object_store_backend
    if run.local_cache is not None:
        env["NDB_LOCAL_CACHE"] = run.local_cache
    if run.helper in {"examples/lazynwb_v0.py", "examples/lazynwb_v1dev.py"}:
        cache_name = f"{run.implementation_id.removesuffix('_cold').removesuffix('_warm')}.sqlite"
        env["NDB_LAZYNWB_CACHE_PATH"] = str(output_root / "matrix_caches" / cache_name)
    if run.zarr_major_version is not None:
        env["NDB_ZARR_MAJOR_VERSION"] = run.zarr_major_version
    if run.helper == "examples/lazynwb_v0.py":
        if run.object_store_backend == "s3fs":
            env["LAZYNWB_USE_OBSTORE"] = "false"
            env["LAZYNWB_USE_REMFILE"] = "false"
        elif run.object_store_backend == "remfile":
            env["LAZYNWB_USE_OBSTORE"] = "false"
            env["LAZYNWB_USE_REMFILE"] = "true"
        elif run.object_store_backend == "obstore":
            env["LAZYNWB_USE_OBSTORE"] = "true"
            env["LAZYNWB_USE_REMFILE"] = "false"
    return env


def _write_status(
    status_path: Path | None,
    run: MatrixRun,
    command: list[str],
    result: dict[str, float | int],
) -> None:
    """Append one JSON Lines status record."""
    if status_path is None:
        return
    record = {
        "run": asdict(run),
        "command": command,
        **result,
    }
    with status_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
