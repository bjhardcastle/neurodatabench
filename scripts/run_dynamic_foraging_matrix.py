#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///

"""Run the LazyNWB and Escapewheel dynamic-foraging implementations."""

from __future__ import annotations

import logging
from pathlib import Path

import pydantic
import pydantic_settings

import neurodatabench.matrix


def find_repo_root(script_path: Path) -> Path:
    """Find the checkout containing the benchmark implementation scripts."""
    script_dir = script_path.resolve().parent
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "implementations").is_dir() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    raise RuntimeError(f"Could not find the neurodatabench checkout from {script_path}")


REPO_ROOT = find_repo_root(Path(__file__))
BENCHMARK = "dynamic_foraging_nwb_v0"
BENCHMARK_SOURCE = "src/neurodatabench/benchmarks/dynamic_foraging_nwb_v0.json"


class MatrixSettings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the dynamic-foraging matrix."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        cli_shortcuts={
            "dry-run": "dry_run",
            "keep-going": "keep_going",
            "log-level": "log_level",
            "profile-interval-ms": "profile_interval_ms",
            "status-jsonl": "status_jsonl",
            "timeout-disabled": "timeout_disabled",
            "timeout-seconds": "timeout_seconds",
        },
        env_prefix="NDB_MATRIX_",
    )

    dry_run: bool = False
    keep_going: bool = True
    only: list[str] = pydantic.Field(default_factory=list)
    skip: list[str] = pydantic.Field(default_factory=list)
    out: Path = Path("results")
    status_jsonl: Path | None = None
    profile_interval_ms: int | None = None
    timeout_seconds: float | None = None
    timeout_disabled: bool = False
    log_level: str = "INFO"

    @pydantic.field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize and validate a Python logging level name."""
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"Unsupported log level: {value}")
        return normalized


def dynamic_foraging_matrix() -> list[neurodatabench.matrix.MatrixRun]:
    """Return the LazyNWB storage variants and Escapewheel implementation."""
    return [
        neurodatabench.matrix.MatrixRun(
            implementation="implementations/lazynwb/dynamic_foraging.py",
            benchmark=BENCHMARK,
            implementation_id="lazynwb-1.0.0dev3-obstore-hdf5-cold",
            benchmark_source=BENCHMARK_SOURCE,
            object_store_backend="obstore",
            local_cache="cold",
            editable_dependencies=(".",),
            environment={"NDB_SOURCE_TYPE": "hdf5"},
        ),
        neurodatabench.matrix.MatrixRun(
            implementation="implementations/lazynwb/dynamic_foraging.py",
            benchmark=BENCHMARK,
            implementation_id="lazynwb-1.0.0dev3-obstore-zarr-cold",
            benchmark_source=BENCHMARK_SOURCE,
            object_store_backend="obstore",
            local_cache="cold",
            editable_dependencies=(".",),
            environment={"NDB_SOURCE_TYPE": "zarr"},
        ),
        neurodatabench.matrix.MatrixRun(
            implementation="implementations/escapewheel/dynamic_foraging.py",
            benchmark=BENCHMARK,
            implementation_id="escapewheel-reader",
            benchmark_source=BENCHMARK_SOURCE,
            editable_dependencies=(".",),
        ),
    ]


def configure_logging(log_level: str) -> None:
    """Configure logging for the matrix process."""
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> int:
    """Select and run the dynamic-foraging implementation matrix."""
    args = MatrixSettings()
    configure_logging(args.log_level)
    runs = neurodatabench.matrix.select_runs(
        dynamic_foraging_matrix(),
        only=args.only,
        skip=args.skip,
    )
    return neurodatabench.matrix.run_matrix(
        runs,
        repo_root=REPO_ROOT,
        output_root=args.out,
        status_path=args.status_jsonl,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        profile_interval_ms=args.profile_interval_ms,
        timeout_seconds=args.timeout_seconds,
        no_timeout=args.timeout_disabled,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    raise SystemExit(main())
