#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { path = ".." }
# ///

"""Run the default NeuroDataBench implementation matrix."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pydantic
import pydantic_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if REPO_SRC.exists():
    sys.path.insert(0, REPO_SRC.as_posix())

import neurodatabench.matrix

BENCHMARKS_BY_FORMAT = {
    "hdf5": "dynamic_routing_nwb_hdf5_v0",
    "zarr": "dynamic_routing_nwb_zarr_v0",
}
REMOTE_FILE_BACKENDS = ("remfile", "s3fs", "obstore")


class MatrixSettings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the benchmark matrix."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_MATRIX_",
    )

    dry_run: bool = False
    keep_going: bool = True
    only: list[str] = pydantic.Field(default_factory=list)
    skip: list[str] = pydantic.Field(default_factory=list)
    limit: int | None = None
    out: Path = Path("results")
    status_jsonl: Path | None = None
    profile_interval_ms: int | None = None
    timeout_seconds: float | None = None
    timeout_enabled: bool = pydantic.Field(default=True, alias="timeout")
    log_level: str = "INFO"

    @pydantic.field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize and validate a Python logging level name."""
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"Unsupported log level: {value}")
        return normalized

    @property
    def no_timeout(self) -> bool:
        """Return whether supervisor timeout enforcement is disabled."""
        return not self.timeout_enabled


def main() -> int:
    """Select and run entries from the default comparison matrix."""
    args = MatrixSettings()
    configure_logging(args.log_level)
    runs = neurodatabench.matrix.select_runs(
        default_matrix(),
        only=args.only,
        skip=args.skip,
    )
    if args.limit is not None:
        runs = runs[: args.limit]
    return neurodatabench.matrix.run_matrix(
        runs,
        repo_root=REPO_ROOT,
        output_root=args.out,
        status_path=args.status_jsonl,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        profile_interval_ms=args.profile_interval_ms,
        timeout_seconds=args.timeout_seconds,
        no_timeout=args.no_timeout,
        log_level=args.log_level,
    )


def configure_logging(log_level: str) -> None:
    """Configure logging for the matrix process."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def default_matrix() -> list[neurodatabench.matrix.MatrixRun]:
    """Build the repository's default benchmark matrix."""
    runs: list[neurodatabench.matrix.MatrixRun] = []
    for nwb_format, benchmark in BENCHMARKS_BY_FORMAT.items():
        for backend in ("obstore", "remfile", "s3fs"):
            runs.append(
                neurodatabench.matrix.MatrixRun(
                    label=f"lazynwb-0.2.91-{nwb_format}-{backend}",
                    implementation="implementations/lazynwb_template.py",
                    benchmark=benchmark,
                    implementation_id=f"lazynwb_0_2_91_{backend}_{nwb_format}",
                    object_store_backend=backend,
                    local_cache="cold",
                    dependencies=("lazynwb==0.2.91",),
                )
            )
        for local_cache in ("cold", "warm"):
            runs.append(
                neurodatabench.matrix.MatrixRun(
                    label=f"lazynwb-1.0.0dev3-{nwb_format}-obstore-{local_cache}",
                    implementation="implementations/lazynwb_template.py",
                    benchmark=benchmark,
                    implementation_id=f"lazynwb_1_0_0dev3_{nwb_format}_{local_cache}",
                    object_store_backend="obstore",
                    local_cache=local_cache,
                    dependencies=("lazynwb==1.0.0dev3",),
                )
            )
            runs.append(
                neurodatabench.matrix.MatrixRun(
                    label=f"lazynwb-1.0.0dev-source-{nwb_format}-obstore-{local_cache}",
                    implementation="implementations/lazynwb_template.py",
                    benchmark=benchmark,
                    implementation_id=f"lazynwb_1_0_0dev-source_{nwb_format}_{local_cache}",
                    object_store_backend="obstore",
                    local_cache=local_cache,
                    dependencies=("git+https://github.com/bjhardcastle/lazynwb@dev6",),
                )
            )

    for backend in REMOTE_FILE_BACKENDS:
        runs.append(
            neurodatabench.matrix.MatrixRun(
                label=f"direct-h5py-hdf5-{backend}",
                implementation="implementations/direct_h5py_template.py",
                benchmark=BENCHMARKS_BY_FORMAT["hdf5"],
                implementation_id=f"direct_h5py_{backend}",
                object_store_backend=backend,
            )
        )

    runs.append(
        neurodatabench.matrix.MatrixRun(
            label="parquet-components-hdf5-polars-s3-anon",
            implementation="implementations/parquet_components_template.py",
            benchmark=BENCHMARKS_BY_FORMAT["hdf5"],
            implementation_id="parquet_components",
            object_store_backend="polars_s3_anon",
        )
    )

    for zarr_major, zarr_requirement in (("2", "zarr<3"), ("3", "zarr>=3,<4")):
        for backend in ("s3fs", "obstore"):
            runs.append(
                neurodatabench.matrix.MatrixRun(
                    label=f"direct-zarr-v{zarr_major}-{backend}",
                    implementation="implementations/direct_zarr_template.py",
                    benchmark=BENCHMARKS_BY_FORMAT["zarr"],
                    implementation_id=f"direct_zarr_v{zarr_major}_{backend}",
                    object_store_backend=backend,
                    dependencies=(zarr_requirement,),
                )
            )

    runs.append(
        neurodatabench.matrix.MatrixRun(
            label="pynwb-zarr-v2-s3fs",
            implementation="implementations/pynwb_zarr_template.py",
            benchmark=BENCHMARKS_BY_FORMAT["zarr"],
            implementation_id="pynwb_hdmf_zarr_direct_s3fs",
            object_store_backend="s3fs",
            dependencies=("zarr<3",),
        )
    )

    for backend in REMOTE_FILE_BACKENDS:
        runs.append(
            neurodatabench.matrix.MatrixRun(
                label=f"pynwb-hdf5-{backend}",
                implementation="implementations/pynwb_hdf5_template.py",
                benchmark=BENCHMARKS_BY_FORMAT["hdf5"],
                implementation_id=f"pynwb_hdf5_nwbfile_{backend}",
                object_store_backend=backend,
            )
        )
    return runs


if __name__ == "__main__":
    sys.exit(main())
