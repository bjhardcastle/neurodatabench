# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "lazynwb>=1.0.0",
#   "hdmf-zarr<0.12",
#   "numpy",
#   "polars",
#   "psutil",
#   "s3fs",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


"""lazynwb answers for the change detection benchmark questions."""

from __future__ import annotations

import importlib.metadata
import os
import numcodecs.pickles
import tempfile
from pathlib import Path
from typing import Any, Literal

import lazynwb
import pydantic_settings
import pandas as pd

import neurodatabench

logger = neurodatabench.get_logger(__name__)
# Zarr 2 settting for decoding arrays with object-dtype stored with a pickle codec
numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True 

_SOURCE_TYPE = "zarr"

class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the lazynwb runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_ignore_unknown_args=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_",
    )

    benchmark: Literal["baseline_nwb_zarr_v0"] = "baseline_nwb_zarr_v0"
    object_store_backend: Literal["obstore", "remfile", "s3fs"] = "obstore"
    implementation_id: str = "lazynwb"
    local_cache: Literal["cold", "warm"] = "cold"
    lazynwb_cache_path: Path | None = None
    aws_region: str = "us-west-2"


settings: Settings | None = None


def _settings() -> Settings:
    """Return the resolved implementation settings."""
    if settings is None:
        raise RuntimeError("Settings must be initialized before running the implementation.")
    return settings

state: dict[str, Any] = {}

#region Baseline solutions
def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.key)

        match question.id:
            case "load_trials_table":
                _load_table(context.benchmark.data_sources[_SOURCE_TYPE][1], '/intervals/trials')
            case "load_units_table":
                _load_table(context.benchmark.data_sources[_SOURCE_TYPE][0], '/units')
            case "load_multiple_trials_tables":
                n_tables = get_param(question, "n_tables")
                _load_multiple_tables(context.benchmark.data_sources[_SOURCE_TYPE], n_tables, 'intervals/trials')
            case "load_multiple_units_tables":
                n_tables = get_param(question, "n_tables")
                _load_multiple_tables(context.benchmark.data_sources[_SOURCE_TYPE], n_tables, 'units')
            case "load_units_column":
                col_name = get_param(question, "col_name")
                _load_units_column(context.benchmark.data_sources[_SOURCE_TYPE][0], col_name)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.key}")

        context.submit_answer(question.key, 0)


def get_param(question: Any, param_name: str) -> Any:
    """Return a question param by name, warning when falling back to its default."""
    match param_name:
        case "n_tables":
            default: Any = 10
        case "col_name":
            default = None
        case _:
            raise ValueError(f"Unsupported param name: {param_name}")

    if question.params and param_name in question.params:
        value = question.params[param_name]
        return int(value) if param_name == "n_tables" else str(value)
    logger.warning(
        "Question %s has no '%s' param; defaulting to %r.", question.id, param_name, default
    )
    return default


def _load_table(file: Any, table_name: str) -> None:
    df = lazynwb.get_df(
        file,
        table_name,
        exclude_array_columns=False
    )
    assert type(df) is pd.DataFrame


def _load_multiple_tables(nwb_paths: list[str], n_tables: int, table_name: str) -> None:
    """Load multiple tables using lazynwb."""
    n = len(nwb_paths)
    if n < n_tables:
        raise ValueError(f"{n_tables} NWB paths were requested, but only {n} were provided.")

    lazynwb.get_df(
        nwb_paths[0:n_tables],
        table_name,
        exclude_array_columns=False
    )


def _load_units_column(nwb_path: str, col_name: str) -> None:
    lazynwb.get_df(
        nwb_path,
        "units",
        include_column_names=[col_name]
    )
#endregion

#region Runtime Configuration
def _backend() -> str:
    """Return the requested lazynwb object-store backend label."""
    return _settings().object_store_backend


def _configure_backend(backend: str) -> None:
    """Configure backend switches exposed by the installed lazynwb version."""
    version = _lazynwb_version()
    logger.debug("Configuring lazynwb %s backend %s.", version, backend)
    if version.split(".", maxsplit=1)[0] == "0":
        if backend not in {"obstore", "remfile", "s3fs"}:
            raise ValueError(f"Unsupported lazynwb pre-1.0 backend: {backend}")
        lazynwb.config.use_obstore = backend == "obstore"
        lazynwb.config.use_remfile = backend == "remfile"
        lazynwb.config.fsspec_storage_options = {"anon": True}
    elif backend != "obstore":
        raise ValueError(f"lazynwb {version} uses obstore; got backend {backend!r}.")


def _lazynwb_version() -> str:
    """Return the installed lazynwb distribution version."""
    return importlib.metadata.version("lazynwb")


def _set_catalog_cache_path() -> Path:
    """Point lazynwb at a matrix-provided cache or a fresh isolated cache."""
    cache_path = _settings().lazynwb_cache_path
    if cache_path is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
        cache_path = cache_dir / "catalog.sqlite"
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = cache_path.as_posix()
    return cache_path


def _local_cache() -> neurodatabench.models.LocalCacheState:
    """Return local cache metadata declared for this run."""
    return _settings().local_cache


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Remove the selected lazynwb catalog before a cold run."""
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.data_sources[_SOURCE_TYPE]),
    )
    cache_path = _set_catalog_cache_path()
    for path in (cache_path, Path(f"{cache_path}-shm"), Path(f"{cache_path}-wal")):
        path.unlink(missing_ok=True)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))
    _set_catalog_cache_path()
    os.environ.setdefault("AWS_REGION", _settings().aws_region)

    lazynwb.config.anon = True
    _configure_backend(_backend())
    state.clear()


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug("Clearing lazynwb state for %d NWB paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))
    state.clear()
#endregion


if __name__ == "__main__":
    settings = Settings()
    neurodatabench.main(
        implementation_id=settings.implementation_id,
        implementation_nwb_interface="lazynwb",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=_local_cache(),
        implementation_remote_cache=False,
        benchmark=settings.benchmark,
        setup=setup,
        clear_cache=None if _local_cache() == "warm" else clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
