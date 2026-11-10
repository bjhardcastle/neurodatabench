# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "hdmf-zarr",
#   "numpy",
#   "pandas",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "pynwb",
#   "s3fs",
#   "zarr<3",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


"""PyNWB NWBZarrIO answers for the change detection benchmark questions."""

from __future__ import annotations

import pydantic_settings
import warnings
import numcodecs.pickles
from typing import Any, Literal
from hdmf_zarr import NWBZarrIO

import neurodatabench

logger = neurodatabench.get_logger(__name__)
# Zarr 2 settting for decoding arrays with object-dtype stored with a pickle codec
numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True 

_SOURCE_TYPE = "zarr"

class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the PyNWB Zarr runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_ignore_unknown_args=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_",
    )

    benchmark: Literal["baseline_nwb_zarr_v0"] = "baseline_nwb_zarr_v0"
    object_store_backend: Literal["s3fs", "obstore"] = "s3fs"
    implementation_id: str = "pynwb"


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
                _load_trials_table(context.benchmark.data_sources[_SOURCE_TYPE][1])
            case "load_units_table":
                _load_units_table(context.benchmark.data_sources[_SOURCE_TYPE][0])
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


def _load_units_column(nwb_path: str, col_name: str) -> None:
    file = _open_file(nwb_path)
    nwb_file = file["nwb_file"]
    df = nwb_file.units.to_dataframe(exclude=set(nwb_file.units.colnames) - {col_name})


def _load_trials_table(nwb_path: str) -> Any:
    file = _open_file(nwb_path)
    nwb_file = file["nwb_file"]
    trials_df = nwb_file.trials.to_dataframe()
    assert not trials_df.empty
    return trials_df


def _load_units_table(nwb_path: str) -> Any:
    file = _open_file(nwb_path)
    nwb_file = file["nwb_file"]
    units_df = nwb_file.units.to_dataframe()
    assert not units_df.empty
    return units_df


def _load_multiple_tables(nwb_paths: list[str], n_tables: int, table_name: str) -> None:
    """Load multiple tables using pynwb"""
    n = len(nwb_paths)
    if n < n_tables:
        raise ValueError(f"{n_tables} NWB paths were requested, but only {n} were provided.")

    table_dfs = []
    for i in range(n_tables):
        if table_name == "intervals/trials":
            df = _load_trials_table(nwb_paths[i])
        elif table_name == "units":
            df = _load_units_table(nwb_paths[i])
        else:
            raise ValueError(f"Unsupported table name: {table_name}")
        table_dfs.append(df)
#endregion


#region Runtime Configuration
def _backend() -> str:
    """Return the requested NWBZarrIO object-store backend."""
    return _settings().object_store_backend


def _open_file(nwb_path: str) -> dict[str, Any]:
    """Return the file record for the given NWB path."""
    try:
        logger.debug("Opening %s through NWBZarrIO.", nwb_path)
        with warnings.catch_warnings():
            """Suppress column name conflict warnings from the hdmf library"""
            warnings.simplefilter("ignore")
            nwb_io = NWBZarrIO(
                path=nwb_path,
                mode="r",
                load_namespaces=True,
                storage_options={"anon": True},
            )
            nwb_file = nwb_io.read()
    except Exception as e:
        logger.error("Failed to open %s: %s", nwb_path, e)
        raise
    return {"nwb_io": nwb_io, "nwb_file": nwb_file}


def _open_files(data_sources: list[str]) -> None:
    """Materialize each remote Zarr store as a PyNWB NWBFile in `state`."""
    state.clear()
    state["files"] = []
    try:
        for nwb_path in data_sources:
            logger.debug("Opening %s through NWBZarrIO.", nwb_path)
            nwb_io = NWBZarrIO(
                path=nwb_path,
                mode="r",
                load_namespaces=True,
                storage_options={"anon": True},
            )
            file_record: dict[str, Any] = {"nwb_io": nwb_io}
            state["files"].append(file_record)
            file_record["nwb_file"] = nwb_io.read()
    except Exception:
        logger.debug("Closing partially opened NWBZarrIO handles after setup failure.")
        _close_files()
        raise


def _close_files() -> None:
    """Close every NWBZarrIO handle and clear implementation state."""
    for file_record in reversed(state.get("files", [])):
        file_record["nwb_io"].close()
    state.clear()


def setup(context: neurodatabench.RunContext) -> None:
    """Materialize each remote Zarr store as a PyNWB NWBFile."""
    backend = _backend()
    if backend != "s3fs":
        raise ValueError(f"NWBZarrIO does not support configured backend {backend!r}.")
    logger.debug(
        "Opening %d Zarr stores through NWBZarrIO and s3fs.",
        len(context.benchmark.data_sources[_SOURCE_TYPE]),
    )


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Declare that this implementation has no managed local cache."""
    logger.debug("No PyNWB Zarr cache to clear for %d paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))


def teardown(context: neurodatabench.RunContext) -> None:
    """Close NWBZarrIO handles and clear materialized state."""
    logger.debug("Closing NWBZarrIO handles for %d paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))
    _close_files()
#endregion


if __name__ == "__main__":
    settings = Settings()
    neurodatabench.main(
        implementation_id=settings.implementation_id,
        implementation_nwb_interface="pynwb/NWBZarrIO",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=settings.benchmark,
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )