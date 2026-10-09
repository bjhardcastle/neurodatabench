# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "lazynwb==1.0.0dev3",
#   "numpy",
#   "polars",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


"""LazyNWB implementation of the dynamic-foraging data-access benchmark."""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import lazynwb
import numpy as np
import pydantic_settings

import neurodatabench

logger = neurodatabench.get_logger(__name__)

SourceType = Literal["hdf5", "zarr"]
_UNIT_SELECTORS = (("671", "45883-2"), ("5", "46101"))
_SPIKE_TIME_DOWNLOAD_COUNT = 1_000_000


class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the LazyNWB runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_ignore_unknown_args=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_",
    )

    benchmark: str = "dynamic_foraging_nwb_v0"
    source_type: SourceType = "zarr"
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


# region Dynamic Foraging solutions
def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    data_sources = _data_sources(context)
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _table_counts(data_sources, "/units")
            case "predicated_spike_times":
                answer = _selected_spike_counts(data_sources)
            case "multisession_table_query":
                answer = _table_counts(data_sources, "/intervals/trials")
            case "large_array":
                answer = _large_array(data_sources[0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _data_sources(context: neurodatabench.RunContext) -> list[str]:
    """Return benchmark sources for the requested NWB storage type."""
    source_type = _settings().source_type
    try:
        data_sources = context.benchmark.data_sources[source_type]
    except KeyError as exc:
        raise ValueError(
            f"Benchmark {context.benchmark.id!r} has no {source_type!r} data sources."
        ) from exc
    if not data_sources:
        raise ValueError(
            f"Benchmark {context.benchmark.id!r} has an empty {source_type!r} source list."
        )
    return data_sources


def _table_counts(data_sources: list[str], table_path: str) -> list[int]:
    """Return each table's row count from its ID dataset shape, in source order."""
    counts: list[int] = []
    id_path = f"{table_path.rstrip('/')}/id"
    for nwb_path in data_sources:
        with lazynwb.FileAccessor(nwb_path) as nwb_file:
            counts.append(int(nwb_file[id_path].shape[0]))
    return counts


def _selected_spike_counts(data_sources: list[str]) -> list[int]:
    """Return spike counts for the benchmark's selected unit in each session."""
    if len(data_sources) != len(_UNIT_SELECTORS):
        raise ValueError(
            f"Expected {len(_UNIT_SELECTORS)} dynamic-foraging sessions, "
            f"found {len(data_sources)}."
        )

    counts: list[int] = []
    for nwb_path, (ks_unit_id, device_name) in zip(data_sources, _UNIT_SELECTORS):
        with lazynwb.FileAccessor(nwb_path) as nwb_file:
            ks_unit_ids = nwb_file["units/ks_unit_id"][:]
            device_names = nwb_file["units/device_name"][:]
            matching_rows = [
                row
                for row, (candidate_id, candidate_device) in enumerate(
                    zip(ks_unit_ids, device_names)
                )
                if _as_text(candidate_id) == ks_unit_id
                and _as_text(candidate_device) == device_name
            ]
            if len(matching_rows) != 1:
                raise ValueError(
                    f"Expected one unit with ks_unit_id={ks_unit_id!r} and "
                    f"device_name={device_name!r} in {nwb_path!r}, found "
                    f"{len(matching_rows)}."
                )
            counts.append(int(nwb_file["units/num_spikes"][matching_rows[0]]))
    return counts


def _as_text(value: Any) -> str:
    """Return an NWB scalar as comparable text."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _large_array(nwb_path: str) -> float:
    """Return the sum of the first million values in `units/spike_times`."""
    logger.debug(
        "Reading the first %d spike_times values from %s.",
        _SPIKE_TIME_DOWNLOAD_COUNT,
        nwb_path,
    )
    with lazynwb.FileAccessor(nwb_path) as nwb_file:
        spike_times = np.asarray(
            nwb_file["units/spike_times"][:_SPIKE_TIME_DOWNLOAD_COUNT],
            dtype=np.float64,
        )
    if spike_times.size != _SPIKE_TIME_DOWNLOAD_COUNT:
        raise ValueError(
            f"Expected {_SPIKE_TIME_DOWNLOAD_COUNT} spike times in {nwb_path!r}, "
            f"found {spike_times.size}."
        )
    return float(np.sum(spike_times, dtype=np.float64))


# endregion


# region Runtime configuration
def _backend() -> str:
    """Return the requested LazyNWB object-store backend label."""
    return _settings().object_store_backend


def _configure_backend(backend: str) -> None:
    """Configure backend switches exposed by the installed LazyNWB version."""
    version = _lazynwb_version()
    logger.debug("Configuring LazyNWB %s backend %s.", version, backend)
    if version.split(".", maxsplit=1)[0] == "0":
        if backend not in {"obstore", "remfile", "s3fs"}:
            raise ValueError(f"Unsupported LazyNWB pre-1.0 backend: {backend}")
        lazynwb.config.use_obstore = backend == "obstore"
        lazynwb.config.use_remfile = backend == "remfile"
        lazynwb.config.fsspec_storage_options = {"anon": True}
    elif backend != "obstore":
        raise ValueError(f"LazyNWB {version} uses obstore; got backend {backend!r}.")


def _lazynwb_version() -> str:
    """Return the installed LazyNWB distribution version."""
    return importlib.metadata.version("lazynwb")


def _set_catalog_cache_path() -> Path:
    """Point LazyNWB at a matrix-provided cache or a fresh isolated cache."""
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
    """Remove the selected LazyNWB catalog before a cold run."""
    logger.debug(
        "Clearing LazyNWB caches for %d %s paths before measured phases.",
        len(_data_sources(context)),
        _settings().source_type,
    )
    cache_path = _set_catalog_cache_path()
    for path in (cache_path, Path(f"{cache_path}-shm"), Path(f"{cache_path}-wal")):
        path.unlink(missing_ok=True)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure LazyNWB before answering benchmark questions."""
    logger.debug(
        "Preparing LazyNWB for %d %s paths.",
        len(_data_sources(context)),
        _settings().source_type,
    )
    _set_catalog_cache_path()
    os.environ.setdefault("AWS_REGION", _settings().aws_region)

    lazynwb.config.anon = True
    _configure_backend(_backend())


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug(
        "Completing LazyNWB run after reading %d %s paths.",
        len(_data_sources(context)),
        _settings().source_type,
    )


# endregion


if __name__ == "__main__":
    settings = Settings()
    neurodatabench.main(
        implementation_id=settings.implementation_id,
        implementation_source_type=settings.source_type,
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
