# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "h5py",
#   "numpy",
#   "polars==1.38.1",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "remfile",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///

"""Runnable parquet-component implementation for the dynamic routing benchmark."""

from __future__ import annotations

import functools
import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import quote

import h5py
import numpy as np
import polars as pl
import remfile

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_DEFAULT_BENCHMARK = "dynamic_routing_nwb_hdf5_v0"
_DEFAULT_COMPONENT_BASE_URL = (
    "s3://aind-scratch-data/"
    "dynamic-routing/cache/nwb_components/v0.0.273"
)
_DEFAULT_IMPLEMENTATION_ID = "parquet_components"
_OBJECT_STORE_BACKEND = "polars_s3_anon"
_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128


def setup(context: neurodatabench.RunContext) -> None:
    """Configure process-level settings before answering benchmark questions."""
    logger.debug(
        "Preparing parquet-component access for %d NWB-derived sessions.",
        len(context.benchmark.data_sources),
    )
    if context.benchmark.id != _DEFAULT_BENCHMARK:
        raise ValueError(
            "The parquet component cache mirrors dynamic_routing_nwb_hdf5_v0; "
            f"got {context.benchmark.id!r}."
        )
    _quiet_storage_debug_loggers()


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear implementation-managed caches before timed benchmark phases."""
    logger.debug(
        "No local parquet-component cache to clear for %d NWB paths.",
        len(context.benchmark.data_sources),
    )


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    data_sources = context.benchmark.data_sources
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _count_visp_default_qc(data_sources)
            case "predicated_spike_times":
                answer = _longest_isi_for_fastest_visp_unit(data_sources)
            case "multisession_table_query":
                answer = _multisession_table_query(data_sources)
            case "large_array":
                answer = _large_array(data_sources)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug(
        "Parquet-component benchmark teardown for %d NWB paths.",
        len(context.benchmark.data_sources),
    )


def _quiet_storage_debug_loggers() -> None:
    """Keep benchmark debug logs focused on implementation-level events."""
    for logger_name in ("fsspec", "requests", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _component_base_url() -> str:
    """Return the remote parquet component cache root."""
    return os.environ.get(
        "NDB_PARQUET_COMPONENT_BASE_URL",
        _DEFAULT_COMPONENT_BASE_URL,
    ).rstrip("/")


def _component_url(component_name: str, session_id: str) -> str:
    """Return the URL for one per-session parquet component."""
    return f"{_component_base_url()}/{component_name}/{session_id}.parquet"

@functools.cache
def _scan_component(component_name: str, nwb_path: str) -> pl.LazyFrame:
    """Open one per-session parquet component as a Polars lazy frame."""
    session_id = _session_id_from_nwb_path(nwb_path)
    url = _component_url(component_name, session_id)
    logger.debug(
        "Scanning parquet component %s for session %s.",
        component_name,
        session_id,
    )
    return pl.scan_parquet(url, storage_options=_storage_options_for(url))


def _storage_options_for(url: str) -> dict[str, str] | None:
    """Return anonymous object-store options for S3 parquet component reads."""
    if not url.startswith("s3://"):
        return None
    return {
        "skip_signature": "true",
        "region": os.environ.get("AWS_REGION", "us-west-2"),
    }


def _session_id_from_nwb_path(nwb_path: str) -> str:
    """Return the dynamic-routing session id used by parquet component files."""
    name = nwb_path.rstrip("/").rsplit("/", maxsplit=1)[-1]
    for suffix in (".nwb.zarr", ".nwb", ".zarr"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name.removesuffix(".parquet")


def _count_visp_default_qc(data_sources: list[str]) -> int:
    """Count VISp units passing default QC across all parquet units tables."""
    count = 0
    for nwb_path in data_sources:
        session_count = (
            _scan_component("units", nwb_path)
            .select(
                (
                    (pl.col("structure") == "VISp")
                    & pl.col("default_qc").fill_null(False)
                )
                .sum()
                .alias("count")
            )
            .collect()
            .item()
        )
        count += int(session_count)
    return count


def _longest_isi_for_fastest_visp_unit(data_sources: list[str]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_path: str | None = None
    top_row = -1
    top_firing_rate = -np.inf
    top_unit_id: str | None = None

    for nwb_path in data_sources:
        candidate = (
            _scan_component("units", nwb_path)
            .with_row_index("_row_index")
            .filter(
                (pl.col("structure") == "VISp")
                & pl.col("firing_rate").is_not_null()
                & ~pl.col("firing_rate").is_nan()
            )
            .select("_row_index", "firing_rate", "unit_id")
            .sort("firing_rate", descending=True)
            .head(1)
            .collect()
        )
        if candidate.is_empty():
            logger.debug("No VISp units with firing_rate in %s.", nwb_path)
            continue
        local_rate = float(candidate["firing_rate"][0])
        if local_rate > top_firing_rate:
            top_path = nwb_path
            top_row = int(candidate["_row_index"][0])
            top_firing_rate = local_rate
            top_unit_id = str(candidate["unit_id"][0])

    if top_path is None:
        raise ValueError("No VISp unit with a finite firing_rate was found.")

    logger.debug(
        "Fetching spike_times for fastest VISp unit %s at row %d.",
        top_unit_id,
        top_row,
    )
    spike_times = _unit_spike_times(top_path, top_row)
    return float(np.diff(spike_times).max())


def _unit_spike_times(nwb_path: str, row_index: int) -> np.ndarray:
    """Return spike times for one unit row from a parquet units table."""
    spike_times = (
        _scan_component("units", nwb_path)
        .with_row_index("_row_index")
        .filter(pl.col("_row_index") == row_index)
        .select("spike_times")
        .collect()
    )
    if spike_times.height != 1:
        raise ValueError(f"Expected one unit row at index {row_index}; got {spike_times.height}.")
    return np.asarray(spike_times["spike_times"][0], dtype=np.float64)


def _multisession_table_query(data_sources: list[str]) -> float:
    """Compute the mean trial duration across all parquet trials tables."""
    total_duration = 0.0
    total_trials = 0
    for nwb_path in data_sources:
        trial_summary = (
            _scan_component("trials", nwb_path)
            .select(
                (pl.col("stop_time") - pl.col("start_time")).sum().alias("duration"),
                pl.len().alias("count"),
            )
            .collect()
        )
        total_duration += float(trial_summary["duration"][0])
        total_trials += int(trial_summary["count"][0])
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


def _large_array(data_sources: list[str]) -> float:
    """Return the facemap block mean using the source NWB file for this non-parquet question."""
    if not data_sources:
        raise ValueError("At least one NWB path is required.")
    logger.debug("Reading facemap data from source NWB because no parquet component exists.")
    with _open_nwb(data_sources[0]) as nwb_file:
        facemap_data = nwb_file["processing"]["behavior"]["facemap_side_camera"]["data"]
        data = np.asarray(
            facemap_data[:_FACEMAP_DOWNLOAD_ROWS, :_FACEMAP_DOWNLOAD_COLUMNS],
            dtype=np.float32,
        )
    return float(np.mean(data, dtype=np.float64))


@contextmanager
def _open_nwb(nwb_path: str) -> Iterator[h5py.File]:
    """Open one remote NWB HDF5 file read-only for the facemap-only fallback."""
    logger.debug("Opening source NWB HDF5 file %s for facemap fallback.", nwb_path)
    file_obj = remfile.File(_to_https_url(nwb_path))
    try:
        with h5py.File(file_obj, mode="r") as nwb_file:
            yield nwb_file
    finally:
        file_obj.close()


def _to_https_url(nwb_path: str) -> str:
    """Convert a public S3 URI to the HTTPS URL expected by remfile."""
    if nwb_path.startswith("https://") or nwb_path.startswith("http://"):
        return nwb_path
    if not nwb_path.startswith("s3://"):
        raise ValueError(f"Unsupported remote NWB path: {nwb_path}")
    bucket, key = nwb_path.removeprefix("s3://").split("/", maxsplit=1)
    return f"https://{bucket}.s3.amazonaws.com/{quote(key)}"


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            _DEFAULT_IMPLEMENTATION_ID,
        ),
        implementation_nwb_interface=None,
        implementation_object_store_backend=os.environ.get(
            "NDB_OBJECT_STORE_BACKEND",
            _OBJECT_STORE_BACKEND,
        ),
        implementation_local_cache=None,
        implementation_remote_cache=True,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
