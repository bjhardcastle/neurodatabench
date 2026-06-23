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
# ]
# ///

"""Runnable PyNWB/HDMF-Zarr NWBFile implementation for the packaged NWB benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.exists():
    sys.path.insert(0, _REPO_SRC.as_posix())

import numpy as np
import pynwb
from hdmf_zarr import NWBZarrIO

import neurodatabench

logger = logging.getLogger(__name__)

state: dict[str, Any] = {}

_DEFAULT_BACKEND = "s3fs"
_DEFAULT_BENCHMARK = "dynamic_routing_zarr_v0"
_DEFAULT_IMPLEMENTATION_ID = "pynwb_hdmf_zarr_nwbfile"
_STORAGE_OPTIONS: dict[str, Any] = {"anon": True}
_UNITS_FRAME_COLUMNS = {"structure", "default_qc", "firing_rate"}
_TRIALS_FRAME_COLUMNS = {"start_time", "stop_time"}


def setup(context: neurodatabench.RunContext) -> None:
    """Open every benchmark NWB Zarr store as a PyNWB NWBFile."""
    logger.debug(
        "Opening %d NWB Zarr stores through %s/%s.",
        len(context.benchmark.nwb_paths),
        pynwb.__name__,
        NWBZarrIO.__name__,
    )
    os.environ.setdefault("AWS_REGION", "us-west-2")
    _quiet_storage_debug_loggers()
    state.clear()
    state["files"] = []

    try:
        for nwb_path in context.benchmark.nwb_paths:
            logger.debug("Opening PyNWB NWBFile from Zarr store %s.", nwb_path)
            nwb_io = _open_nwb_io(nwb_path)
            state["files"].append(
                {
                    "path": nwb_path,
                    "nwb_io": nwb_io,
                    "nwb_file": nwb_io.read(),
                },
            )
    except Exception:
        logger.debug("Closing partially opened PyNWB/HDMF-Zarr files after setup failure.")
        _close_open_ios()
        state.clear()
        raise


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear implementation-managed caches before timed benchmark phases."""
    logger.debug(
        "No local PyNWB/HDMF-Zarr cache to clear for %d NWB paths.",
        len(context.benchmark.nwb_paths),
    )


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "units_VISp_default_qc":
                answer = _count_visp_default_qc(state["files"])
            case "mean_inter_spike_interval":
                answer = _longest_isi_for_fastest_visp_unit(
                    state["files"],
                )
            case "mean_trial_length":
                answer = _mean_trial_length(state["files"])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Release PyNWB/HDMF-Zarr objects opened during setup."""
    logger.debug(
        "Closing PyNWB/HDMF-Zarr state for %d NWB paths.",
        len(context.benchmark.nwb_paths),
    )
    _close_open_ios()
    state.clear()


def _quiet_storage_debug_loggers() -> None:
    """Keep benchmark debug logs focused on implementation-level events."""
    for logger_name in ("aiobotocore", "botocore", "fsspec", "s3fs", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _backend() -> str:
    """Return the requested PyNWB/HDMF-Zarr object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def _open_nwb_io(nwb_path: str) -> NWBZarrIO:
    """Open one remote NWB Zarr store through HDMF-Zarr's PyNWB IO class."""
    backend = _backend()
    logger.debug("Opening NWB Zarr store %s through %s.", nwb_path, backend)
    if backend == "s3fs":
        return NWBZarrIO(path=nwb_path, mode="r", storage_options=_STORAGE_OPTIONS)
    if backend == "obstore":
        raise RuntimeError(
            "HDMF-Zarr NWBZarrIO only accepts paths or zarr-python directory stores; "
            "use the direct Zarr benchmark for obstore-backed Zarr access.",
        )
    if backend in {"remfile", "ros"}:
        raise RuntimeError(f"{backend} is a file backend and cannot open directory Zarr stores.")
    raise ValueError(f"Unsupported PyNWB/HDMF-Zarr backend: {backend}")


def _close_open_ios() -> None:
    """Close any HDMF-Zarr IO objects kept alive in module state."""
    for file_record in state.get("files", []):
        nwb_io = file_record.get("nwb_io")
        close = getattr(nwb_io, "close", None)
        if close is not None:
            close()


def _count_visp_default_qc(file_records: list[dict[str, Any]]) -> int:
    """Count VISp units passing default QC across opened PyNWB NWBFiles."""
    count = 0
    for file_record in file_records:
        units = _units_frame(file_record)
        structure = _string_array(units["structure"])
        default_qc = np.asarray(units["default_qc"], dtype=np.bool_)
        count += int(np.count_nonzero((structure == "VISp") & default_qc))
    return count


def _longest_isi_for_fastest_visp_unit(file_records: list[dict[str, Any]]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_file_record: dict[str, Any] | None = None
    top_row = -1
    top_firing_rate = -np.inf

    for file_record in file_records:
        nwb_file = file_record["nwb_file"]
        units = _units_frame(file_record)
        structure = _string_array(units["structure"])
        firing_rate = np.asarray(units["firing_rate"], dtype=np.float64)
        candidate_rows = np.flatnonzero((structure == "VISp") & ~np.isnan(firing_rate))
        if candidate_rows.size == 0:
            logger.debug("No VISp units with firing_rate in %s.", nwb_file.identifier)
            continue
        local_row = int(candidate_rows[np.argmax(firing_rate[candidate_rows])])
        local_rate = float(firing_rate[local_row])
        if local_rate > top_firing_rate:
            top_file_record = file_record
            top_row = local_row
            top_firing_rate = local_rate

    if top_file_record is None:
        raise ValueError("No VISp unit with a finite firing_rate was found.")

    top_nwb_file = top_file_record["nwb_file"]
    logger.debug(
        "Fetching spike_times for fastest VISp unit in %s at row %d.",
        top_nwb_file.identifier,
        top_row,
    )
    spike_times = np.asarray(
        top_file_record["units_table"].get_unit_spike_times(top_row),
        dtype=np.float64,
    )
    return float(np.diff(spike_times).max())


def _mean_trial_length(file_records: list[dict[str, Any]]) -> float:
    """Compute the mean trial duration across opened PyNWB NWBFiles."""
    total_duration = 0.0
    total_trials = 0
    for file_record in file_records:
        trials = _trials_frame(file_record)
        start_time = np.asarray(trials["start_time"], dtype=np.float64)
        stop_time = np.asarray(trials["stop_time"], dtype=np.float64)
        total_duration += float(np.sum(stop_time - start_time))
        total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


def _units_frame(file_record: dict[str, Any]) -> Any:
    """Return a cached units DataFrame with unneeded PyNWB columns excluded."""
    if "units_frame" not in file_record:
        nwb_file = file_record["nwb_file"]
        if nwb_file.units is None:
            raise ValueError(
                f"NWBFile {nwb_file.identifier} does not contain units.",
            )
        file_record["units_table"] = nwb_file.units
        file_record["units_frame"] = nwb_file.units.to_dataframe(
            exclude=_table_columns_to_exclude(nwb_file.units, _UNITS_FRAME_COLUMNS),
        )
    return file_record["units_frame"]


def _trials_frame(file_record: dict[str, Any]) -> Any:
    """Return a cached trials DataFrame with unneeded PyNWB columns excluded."""
    if "trials_frame" not in file_record:
        nwb_file = file_record["nwb_file"]
        if nwb_file.trials is None:
            raise ValueError(
                f"NWBFile {nwb_file.identifier} does not contain trials.",
            )
        file_record["trials_table"] = nwb_file.trials
        file_record["trials_frame"] = nwb_file.trials.to_dataframe(
            exclude=_table_columns_to_exclude(nwb_file.trials, _TRIALS_FRAME_COLUMNS),
        )
    return file_record["trials_frame"]


def _table_columns_to_exclude(table: Any, columns_to_keep: set[str]) -> set[str]:
    """Return table column names that should be skipped by PyNWB to_dataframe."""
    return {column_name for column_name in table.colnames if column_name not in columns_to_keep}


def _string_array(values: Any) -> np.ndarray:
    """Convert a table column to a NumPy string array."""
    raw_values = np.asarray(values)
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_values
        ],
        dtype=str,
    )


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            f"{_DEFAULT_IMPLEMENTATION_ID}_{_backend()}",
        ),
        implementation_nwb_interface="pynwb/hdmf-zarr",
        implementation_object_store_backend=_backend(),
        implementation_local_cache="cold",
        implementation_remote_cache=False,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
