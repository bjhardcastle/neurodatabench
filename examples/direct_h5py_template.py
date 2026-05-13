# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "h5py",
#   "numpy",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "remfile",
# ]
# ///

"""Runnable direct h5py implementation for the packaged NWB benchmark."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.exists():
    sys.path.insert(0, _REPO_SRC.as_posix())

import h5py
import numpy as np
import remfile

import neurodatabench

logger = logging.getLogger(__name__)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure process-level settings before answering benchmark questions."""
    logger.debug(
        "Preparing direct h5py/remfile access for %d NWB files.",
        len(context.benchmark.nwb_paths),
    )
    _quiet_storage_debug_loggers()


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear implementation-managed caches before timed benchmark phases."""
    logger.debug(
        "No local direct h5py/remfile disk cache to clear for %d NWB paths.",
        len(context.benchmark.nwb_paths),
    )


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "units_VISp_default_qc":
                answer = _count_visp_default_qc(context.benchmark.nwb_paths)
            case "mean_inter_spike_interval":
                answer = _longest_isi_for_fastest_visp_unit(
                    context.benchmark.nwb_paths,
                )
            case "mean_trial_length":
                answer = _mean_trial_length(context.benchmark.nwb_paths)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug(
        "Direct h5py benchmark teardown for %d NWB paths.",
        len(context.benchmark.nwb_paths),
    )


def _quiet_storage_debug_loggers() -> None:
    """Keep benchmark debug logs focused on implementation-level events."""
    for logger_name in ("requests", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


@contextmanager
def _open_nwb(nwb_path: str) -> Iterator[h5py.File]:
    """Open one remote NWB HDF5 file read-only through remfile."""
    logger.debug("Opening NWB HDF5 file %s.", nwb_path)
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


def _count_visp_default_qc(nwb_paths: list[str]) -> int:
    """Count VISp units passing default QC across all NWB files."""
    count = 0
    for nwb_path in nwb_paths:
        with _open_nwb(nwb_path) as nwb_file:
            units = nwb_file["units"]
            structure = _read_string_array(units["structure"])
            default_qc = np.asarray(units["default_qc"][:], dtype=np.bool_)
            count += int(np.count_nonzero((structure == "VISp") & default_qc))
    return count


def _longest_isi_for_fastest_visp_unit(nwb_paths: list[str]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_path: str | None = None
    top_row = -1
    top_firing_rate = -np.inf

    for nwb_path in nwb_paths:
        with _open_nwb(nwb_path) as nwb_file:
            units = nwb_file["units"]
            structure = _read_string_array(units["structure"])
            firing_rate = np.asarray(units["firing_rate"][:], dtype=np.float64)
            candidate_rows = np.flatnonzero(
                (structure == "VISp") & ~np.isnan(firing_rate),
            )
            if candidate_rows.size == 0:
                logger.debug("No VISp units with firing_rate in %s.", nwb_path)
                continue
            local_row = int(candidate_rows[np.argmax(firing_rate[candidate_rows])])
            local_rate = float(firing_rate[local_row])
            if local_rate > top_firing_rate:
                top_path = nwb_path
                top_row = local_row
                top_firing_rate = local_rate

    if top_path is None:
        raise ValueError("No VISp unit with a finite firing_rate was found.")

    logger.debug(
        "Fetching spike_times for fastest VISp unit in %s at row %d.",
        top_path,
        top_row,
    )
    with _open_nwb(top_path) as nwb_file:
        spike_times = _get_unit_spike_times(nwb_file["units"], top_row)
    return float(np.diff(spike_times).max())


def _get_unit_spike_times(units: h5py.Group, unit_index: int) -> np.ndarray:
    """Return spike times for one unit from an HDF5-backed Units table."""
    spike_times_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
    start = 0 if unit_index == 0 else int(spike_times_index[unit_index - 1])
    stop = int(spike_times_index[unit_index])
    return np.asarray(units["spike_times"][start:stop], dtype=np.float64)


def _mean_trial_length(nwb_paths: list[str]) -> float:
    """Compute the mean trial duration across all NWB files."""
    total_duration = 0.0
    total_trials = 0
    for nwb_path in nwb_paths:
        with _open_nwb(nwb_path) as nwb_file:
            trials = nwb_file["intervals"]["trials"]
            start_time = np.asarray(trials["start_time"][:], dtype=np.float64)
            stop_time = np.asarray(trials["stop_time"][:], dtype=np.float64)
            total_duration += float(np.sum(stop_time - start_time))
            total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


def _read_string_array(dataset: h5py.Dataset) -> np.ndarray:
    """Read an HDF5 string dataset as a NumPy array of Python strings."""
    values = dataset.asstr()[:]
    return np.asarray(values, dtype=str)


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id="direct_h5py_remfile",
        implementation_local_cache=None,
        implementation_remote_cache=None,
        benchmark="dynamic_routing_hdf5_v0",
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
