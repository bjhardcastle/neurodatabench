# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "hdmf-zarr",
#   "numpy",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "pynwb",
#   "s3fs",
#   "zarr<3",
# ]
# ///

"""Runnable PyNWB/HDMF-Zarr implementation for the packaged NWB benchmark."""

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
import zarr
from hdmf_zarr import NWBZarrIO

import neurodatabench

logger = logging.getLogger(__name__)

_STORAGE_OPTIONS: dict[str, Any] = {"anon": True}


def setup(context: neurodatabench.RunContext) -> None:
    """Configure process-level settings before answering benchmark questions."""
    logger.debug(
        "Preparing %s/%s for %d NWB Zarr stores.",
        pynwb.__name__,
        NWBZarrIO.__name__,
        len(context.benchmark.nwb_paths),
    )
    os.environ.setdefault("AWS_REGION", "us-west-2")


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
        "PyNWB/HDMF-Zarr benchmark teardown for %d NWB paths.",
        len(context.benchmark.nwb_paths),
    )


def _open_store(nwb_path: str) -> zarr.hierarchy.Group:
    """Open one remote NWB Zarr store as a read-only Zarr group."""
    logger.debug("Opening NWB Zarr store %s.", nwb_path)
    return zarr.open(nwb_path, mode="r", storage_options=_STORAGE_OPTIONS)


def _count_visp_default_qc(nwb_paths: list[str]) -> int:
    """Count VISp units passing default QC across all NWB stores."""
    count = 0
    for nwb_path in nwb_paths:
        units = _open_store(nwb_path)["units"]
        structure = np.asarray(units["structure"][:])
        default_qc = np.asarray(units["default_qc"][:], dtype=np.bool_)
        count += int(np.count_nonzero((structure == "VISp") & default_qc))
    return count


def _longest_isi_for_fastest_visp_unit(nwb_paths: list[str]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_path: str | None = None
    top_row = -1
    top_firing_rate = -np.inf

    for nwb_path in nwb_paths:
        units = _open_store(nwb_path)["units"]
        structure = np.asarray(units["structure"][:])
        firing_rate = np.asarray(units["firing_rate"][:], dtype=np.float64)
        candidate_rows = np.flatnonzero((structure == "VISp") & ~np.isnan(firing_rate))
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
    units = _open_store(top_path)["units"]
    spike_times = _get_unit_spike_times(units, top_row)
    return float(np.diff(spike_times).max())


def _get_unit_spike_times(units: zarr.hierarchy.Group, unit_index: int) -> np.ndarray:
    """Return spike times for one unit from a Zarr-backed Units table."""
    spike_times_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
    start = 0 if unit_index == 0 else int(spike_times_index[unit_index - 1])
    stop = int(spike_times_index[unit_index])
    return np.asarray(units["spike_times"][start:stop], dtype=np.float64)


def _mean_trial_length(nwb_paths: list[str]) -> float:
    """Compute the mean trial duration across all NWB stores."""
    total_duration = 0.0
    total_trials = 0
    for nwb_path in nwb_paths:
        trials = _open_store(nwb_path)["intervals"]["trials"]
        start_time = np.asarray(trials["start_time"][:], dtype=np.float64)
        stop_time = np.asarray(trials["stop_time"][:], dtype=np.float64)
        total_duration += float(np.sum(stop_time - start_time))
        total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id="pynwb_hdmf_zarr_direct",
        implementation_nwb_interface=None,
        implementation_object_store_backend="s3fs",
        implementation_local_cache="cold",
        implementation_remote_cache=False,
        benchmark="dynamic_routing_zarr_v0",
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
