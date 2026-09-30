import numpy as np
import neurodatabench
import logging
from typing import Any
import helpers

_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128


logger = logging.getLogger(__name__)


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _count_visp_default_qc(context.benchmark.data_sources)
            case "predicated_spike_times":
                answer = _longest_isi_for_fastest_visp_unit(context.benchmark.data_sources)
            case "multisession_table_query":
                answer = _multisession_table_query(context.benchmark.data_sources)
            case "large_array":
                answer = _large_array(context.benchmark.data_sources)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _get_unit_spike_times(units: Any, unit_index: int) -> np.ndarray:
    """Return spike times for one unit from a Zarr-backed Units table."""
    spike_times_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
    start = 0 if unit_index == 0 else int(spike_times_index[unit_index - 1])
    stop = int(spike_times_index[unit_index])
    return np.asarray(units["spike_times"][start:stop], dtype=np.float64)


def _count_visp_default_qc(data_sources: list[str]) -> int:
    """Count VISp units passing default QC across all NWB stores."""
    count = 0
    for nwb_path in data_sources:
        units = helpers.open_store(nwb_path)["units"]
        structure = np.asarray(units["structure"][:])
        default_qc = np.asarray(units["default_qc"][:], dtype=np.bool_)
        count += int(np.count_nonzero((structure == "VISp") & default_qc))
    return count


def _longest_isi_for_fastest_visp_unit(data_sources: list[str]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_path: str | None = None
    top_row = -1
    top_firing_rate = -np.inf

    for nwb_path in data_sources:
        units = helpers.open_store(nwb_path)["units"]
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
    units = helpers.open_store(top_path)["units"]
    spike_times = _get_unit_spike_times(units, top_row)
    return float(np.diff(spike_times).max())


def _multisession_table_query(data_sources: list[str]) -> float:
    """Compute the mean trial duration across all NWB stores."""
    total_duration = 0.0
    total_trials = 0
    for nwb_path in data_sources:
        trials = helpers.open_store(nwb_path)["intervals"]["trials"]
        start_time = np.asarray(trials["start_time"][:], dtype=np.float64)
        stop_time = np.asarray(trials["stop_time"][:], dtype=np.float64)
        total_duration += float(np.sum(stop_time - start_time))
        total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


def _large_array(data_sources: list[str]) -> float:
    """Return the mean of a 6.6 MB facemap data block from the first NWB store."""
    if not data_sources:
        raise ValueError("At least one NWB path is required.")
    facemap = helpers.open_store(data_sources[0])["processing"]["behavior"]["facemap_side_camera"]
    data = np.asarray(
        facemap["data"][:_FACEMAP_DOWNLOAD_ROWS, :_FACEMAP_DOWNLOAD_COLUMNS],
        dtype=np.float32,
    )
    return float(np.mean(data, dtype=np.float64))