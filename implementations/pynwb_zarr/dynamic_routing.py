"""PyNWB NWBZarrIO answers for the dynamic routing benchmark questions."""

import logging
from typing import Any

import numpy as np
import pandas as pd

import neurodatabench
import helpers


_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128


logger = logging.getLogger(__name__)


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    files = helpers.files()
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _count_visp_default_qc(files)
            case "predicated_spike_times":
                answer = _longest_isi_for_fastest_visp_unit(files)
            case "multisession_table_query":
                answer = _multisession_table_query(files)
            case "large_array":
                answer = _large_array(files)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _unit_metrics(file_record: dict[str, Any]) -> pd.DataFrame:
    """Return the cached units DataFrame without heavy ragged columns."""
    if "unit_metrics" not in file_record:
        nwb_file = file_record["nwb_file"]
        if nwb_file.units is None:
            raise ValueError(f"NWBFile {nwb_file.identifier} does not contain units.")
        file_record["units"] = nwb_file.units
        file_record["unit_metrics"] = nwb_file.units.to_dataframe(
            exclude={
                "spike_times",
                "spike_amplitudes",
                "waveform_mean",
                "waveform_std",
            },
        )
    return file_record["unit_metrics"]


def _count_visp_default_qc(file_records: list[dict[str, Any]]) -> int:
    """Count VISp units passing default QC across opened PyNWB NWBFiles."""
    count = 0
    for file_record in file_records:
        units = _unit_metrics(file_record)
        structure = helpers.string_array(units["structure"])
        default_qc = np.asarray(units["default_qc"], dtype=np.bool_)
        count += int(np.count_nonzero((structure == "VISp") & default_qc))
    return count


def _longest_isi_for_fastest_visp_unit(file_records: list[dict[str, Any]]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_file_record: dict[str, Any] | None = None
    top_row = -1
    top_firing_rate = -np.inf
    for file_record in file_records:
        units = _unit_metrics(file_record)
        structure = helpers.string_array(units["structure"])
        firing_rate = np.asarray(units["firing_rate"], dtype=np.float64)
        candidate_rows = np.flatnonzero((structure == "VISp") & ~np.isnan(firing_rate))
        if candidate_rows.size == 0:
            logger.debug("No VISp units with firing_rate in %s.", file_record["nwb_file"].identifier)
            continue
        local_row = int(candidate_rows[np.argmax(firing_rate[candidate_rows])])
        local_rate = float(firing_rate[local_row])
        if local_rate > top_firing_rate:
            top_file_record = file_record
            top_row = local_row
            top_firing_rate = local_rate
    if top_file_record is None:
        raise ValueError("No VISp unit with a finite firing_rate was found.")
    spike_times = np.asarray(
        top_file_record["units"].get_unit_spike_times(top_row),
        dtype=np.float64,
    )
    return float(np.diff(spike_times).max())


def _multisession_table_query(file_records: list[dict[str, Any]]) -> float:
    """Compute the mean trial duration across opened PyNWB NWBFiles."""
    total_duration = 0.0
    total_trials = 0
    for file_record in file_records:
        trials = helpers.trials_frame(file_record)
        start_time = np.asarray(trials["start_time"], dtype=np.float64)
        stop_time = np.asarray(trials["stop_time"], dtype=np.float64)
        total_duration += float(np.sum(stop_time - start_time))
        total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


def _large_array(file_records: list[dict[str, Any]]) -> float:
    """Return the mean of a 6.6 MB facemap block from the first NWBFile."""
    if not file_records:
        raise ValueError("At least one NWBFile record is required.")
    nwb_file = file_records[0]["nwb_file"]
    facemap = nwb_file.processing["behavior"]["facemap_side_camera"]
    data = np.asarray(
        facemap.data[:_FACEMAP_DOWNLOAD_ROWS, :_FACEMAP_DOWNLOAD_COLUMNS],
        dtype=np.float32,
    )
    return float(np.mean(data, dtype=np.float64))
