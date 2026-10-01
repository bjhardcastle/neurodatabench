# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "hdmf-zarr",
#   "numpy",
#   "pandas",
#   "pynwb",
#   "s3fs",
#   "zarr<3",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


"""PyNWB NWBZarrIO answers for the change detection benchmark questions."""

import logging
from typing import Any

import numpy as np

import neurodatabench
import helpers


_RUNNING_SPEED_DOWNLOAD_SAMPLES = 20_000


logger = logging.getLogger(__name__)


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    files = helpers.files()
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_trials_hit_rate":
                answer = _multisession_trials_hit_rate(files)
            case "max_speed_stimulus":
                answer = _max_speed_stimulus(files[0])
            case "multisession_lick_rate_average":
                answer = _multisession_lick_rate_average(files)
            case "change_detection_large_array":
                answer = _change_detection_large_array(files[0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _multisession_trials_hit_rate(file_records: list[dict[str, Any]]) -> float:
    """Fraction of `intervals/trials` rows that were hits across all sessions."""
    total_trials = 0
    total_hits = 0
    for file_record in file_records:
        trials = helpers.trials_frame(file_record)
        hit = np.asarray(trials["hit"], dtype=np.bool_)
        total_hits += int(np.count_nonzero(hit))
        total_trials += int(hit.size)
    return float(total_hits / total_trials)


def _max_speed_stimulus(first_file_record: dict[str, Any]) -> str:
    """Stimulus with the highest mean running speed in the first session."""
    nwb_file = first_file_record["nwb_file"]
    stim = nwb_file.intervals["stimulus_presentations"].to_dataframe()
    image_name = helpers.string_array(stim["image_name"])
    start_time = np.asarray(stim["start_time"], dtype=np.float64)
    stop_time = np.asarray(stim["stop_time"], dtype=np.float64)

    speed_series = nwb_file.processing["running"]["speed"]
    speed_data = np.asarray(speed_series.data[:], dtype=np.float64)
    speed_ts = np.asarray(speed_series.timestamps[:], dtype=np.float64)

    # Assign each speed sample to the presentation whose start_time most
    # recently precedes it (asof match), then keep only samples whose timestamp
    # is also inside that presentation's [start_time, stop_time] interval.
    order = np.argsort(start_time)
    sorted_starts = start_time[order]
    sorted_stops = stop_time[order]
    sorted_labels = image_name[order]

    candidate = np.searchsorted(sorted_starts, speed_ts, side="right") - 1
    within = candidate >= 0
    clipped = np.where(within, candidate, 0)
    within &= speed_ts <= sorted_stops[clipped]

    matched_labels = sorted_labels[clipped]

    best_label = ""
    best_mean = -np.inf
    for label in np.unique(matched_labels[within]):
        mask = within & (matched_labels == label)
        mean_speed = float(speed_data[mask].mean())
        if mean_speed > best_mean:
            best_mean = mean_speed
            best_label = str(label)
    return best_label


def _multisession_lick_rate_average(file_records: list[dict[str, Any]]) -> float:
    """Highest per-session mean lick rate (licks / second) across sessions."""
    best_rate = -np.inf
    for file_record in file_records:
        nwb_file = file_record["nwb_file"]
        events = nwb_file.events["events"].to_dataframe()
        event_type = helpers.string_array(events["event_type"])
        timestamps = np.asarray(events["timestamp"], dtype=np.float64)

        n_licks = int(np.count_nonzero(event_type == "lick"))
        duration = float(timestamps.max() - timestamps.min())

        rate = n_licks / duration
        if rate > best_rate:
            best_rate = rate
    return float(best_rate)


def _change_detection_large_array(first_file_record: dict[str, Any]) -> float:
    """Mean of the first 20,000 samples of `processing/running/speed/data`."""
    nwb_file = first_file_record["nwb_file"]
    speed_series = nwb_file.processing["running"]["speed"]
    data = np.asarray(
        speed_series.data[:_RUNNING_SPEED_DOWNLOAD_SAMPLES],
        dtype=np.float64,
    )
    return float(np.mean(data, dtype=np.float64))
