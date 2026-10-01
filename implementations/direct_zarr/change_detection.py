# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "neurodatabench",
#   "zarr",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


import numpy as np
import neurodatabench
import logging
import helpers


logger = logging.getLogger(__name__)


_RUNNING_SPEED_DOWNLOAD_SAMPLES = 20_000


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_trials_hit_rate":
                answer = _multisession_trials_hit_rate(context.benchmark.data_sources)
            case "max_speed_stimulus":
                answer = _max_speed_stimulus(context.benchmark.data_sources[0])
            case "multisession_lick_rate_average":
                answer = _multisession_lick_rate_average(context.benchmark.data_sources)
            case "change_detection_large_array":
                answer = _change_detection_large_array(context.benchmark.data_sources[0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _multisession_trials_hit_rate(data_sources: list[str]) -> float:
    """Fraction of `intervals/trials` rows that were hits across all sessions."""
    total_trials = 0
    total_hits = 0
    for nwb_path in data_sources:
        trials = helpers.open_store(nwb_path)["intervals"]["trials"]
        all_hits = np.asarray(trials["hit"][:], dtype=bool)
        hit_only = all_hits[all_hits]
        total_hits += hit_only.size
        total_trials += all_hits.size

    return float(total_hits / total_trials)


def _max_speed_stimulus(first_session_path: str) -> str:
    """Stimulus with the highest mean running speed in the first session."""
    store = helpers.open_store(first_session_path)
    stim = store["intervals"]["stimulus_presentations"]
    image_name = np.asarray(stim["image_name"][:])
    start_time = np.asarray(stim["start_time"][:], dtype=np.float64)
    stop_time = np.asarray(stim["stop_time"][:], dtype=np.float64)

    speed_group = store["processing"]["running"]["speed"]
    speed_data = np.asarray(speed_group["data"][:], dtype=np.float64)
    speed_ts = np.asarray(speed_group["timestamps"][:], dtype=np.float64)

    # Assign each speed sample to the presentation whose start_time most
    # recently precedes it (asof match), then keep only samples that also lie
    # before that presentation's stop_time.
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


def _multisession_lick_rate_average(data_sources: list[str]) -> float:
    """Highest per-session mean lick rate (licks / second) across sessions."""
    best_rate = -np.inf
    for nwb_path in data_sources:
        events = helpers.open_store(nwb_path)["events"]["events"]
        event_type = helpers.string_array(events["event_type"][:])
        timestamps = np.asarray(events["timestamp"][:], dtype=np.float64)
        n_licks = int(np.count_nonzero(event_type == "lick"))
        duration = float(timestamps.max() - timestamps.min())

        rate = n_licks / duration
        if rate > best_rate:
            best_rate = rate
            
    return float(best_rate)


def _change_detection_large_array(first_session_path: str) -> float:
    """Mean of the first 20,000 samples of `processing/running/speed/data`."""
    store = helpers.open_store(first_session_path)
    data = np.asarray(
        store["processing"]["running"]["speed"]["data"][
            :_RUNNING_SPEED_DOWNLOAD_SAMPLES
        ],
        dtype=np.float64,
    )
    return float(np.mean(data, dtype=np.float64))