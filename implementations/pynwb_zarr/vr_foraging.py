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


"""PyNWB NWBZarrIO answers for the VR foraging benchmark questions."""

import logging
from typing import Any

import numpy as np

import neurodatabench
import helpers


_STOP_VELOCITY_THRESHOLD = "VrForagingDataset.Behavior.SoftwareEvents.StopVelocityThreshold"


logger = logging.getLogger(__name__)


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    files = helpers.files()
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_reward_rate":
                answer = _multisession_reward_rate(files)
            case "multisession_patch_stop":
                answer = _multisession_patch_stop(files)
            case "reward_lick_latency":
                answer = _reward_lick_latency(files[0])
            case "multisession_stopped_fraction":
                answer = _multisession_stopped_fraction(files)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _multisession_reward_rate(file_records: list[dict[str, Any]]) -> float:
    """Fraction of stopped-at `RewardSite` trials that were rewarded, across all sessions."""
    total_choices = 0
    total_rewards = 0
    for file_record in file_records:
        trials = helpers.trials_frame(file_record)
        is_reward_site = helpers.string_array(trials["site_label"]) == "RewardSite"
        has_choice = np.asarray(trials["has_choice"], dtype=bool)
        has_reward = np.asarray(trials["has_reward"], dtype=bool)
        total_choices += int(np.count_nonzero(is_reward_site & has_choice))
        total_rewards += int(np.count_nonzero(is_reward_site & has_reward))
    if total_choices == 0:
        raise ValueError("No RewardSite trials with has_choice were found.")
    return total_rewards / total_choices


def _multisession_patch_stop(file_records: list[dict[str, Any]]) -> float:
    """Mean number of stopped-at `RewardSite` trials per patch visit, across all sessions."""
    stops_per_patch: list[float] = []
    for file_record in file_records:
        trials = helpers.trials_frame(file_record)
        is_reward_site = helpers.string_array(trials["site_label"]) == "RewardSite"
        has_choice = np.asarray(trials["has_choice"], dtype=bool)
        patch_index = np.asarray(trials["patch_index"], dtype=np.int64)

        _, visit = np.unique(patch_index[is_reward_site], return_inverse=True)
        stops_per_patch.extend(np.bincount(visit, weights=has_choice[is_reward_site]).tolist())
    return float(np.mean(stops_per_patch))


def _reward_lick_latency(first_file_record: dict[str, Any]) -> float:
    """Median seconds from each reward onset to the next lick onset in the first session."""
    nwb_file = first_file_record["nwb_file"]
    reward_onset = np.asarray(
        helpers.trials_frame(first_file_record)["reward_onset_time"], dtype=np.float64
    )

    licks = nwb_file.processing["behavior"]["licks"]
    is_onset = np.asarray(licks.data[:], dtype=bool)
    lick_onsets = np.sort(np.asarray(licks.timestamps[:], dtype=np.float64)[is_onset])

    next_lick = np.searchsorted(lick_onsets, reward_onset, side="left")
    has_next_lick = next_lick < lick_onsets.size

    latencies = lick_onsets[next_lick[has_next_lick]] - reward_onset[has_next_lick]
    return float(np.median(latencies))


def _multisession_stopped_fraction(file_records: list[dict[str, Any]]) -> float:
    """Fraction of all treadmill velocity samples below each session's stop threshold."""
    total_below = 0
    total_samples = 0
    for file_record in file_records:
        nwb_file = file_record["nwb_file"]
        threshold = float(nwb_file.acquisition[_STOP_VELOCITY_THRESHOLD]["data"].data[0])
        velocity = np.asarray(
            nwb_file.processing["behavior"]["position_velocity"]["velocity"].data[:],
            dtype=np.float64,
        )
        total_below += int(np.count_nonzero(velocity < threshold))
        total_samples += int(velocity.size)
    return total_below / total_samples
