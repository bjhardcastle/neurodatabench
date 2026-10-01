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


"""Direct Zarr answers for the VR foraging benchmark questions."""

import numpy as np
import neurodatabench
import logging
import helpers


logger = logging.getLogger(__name__)


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""

    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_reward_rate":
                answer = _multisession_reward_rate(context.benchmark.data_sources)
            case "multisession_patch_stop":
                answer = _multisession_patch_stop(context.benchmark.data_sources)
            case "reward_lick_latency":
                answer = _reward_lick_latency(context.benchmark.data_sources[0])
            case "multisession_stopped_fraction":
                answer = _multisession_stopped_fraction(context.benchmark.data_sources)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _multisession_reward_rate(data_sources: list[str]) -> float:
    """
    Fraction of stopped-at `RewardSite` trials that were rewarded, across all sessions.
    """
    
    total_choices = 0
    total_rewards = 0
    for nwb_path in data_sources:
        trials = helpers.open_store(nwb_path)["intervals"]["trials"]
        is_reward_site = np.asarray(trials["site_label"][:]) == "RewardSite"
        has_choice = np.asarray(trials["has_choice"][:], dtype=bool)
        has_reward = np.asarray(trials["has_reward"][:], dtype=bool)

        session_choices = int(np.count_nonzero(is_reward_site & has_choice))
        session_rewards = int(np.count_nonzero(is_reward_site & has_reward))

        total_choices += session_choices
        total_rewards += session_rewards
    if total_choices == 0:
        raise ValueError("No RewardSite trials with has_choice were found.")
    return total_rewards / total_choices


def _multisession_patch_stop(data_sources: list[str]) -> float:
    """
    Mean number of stopped-at `RewardSite` trials per patch visit, across all sessions.
    """
    stops_per_patch: list[float] = []
    for nwb_path in data_sources:
        trials = helpers.open_store(nwb_path)["intervals"]["trials"]
        is_reward_site = np.asarray(trials["site_label"][:]) == "RewardSite"
        has_choice = np.asarray(trials["has_choice"][:], dtype=bool)
        patch_index = np.asarray(trials["patch_index"][:], dtype=np.int64)

        _, visit = np.unique(patch_index[is_reward_site], return_inverse=True)
        session_stops = np.bincount(visit, weights=has_choice[is_reward_site]).tolist()
        stops_per_patch.extend(session_stops)

    return float(np.mean(stops_per_patch))


def _reward_lick_latency(first_session_path: str) -> float:
    """
    Median seconds from each reward onset to the next lick onset in the first session.
    """

    store = helpers.open_store(first_session_path)
    reward_onset = np.asarray(
        store["intervals"]["trials"]["reward_onset_time"][:], dtype=np.float64
    )

    licks = store["processing"]["behavior"]["licks"]
    is_onset = np.asarray(licks["data"][:], dtype=bool)
    lick_onsets = np.sort(np.asarray(licks["timestamps"][:], dtype=np.float64)[is_onset])

    next_lick = np.searchsorted(lick_onsets, reward_onset, side="left")
    has_next_lick = next_lick < lick_onsets.size

    latencies = lick_onsets[next_lick[has_next_lick]] - reward_onset[has_next_lick]
    return float(np.median(latencies))


def _multisession_stopped_fraction(data_sources: list[str]) -> float:
    """
    Fraction of all treadmill velocity samples below each session's stop threshold.
    """

    stop_velocity_path = (
        "acquisition/VrForagingDataset.Behavior.SoftwareEvents.StopVelocityThreshold"
    )
    total_below = 0
    total_samples = 0
    for nwb_path in data_sources:
        store = helpers.open_store(nwb_path)

        threshold = float(store[stop_velocity_path]["data"][0])
        velocity = np.asarray(
            store["processing"]["behavior"]["position_velocity"]["velocity"][:],
            dtype=np.float64,
        )

        session_below = int(np.count_nonzero(velocity < threshold))
        total_below += session_below
        total_samples += velocity.size

    return total_below / total_samples
