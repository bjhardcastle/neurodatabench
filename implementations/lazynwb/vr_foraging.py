"""lazynwb answers for the VR foraging benchmark questions."""

import logging

import lazynwb
import numpy as np
import polars as pl

import neurodatabench
import helpers


_STOP_VELOCITY_THRESHOLD_PATH = (
    "/acquisition/VrForagingDataset.Behavior.SoftwareEvents.StopVelocityThreshold"
)


logger = logging.getLogger(__name__)


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_reward_rate":
                answer = _multisession_reward_rate(helpers.trials())
            case "multisession_patch_stop":
                answer = _multisession_patch_stop(helpers.trials())
            case "reward_lick_latency":
                answer = _reward_lick_latency(context.benchmark.data_sources[0])
            case "multisession_stopped_fraction":
                answer = _multisession_stopped_fraction(context.benchmark.data_sources)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _multisession_reward_rate(trials: pl.LazyFrame) -> float:
    """Fraction of stopped-at `RewardSite` trials that were rewarded, across all sessions."""
    counts = (
        trials
        .filter(pl.col("site_label").eq("RewardSite"))
        .select(
            pl.col("has_choice").sum().alias("choices"),
            pl.col("has_reward").sum().alias("rewards"),
        )
        .collect()
    )
    choices = int(counts["choices"].item())
    if choices == 0:
        raise ValueError("No RewardSite trials with has_choice were found.")
    return int(counts["rewards"].item()) / choices


def _multisession_patch_stop(trials: pl.LazyFrame) -> float:
    """Mean number of stopped-at `RewardSite` trials per patch visit, across all sessions."""
    return float(
        trials
        .filter(pl.col("site_label").eq("RewardSite"))
        .group_by(lazynwb.NWB_PATH_COLUMN_NAME, "patch_index")
        .agg(pl.col("has_choice").sum().alias("stops"))
        .select(pl.col("stops").mean())
        .collect()
        .item()
    )


def _reward_lick_latency(nwb_path: str) -> float:
    """Median seconds from each reward onset to the next lick onset in the first session."""
    reward_onset = (
        lazynwb.scan_nwb(nwb_path, "/intervals/trials", disable_progress=True)
        .select("reward_onset_time")
        .collect()["reward_onset_time"]
        .to_numpy()
        .astype(np.float64)
    )
    licks = lazynwb.get_timeseries(
        nwb_path,
        "/processing/behavior/licks",
        exact_path=True,
    )
    is_onset = np.asarray(licks.data[:], dtype=bool)
    lick_onsets = np.sort(np.asarray(licks.timestamps[:], dtype=np.float64)[is_onset])

    next_lick = np.searchsorted(lick_onsets, reward_onset, side="left")
    has_next_lick = next_lick < lick_onsets.size

    latencies = lick_onsets[next_lick[has_next_lick]] - reward_onset[has_next_lick]
    return float(np.median(latencies))


def _multisession_stopped_fraction(nwb_paths: list[str]) -> float:
    """Fraction of all treadmill velocity samples below each session's stop threshold."""
    thresholds = (
        lazynwb.scan_nwb(nwb_paths, _STOP_VELOCITY_THRESHOLD_PATH, disable_progress=True)
        .filter(pl.col(lazynwb.TABLE_INDEX_COLUMN_NAME).eq(0))
        .select(
            lazynwb.NWB_PATH_COLUMN_NAME,
            pl.col("data").cast(pl.Utf8).cast(pl.Float64).alias("threshold"),
        )
    )
    counts = (
        lazynwb.scan_nwb(nwb_paths, "/processing/behavior/position_velocity", disable_progress=True)
        .select(lazynwb.NWB_PATH_COLUMN_NAME, "velocity")
        .join(thresholds, on=lazynwb.NWB_PATH_COLUMN_NAME)
        .select(
            (pl.col("velocity") < pl.col("threshold")).sum().alias("below"),
            pl.len().alias("samples"),
        )
        .collect()
    )
    return int(counts["below"].item()) / int(counts["samples"].item())
