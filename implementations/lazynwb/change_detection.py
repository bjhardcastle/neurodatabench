"""lazynwb answers for the change detection benchmark questions."""

import logging

import lazynwb
import numpy as np
import polars as pl

import neurodatabench
import helpers


_RUNNING_SPEED_DOWNLOAD_SAMPLES = 20_000


logger = logging.getLogger(__name__)


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_trials_hit_rate":
                answer = _multisession_trials_hit_rate(helpers.trials())
            case "max_speed_stimulus":
                answer = _max_speed_stimulus(context.benchmark.data_sources[0])
            case "multisession_lick_rate_average":
                answer = _multisession_lick_rate_average(context.benchmark.data_sources)
            case "change_detection_large_array":
                answer = _change_detection_large_array(context.benchmark.data_sources[0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _multisession_trials_hit_rate(trials: pl.LazyFrame) -> float:
    """Fraction of `intervals/trials` rows that were hits across all sessions."""
    all_hit = trials.select(pl.col("hit")).collect()
    hit_only = all_hit.filter(pl.col("hit").eq(True))
    return float(hit_only.height / all_hit.height)


def _max_speed_stimulus(nwb_path: str) -> str:
    """Stimulus with the highest mean running speed in the first session."""
    stim = (
        lazynwb.scan_nwb(nwb_path, "/intervals/stimulus_presentations")
        .select("image_name", "start_time", "stop_time")
        .collect()
    )
    speed = (
        lazynwb.scan_nwb(nwb_path, "/processing/running/speed")
        .select("data", "timestamps")
        .collect()
    )
    # Assign each speed sample to the presentation whose start_time most
    # recently precedes it, then keep only samples inside the presentation.
    samples = (
        speed
        .sort("timestamps")
        .join_asof(
            stim.sort("start_time"),
            left_on="timestamps",
            right_on="start_time",
            strategy="backward",
        )
        .filter(pl.col("timestamps") < pl.col("stop_time"))
    )
    return str(
        samples
        .group_by("image_name")
        .agg(pl.col("data").mean().alias("mean_speed"))
        .sort("mean_speed", descending=True)
        .head(1)["image_name"]
        .item()
    )


def _multisession_lick_rate_average(nwb_paths: list[str]) -> float:
    """Highest per-session mean lick rate (licks / second) across sessions."""
    events = lazynwb.scan_nwb(
        nwb_paths,
        "/events/events",
        disable_progress=True,
    )
    rates = (
        events
        .select(
            lazynwb.NWB_PATH_COLUMN_NAME,
            "event_type",
            "timestamp",
        )
        .group_by(lazynwb.NWB_PATH_COLUMN_NAME)
        .agg(
            (
                pl.col("event_type").eq("lick").sum()
                / (pl.col("timestamp").max() - pl.col("timestamp").min())
            ).alias("lick_rate_hz")
        )
        .sort("lick_rate_hz", descending=True)
        .collect()
    )
    return float(rates["lick_rate_hz"][0])


def _change_detection_large_array(nwb_path: str) -> float:
    """Mean of the first 20,000 samples of `processing/running/speed/data`."""
    speed = lazynwb.get_timeseries(
        nwb_path,
        "/processing/running/speed",
        exact_path=True,
    )
    data = np.asarray(
        speed.data[:_RUNNING_SPEED_DOWNLOAD_SAMPLES],
        dtype=np.float64,
    )
    return float(np.mean(data))
