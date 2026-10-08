# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "lazynwb",
#   "numpy",
#   "polars",
#   "psutil",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


"""lazynwb answers for the change detection benchmark questions."""

from __future__ import annotations

import importlib
import os
import tempfile
from typing import Any
from pathlib import Path

import lazynwb
import polars as pl
import numpy as np

import neurodatabench

logger = neurodatabench.get_logger(__name__)


_DEFAULT_BENCHMARK = "change_detection_nwb_zarr_v0"
_DEFAULT_BACKEND = "obstore"
_RUNNING_SPEED_DOWNLOAD_SAMPLES = 20_000

state: dict[str, Any] = {}


#region Change Detection solutions
def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_trials_hit_rate":
                answer = _multisession_trials_hit_rate(_get_trials_table(context.benchmark.data_sources))
            case "max_speed_stimulus":
                answer = _max_speed_stimulus(context.benchmark.data_sources[0])
            case "multisession_lick_rate_average":
                answer = _multisession_lick_rate_average(context.benchmark.data_sources)
            case "change_detection_large_array":
                answer = _change_detection_large_array(context.benchmark.data_sources[0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _get_trials_table(data_sources: list[str]) -> pl.LazyFrame:
    """Return the cached multi-session units LazyFrame."""
    if "trials" not in state:
        state["trials"] = lazynwb.scan_nwb(
            data_sources,
            "/intervals/trials",
            disable_progress=True,
        )
    return state["trials"]


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
#endregion


#region Runtime Configuration
def backend() -> str:
    """Return the requested lazynwb object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def configure_backend(backend: str) -> None:
    """Configure backend switches exposed by the installed lazynwb version."""
    version = lazynwb_version()
    logger.debug("Configuring lazynwb %s backend %s.", version, backend)
    if version.split(".", maxsplit=1)[0] == "0":
        if backend not in {"obstore", "remfile", "s3fs"}:
            raise ValueError(f"Unsupported lazynwb pre-1.0 backend: {backend}")
        lazynwb.config.use_obstore = backend == "obstore"
        lazynwb.config.use_remfile = backend == "remfile"
        lazynwb.config.fsspec_storage_options = {"anon": True}
    elif backend != "obstore":
        raise ValueError(f"lazynwb {version} uses obstore; got backend {backend!r}.")


def lazynwb_version() -> str:
    """Return the installed lazynwb distribution version."""
    return importlib.metadata.version("lazynwb")


def set_catalog_cache_path() -> Path:
    """Point lazynwb at a matrix-provided cache or a fresh isolated cache."""
    cache_path = os.environ.get("NDB_LAZYNWB_CACHE_PATH")
    if cache_path is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
        cache_path = (cache_dir / "catalog.sqlite").as_posix()
        os.environ["NDB_LAZYNWB_CACHE_PATH"] = cache_path
    else:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = cache_path
    return Path(cache_path)


def local_cache() -> neurodatabench.models.LocalCacheState:
    """Return local cache metadata declared for this run."""
    value = os.environ.get("NDB_LOCAL_CACHE", "cold")
    if value not in {"cold", "warm"}:
        raise ValueError("NDB_LOCAL_CACHE must be 'cold' or 'warm' for lazynwb.")
    return value  # type: ignore[return-value]


def default_implementation_id() -> str:
    """Return an ID containing the installed lazynwb version and backend."""
    version = lazynwb_version().replace(".", "_").replace("+", "_")
    return f"lazynwb_{version}_{backend()}"


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Remove the selected lazynwb catalog before a cold run."""
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.data_sources),
    )
    cache_path = set_catalog_cache_path()
    for path in (cache_path, Path(f"{cache_path}-shm"), Path(f"{cache_path}-wal")):
        path.unlink(missing_ok=True)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.data_sources))
    set_catalog_cache_path()
    os.environ.setdefault("AWS_REGION", "us-west-2")

    lazynwb.config.anon = True
    configure_backend(backend())
    state.clear()


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug("Clearing lazynwb state for %d NWB paths.", len(context.benchmark.data_sources))
    state.clear()
#endregion


if __name__ == "__main__":
    benchmark = os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK)
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            default_implementation_id(),
        ),
        implementation_nwb_interface="lazynwb",
        implementation_object_store_backend=backend(),
        implementation_local_cache=local_cache(),
        implementation_remote_cache=False,
        benchmark=benchmark,
        setup=setup,
        clear_cache=None if local_cache() == "warm" else clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )