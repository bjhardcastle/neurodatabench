# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "lazynwb",
#   "numpy",
#   "polars",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


"""lazynwb answers for the vr foraging benchmark questions."""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import lazynwb
import numpy as np
import polars as pl
import pydantic_settings

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_SOURCE_TYPE = "zarr"
_STOP_VELOCITY_THRESHOLD_PATH = (
    "/acquisition/VrForagingDataset.Behavior.SoftwareEvents.StopVelocityThreshold"
)


class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the lazynwb runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True, cli_kebab_case=True, cli_parse_args=True, env_prefix="NDB_"
    )

    benchmark: Literal["vr_foraging_nwb_v0"] = "vr_foraging_nwb_v0"
    object_store_backend: Literal["obstore", "remfile", "s3fs"] = "obstore"
    implementation_id: str = "lazynwb"
    local_cache: Literal["cold", "warm"] = "cold"
    lazynwb_cache_path: Path | None = None
    aws_region: str = "us-west-2"

state: dict[str, Any] = {}
settings: Settings | None = None


def _settings() -> Settings:
    """Return the resolved implementation settings."""
    if settings is None:
        raise RuntimeError("Settings must be initialized before running the implementation.")
    return settings

#region VR Foraging solutions
def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_reward_rate":
                answer = _multisession_reward_rate(_get_trials_table(context.benchmark.data_sources[_SOURCE_TYPE]))
            case "multisession_patch_stop":
                answer = _multisession_patch_stop(_get_trials_table(context.benchmark.data_sources[_SOURCE_TYPE]))
            case "reward_lick_latency":
                answer = _reward_lick_latency(context.benchmark.data_sources[_SOURCE_TYPE][0])
            case "multisession_stopped_fraction":
                answer = _multisession_stopped_fraction(context.benchmark.data_sources[_SOURCE_TYPE])
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
#endregion


#region Runtime Configuration
def _backend() -> str:
    """Return the requested lazynwb object-store backend label."""
    return _settings().object_store_backend


def _configure_backend(backend: str) -> None:
    """Configure backend switches exposed by the installed lazynwb version."""
    version = _lazynwb_version()
    logger.debug("Configuring lazynwb %s backend %s.", version, backend)
    if version.split(".", maxsplit=1)[0] == "0":
        if backend not in {"obstore", "remfile", "s3fs"}:
            raise ValueError(f"Unsupported lazynwb pre-1.0 backend: {backend}")
        lazynwb.config.use_obstore = backend == "obstore"
        lazynwb.config.use_remfile = backend == "remfile"
        lazynwb.config.fsspec_storage_options = {"anon": True}
    elif backend != "obstore":
        raise ValueError(f"lazynwb {version} uses obstore; got backend {backend!r}.")


def _lazynwb_version() -> str:
    """Return the installed lazynwb distribution version."""
    return importlib.metadata.version("lazynwb")


def _set_catalog_cache_path() -> Path:
    """Point lazynwb at a matrix-provided cache or a fresh isolated cache."""
    cache_path = _settings().lazynwb_cache_path
    if cache_path is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
        cache_path = cache_dir / "catalog.sqlite"
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = cache_path.as_posix()
    return cache_path


def _local_cache() -> neurodatabench.models.LocalCacheState:
    """Return local cache metadata declared for this run."""
    return _settings().local_cache


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Remove the selected lazynwb catalog before a cold run."""
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.data_sources[_SOURCE_TYPE]),
    )
    cache_path = _set_catalog_cache_path()
    for path in (cache_path, Path(f"{cache_path}-shm"), Path(f"{cache_path}-wal")):
        path.unlink(missing_ok=True)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))
    _set_catalog_cache_path()
    os.environ.setdefault("AWS_REGION", _settings().aws_region)

    lazynwb.config.anon = True
    _configure_backend(_backend())
    state.clear()


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug("Clearing lazynwb state for %d NWB paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))
    state.clear()
#endregion


if __name__ == "__main__":
    settings = Settings()
    neurodatabench.main(
        implementation_id=settings.implementation_id,
        implementation_nwb_interface="lazynwb",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=_local_cache(),
        implementation_remote_cache=False,
        benchmark=settings.benchmark,
        setup=setup,
        clear_cache=None if _local_cache() == "warm" else clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
