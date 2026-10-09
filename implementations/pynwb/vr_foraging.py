# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "hdmf-zarr",
#   "numpy",
#   "pandas",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "pynwb",
#   "s3fs",
#   "zarr<3",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


"""PyNWB NWBZarrIO answers for the VR foraging benchmark questions."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import pydantic_settings
from hdmf_zarr import NWBZarrIO

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_SOURCE_TYPE = "zarr"
_STOP_VELOCITY_THRESHOLD = "VrForagingDataset.Behavior.SoftwareEvents.StopVelocityThreshold"


class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the PyNWB Zarr runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True, cli_kebab_case=True, cli_parse_args=True, env_prefix="NDB_"
    )

    benchmark: Literal["vr_foraging_nwb_v0"] = "vr_foraging_nwb_v0"
    object_store_backend: Literal["s3fs", "obstore"] = "s3fs"
    implementation_id: str = "pynwb"


settings: Settings | None = None


def _settings() -> Settings:
    """Return the resolved implementation settings."""
    if settings is None:
        raise RuntimeError("Settings must be initialized before running the implementation.")
    return settings


state: dict[str, Any] = {}


#region VR Foraging solutions
def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    files = state["files"]
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
        trials = _get_trials_table(file_record)
        is_reward_site = _string_array(trials["site_label"]) == "RewardSite"
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
        trials = _get_trials_table(file_record)
        is_reward_site = _string_array(trials["site_label"]) == "RewardSite"
        has_choice = np.asarray(trials["has_choice"], dtype=bool)
        patch_index = np.asarray(trials["patch_index"], dtype=np.int64)

        _, visit = np.unique(patch_index[is_reward_site], return_inverse=True)
        stops_per_patch.extend(np.bincount(visit, weights=has_choice[is_reward_site]).tolist())
    return float(np.mean(stops_per_patch))


def _reward_lick_latency(first_file_record: dict[str, Any]) -> float:
    """Median seconds from each reward onset to the next lick onset in the first session."""
    nwb_file = first_file_record["nwb_file"]
    reward_onset = np.asarray(
        _get_trials_table(first_file_record)["reward_onset_time"], dtype=np.float64
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


def _get_trials_table(file_record: dict[str, Any]) -> pd.DataFrame:
    """Return the cached trials DataFrame for one opened NWBFile."""
    if "trials_frame" not in file_record:
        nwb_file = file_record["nwb_file"]
        if nwb_file.trials is None:
            raise ValueError(f"NWBFile {nwb_file.identifier} does not contain trials.")
        file_record["trials_frame"] = nwb_file.trials.to_dataframe()
    return file_record["trials_frame"]


def _string_array(values: Any) -> np.ndarray:
    """Convert a table column to a NumPy string array."""
    raw_values = np.asarray(values)
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_values
        ],
        dtype=str,
    )
#endregion


#region Runtime Configuration
def _backend() -> str:
    """Return the requested NWBZarrIO object-store backend."""
    return _settings().object_store_backend


def _open_files(data_sources: list[str]) -> None:
    """Materialize each remote Zarr store as a PyNWB NWBFile in `state`."""
    state.clear()
    state["files"] = []
    try:
        for nwb_path in data_sources:
            logger.debug("Opening %s through NWBZarrIO.", nwb_path)
            nwb_io = NWBZarrIO(
                path=nwb_path,
                mode="r",
                load_namespaces=True,
                storage_options={"anon": True},
            )
            file_record: dict[str, Any] = {"nwb_io": nwb_io}
            state["files"].append(file_record)
            file_record["nwb_file"] = nwb_io.read()
    except Exception:
        logger.debug("Closing partially opened NWBZarrIO handles after setup failure.")
        _close_files()
        raise


def _close_files() -> None:
    """Close every NWBZarrIO handle and clear implementation state."""
    for file_record in reversed(state.get("files", [])):
        file_record["nwb_io"].close()
    state.clear()


def setup(context: neurodatabench.RunContext) -> None:
    """Materialize each remote Zarr store as a PyNWB NWBFile."""
    backend = _backend()
    if backend != "s3fs":
        raise ValueError(f"NWBZarrIO does not support configured backend {backend!r}.")
    logger.debug(
        "Opening %d Zarr stores through NWBZarrIO and s3fs.",
        len(context.benchmark.data_sources[_SOURCE_TYPE]),
    )
    _open_files(context.benchmark.data_sources[_SOURCE_TYPE])


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Declare that this implementation has no managed local cache."""
    logger.debug("No PyNWB Zarr cache to clear for %d paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))


def teardown(context: neurodatabench.RunContext) -> None:
    """Close NWBZarrIO handles and clear materialized state."""
    logger.debug("Closing NWBZarrIO handles for %d paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))
    _close_files()
#endregion


if __name__ == "__main__":
    settings = Settings()
    neurodatabench.main(
        implementation_id=settings.implementation_id,
        implementation_nwb_interface="pynwb/NWBZarrIO",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=settings.benchmark,
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
