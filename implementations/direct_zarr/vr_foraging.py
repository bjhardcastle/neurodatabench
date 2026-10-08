# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "numpy",
#   "obstore",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "remfile",
#   "s3fs",
#   "zarr",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


"""Direct Zarr implementation for the VR foraging benchmark questions."""

from __future__ import annotations

import logging
import numpy as np
import os
import zarr
from typing import Any, Iterator, MutableMapping

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_DEFAULT_BENCHMARK = "vr_foraging_nwb_zarr_v0"
_DEFAULT_IMPLEMENTATION_ID = "direct_zarr"
_DEFAULT_BACKEND = "s3fs"


#region VR Foraging solutions
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
        trials = _open_store(nwb_path)["intervals"]["trials"]
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
        trials = _open_store(nwb_path)["intervals"]["trials"]
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

    store = _open_store(first_session_path)
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
        store = _open_store(nwb_path)

        threshold = float(store[stop_velocity_path]["data"][0])
        velocity = np.asarray(
            store["processing"]["behavior"]["position_velocity"]["velocity"][:],
            dtype=np.float64,
        )

        session_below = int(np.count_nonzero(velocity < threshold))
        total_below += session_below
        total_samples += velocity.size

    return total_below / total_samples
#endregion


#region Runtime Configuration
def _backend() -> str:
    """Return the requested direct Zarr object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure process-level settings before answering benchmark questions."""
    logger.debug(
        "Preparing direct Zarr/%s access for %d NWB stores.",
        _backend(),
        len(context.benchmark.data_sources),
    )
    os.environ.setdefault("AWS_REGION", "us-west-2")
    _quiet_storage_debug_loggers()


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear implementation-managed caches before timed benchmark phases."""
    logger.debug(
        "No local direct Zarr cache to clear for %d NWB paths.",
        len(context.benchmark.data_sources),
    )


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug(
        "Direct Zarr benchmark teardown for %d NWB paths.",
        len(context.benchmark.data_sources),
    )


def _quiet_storage_debug_loggers() -> None:
    """Keep benchmark debug logs focused on implementation-level events."""
    for logger_name in ("aiobotocore", "botocore", "fsspec", "s3fs", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
#endregion


#region Zarr store access
def _open_store(nwb_path: str) -> Any:
    """Open one remote NWB Zarr store as a read-only Zarr group."""
    backend = _backend()
    logger.debug("Opening NWB Zarr store %s through %s.", nwb_path, backend)
    if backend == "s3fs":
        if _is_zarr_v3():
            return zarr.open_group(
                nwb_path,
                mode="r",
                storage_options={"anon": True},
                use_consolidated=False,
            )
        return zarr.open(nwb_path, mode="r", storage_options={"anon": True})
    if backend == "obstore":
        from obstore import fsspec as obstore_fsspec

        os.environ.setdefault("AWS_SKIP_SIGNATURE", "true")
        if _is_zarr_v3():
            obstore_fsspec.register("s3")
            return zarr.open_group(nwb_path, mode="r", use_consolidated=False)
        bucket, key = _split_s3_uri(nwb_path)
        fs = obstore_fsspec.FsspecStore("s3", config={"skip_signature": True})
        return zarr.open(_ObstoreZarrV2Store(fs, f"{bucket}/{key}"), mode="r")
    if backend in {"remfile", "ros"}:
        raise RuntimeError(f"{backend} is a file backend and cannot open directory Zarr stores.")
    raise ValueError(f"Unsupported direct Zarr backend: {backend}")


def _is_zarr_v3() -> bool:
    """Return whether the active zarr-python runtime is major version 3."""
    return zarr.__version__.split(".", maxsplit=1)[0] == "3"


def _split_s3_uri(nwb_path: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and key components."""
    if not nwb_path.startswith("s3://"):
        raise ValueError(f"Unsupported remote NWB path: {nwb_path}")
    bucket, key = nwb_path.removeprefix("s3://").split("/", maxsplit=1)
    return bucket, key


class _ObstoreZarrV2Store(MutableMapping[str, bytes]):
    """Read-only Zarr v2 mapping backed by obstore's fsspec adapter."""

    def __init__(self, fs: Any, root_path: str) -> None:
        """Create a store rooted at a bucket-relative object prefix."""
        self._fs = fs
        self._root_path = root_path.rstrip("/")

    def __getitem__(self, key: str) -> bytes:
        """Return one Zarr metadata or chunk object."""
        try:
            return bytes(self._fs.cat_file(self._path_for(key)))
        except Exception as exc:
            if isinstance(exc, self._missing_exceptions()):
                raise KeyError(key) from exc
            raise

    def __setitem__(self, key: str, value: bytes) -> None:
        """Reject writes because benchmark stores are read-only."""
        raise TypeError(f"{type(self).__name__} is read-only")

    def __delitem__(self, key: str) -> None:
        """Reject deletes because benchmark stores are read-only."""
        raise TypeError(f"{type(self).__name__} is read-only")

    def __iter__(self) -> Iterator[str]:
        """Yield Zarr object keys relative to the store root."""
        prefix = f"{self._root_path}/"
        for path in self._fs.find(self._root_path):
            yield path.removeprefix(prefix)

    def __len__(self) -> int:
        """Return the number of objects below the store root."""
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        """Return whether a Zarr key exists in the store."""
        if not isinstance(key, str):
            return False
        try:
            return bool(self._fs.exists(self._path_for(key)))
        except Exception:
            return False

    def _path_for(self, key: str) -> str:
        """Return the bucket-relative object path for a Zarr key."""
        stripped = key.lstrip("/")
        if not stripped:
            return self._root_path
        return f"{self._root_path}/{stripped}"

    def _missing_exceptions(self) -> tuple[type[BaseException], ...]:
        """Return filesystem exceptions that should behave like missing keys."""
        missing = getattr(self._fs, "missing_exceptions", (FileNotFoundError,))
        if isinstance(missing, tuple):
            return missing
        return tuple(missing)
#endregion


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            f"{_DEFAULT_IMPLEMENTATION_ID}_{_backend()}_zarr{zarr.__version__.split('.')[0]}",
        ),
        implementation_nwb_interface=None,
        implementation_object_store_backend=_backend(),
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )