# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "lazynwb==0.2.91",
#   "numpy",
#   "polars==1.38.1",
#   "psutil",
# ]
# ///

"""Runnable lazynwb pre-1.0 implementation for the packaged NWB benchmark."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.exists():
    sys.path.insert(0, _REPO_SRC.as_posix())

import lazynwb
import lazynwb.tables
import lazynwb.timeseries
import neurodatabench
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

state = {}

_DEFAULT_BACKEND = "s3fs"
_DEFAULT_BENCHMARK = "dynamic_routing_hdf5_v0"
_DEFAULT_IMPLEMENTATION_ID = "lazynwb_v0"
_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear lazynwb caches and prepare an isolated catalog path before timing."""
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.data_sources),
    )
    _set_catalog_cache_path()


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.data_sources))
    _set_catalog_cache_path()
    os.environ.setdefault("AWS_REGION", "us-west-2")

    lazynwb.config.anon = True
    _configure_backend(_backend())
    state.clear()

    state["units"] = lazynwb.scan_nwb(
        context.benchmark.data_sources,
        "/units",
        disable_progress=True,
    )
    state["trials"] = lazynwb.scan_nwb(
        context.benchmark.data_sources,
        "/intervals/trials",
        disable_progress=True,
    )
    state["facemap_side_camera"] = lazynwb.timeseries.get_timeseries(
        context.benchmark.data_sources[0],
        "/processing/behavior/facemap_side_camera",
        exact_path=True,
    )


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = int(
                    state["units"]
                    .filter((pl.col("structure") == "VISp") & pl.col("default_qc"))
                    .select(pl.len().alias("count"))
                    .collect()
                    .item()
                )
            case "predicated_spike_times":
                visp_units = (
                    state["units"]
                    .filter(
                        pl.col("structure") == "VISp",
                        pl.col("firing_rate").is_not_null(),
                    )
                    .sort("firing_rate", descending=True)
                    .head(1)
                    .select("spike_times")
                    .collect()
                )
                spike_times = visp_units["spike_times"][0]
                answer = float(np.diff(spike_times).max())
            case "multisession_table_query":
                answer = float(
                    state["trials"]
                    .select(
                        (pl.col("stop_time") - pl.col("start_time")).mean().alias("mean_length")
                    )
                    .collect()
                    .item()
                )
            case "large_array":
                data = np.asarray(
                    state["facemap_side_camera"].data[
                        :_FACEMAP_DOWNLOAD_ROWS,
                        :_FACEMAP_DOWNLOAD_COLUMNS,
                    ],
                    dtype=np.float32,
                )
                answer = float(
                    np.mean(data, dtype=np.float64),
                )
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    pass


def _backend() -> str:
    """Return the requested lazynwb object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def _configure_backend(backend: str) -> None:
    """Enable the requested lazynwb backend by disabling competing backends."""
    logger.debug("Configuring lazynwb pre-1.0 backend %s.", backend)
    if hasattr(lazynwb.config, "use_obstore"):
        lazynwb.config.use_obstore = backend == "obstore"
    if hasattr(lazynwb.config, "use_remfile"):
        lazynwb.config.use_remfile = backend == "remfile"
    if hasattr(lazynwb.config, "fsspec_storage_options"):
        lazynwb.config.fsspec_storage_options = {"anon": True}


def _set_catalog_cache_path() -> None:
    """Point lazynwb at a matrix-provided cache or a fresh isolated cache."""
    cache_path = os.environ.get("NDB_LAZYNWB_CACHE_PATH")
    if cache_path is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
        cache_path = (cache_dir / "catalog.sqlite").as_posix()
    else:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = cache_path


def _local_cache() -> neurodatabench.models.LocalCacheState:
    """Return local cache metadata declared for this run."""
    value = os.environ.get("NDB_LOCAL_CACHE", "cold")
    if value not in {"cold", "warm"}:
        raise ValueError("NDB_LOCAL_CACHE must be 'cold' or 'warm' for lazynwb.")
    return value  # type: ignore[return-value]


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            f"{_DEFAULT_IMPLEMENTATION_ID}_{_backend()}",
        ),
        implementation_nwb_interface="lazynwb",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=_local_cache(),
        implementation_remote_cache=False,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
