# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "lazynwb[pynwb]==1.0.0dev2",
#   "numpy",
#   "polars==1.38.1",
#   "psutil",
# ]
# ///

"""Runnable lazynwb 1.0.0dev2 implementation for the packaged NWB benchmark."""

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
import numpy as np
import polars as pl

import neurodatabench

logger = logging.getLogger(__name__)

state = {}


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear lazynwb caches and prepare an isolated catalog path before timing."""
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.nwb_paths),
    )
    cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = (cache_dir / "catalog.sqlite").as_posix()


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.nwb_paths))
    cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = (cache_dir / "catalog.sqlite").as_posix()
    os.environ.setdefault("AWS_REGION", "us-west-2")

    lazynwb.config.anon = True

    state["units"] = lazynwb.scan_nwb(
        context.benchmark.nwb_paths,
        "/units",
        disable_progress=True,
    )
    state["trials"] = lazynwb.scan_nwb(
        context.benchmark.nwb_paths,
        "/intervals/trials",
        disable_progress=True,
    )


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "units_VISp_default_qc":
                answer = int(
                    state["units"]
                    .filter((pl.col("structure") == "VISp") & pl.col("default_qc"))
                    .select(pl.len().alias("count"))
                    .collect()
                    .item()
                )
            case "mean_inter_spike_interval":
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
            case "mean_trial_length":
                answer = float(
                    state["trials"]
                    .select(
                        (pl.col("stop_time") - pl.col("start_time")).mean().alias("mean_length")
                    )
                    .collect()
                    .item()
                )
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    pass


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id="lazynwb_v1_dev2",
        implementation_nwb_interface="lazynwb",
        implementation_object_store_backend=None,
        implementation_local_cache="cold",
        implementation_remote_cache=False,
        benchmark="dynamic_routing_hdf5_v0",
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
