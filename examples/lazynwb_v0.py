# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "lazynwb<1.0",
#   "numpy",
#   "psutil",
# ]
# ///

"""Runnable lazynwb 0.x implementation for the packaged NWB benchmark."""

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


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.nwb_paths))
    cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = (cache_dir / "catalog.sqlite").as_posix()
    os.environ.setdefault("AWS_REGION", "us-west-2")

    lazynwb.config.use_remfile = False
    lazynwb.config.use_obstore = False
    lazynwb.config.fsspec_storage_options = {"anon": True}
    lazynwb.config.anon = True


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear lazynwb caches and prepare an isolated catalog path before timing."""
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.nwb_paths),
    )
    cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = (cache_dir / "catalog.sqlite").as_posix()
    lazynwb.clear_cache()
    lazynwb.clear_attrs_cache()


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "units_VISp_default_qc":
                answer = int(
                    lazynwb.scan_nwb(
                        context.benchmark.nwb_paths,
                        "/units",
                        exclude_array_columns=True,
                        disable_progress=True,
                    )
                    .filter((pl.col("structure") == "VISp") & pl.col("default_qc"))
                    .select(pl.len().alias("count"))
                    .collect()
                    .item()
                )
            case "mean_inter_spike_interval":
                visp_units = (
                    lazynwb.scan_nwb(
                        context.benchmark.nwb_paths,
                        "/units",
                        exclude_array_columns=True,
                        disable_progress=True,
                    )
                    .filter(pl.col("structure") == "VISp")
                    .select(["_nwb_path", "_table_index", "firing_rate", "unit_id"])
                    .collect()
                )
                ranked_units = visp_units.filter(
                    pl.col("firing_rate").is_not_null()
                ).sort(
                    "firing_rate",
                    descending=True,
                )
                top_unit = ranked_units.row(0, named=True)
                nwb_path = str(top_unit["_nwb_path"])
                table_index = int(top_unit["_table_index"])
                logger.debug(
                    "Fetching spike_times for %s table index %d.",
                    nwb_path,
                    table_index,
                )
                spike_times_df = lazynwb.tables.get_df(
                    nwb_data_sources=[nwb_path],
                    search_term="/units",
                    exact_path=True,
                    include_column_names=["spike_times"],
                    nwb_path_to_row_indices={nwb_path: [table_index]},
                    exclude_array_columns=False,
                    disable_progress=True,
                    use_process_pool=False,
                    as_polars=True,
                )
                spike_times = np.asarray(
                    spike_times_df.select("spike_times").item(),
                    dtype=np.float64,
                )
                answer = float(np.diff(spike_times).max())
            case "mean_trial_length":
                answer = float(
                    lazynwb.scan_nwb(
                        context.benchmark.nwb_paths,
                        "/intervals/trials",
                        exclude_array_columns=True,
                        disable_progress=True,
                    )
                    .select(
                        (pl.col("stop_time") - pl.col("start_time")).mean().alias(
                            "mean_length"
                        )
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
        implementation_id="lazynwb_v0_obstore",
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark="dynamic_routing_zarr_v0",
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
