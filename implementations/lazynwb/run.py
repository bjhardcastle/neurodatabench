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

"""Version-agnostic lazynwb implementation for packaged NWB benchmarks."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import lazynwb

import neurodatabench
import helpers

logger = neurodatabench.get_logger(__name__)


_DEFAULT_BENCHMARK = "dynamic_routing_nwb_hdf5_v0"


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Remove the selected lazynwb catalog before a cold run."""
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.data_sources),
    )
    cache_path = helpers.set_catalog_cache_path()
    for path in (cache_path, Path(f"{cache_path}-shm"), Path(f"{cache_path}-wal")):
        path.unlink(missing_ok=True)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.data_sources))
    helpers.set_catalog_cache_path()
    os.environ.setdefault("AWS_REGION", "us-west-2")

    lazynwb.config.anon = True
    helpers.configure_backend(helpers.backend())
    helpers.state.clear()

    helpers.state["trials"] = lazynwb.scan_nwb(
        context.benchmark.data_sources,
        "/intervals/trials",
        disable_progress=True,
    )


def submit_answers(benchmark_id: str) -> Callable:
    """Return the submit_answers function for this benchmark."""
    import change_detection, dynamic_routing, vr_foraging

    if benchmark_id.startswith("change_detection"):
        return change_detection.submit_answers
    elif benchmark_id.startswith("vr_foraging"):
        return vr_foraging.submit_answers
    return dynamic_routing.submit_answers


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug("Clearing lazynwb state for %d NWB paths.", len(context.benchmark.data_sources))
    helpers.state.clear()


if __name__ == "__main__":
    benchmark = os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK)
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            helpers.default_implementation_id(),
        ),
        implementation_nwb_interface="lazynwb",
        implementation_object_store_backend=helpers.backend(),
        implementation_local_cache=helpers.local_cache(),
        implementation_remote_cache=False,
        benchmark=benchmark,
        setup=setup,
        clear_cache=None if helpers.local_cache() == "warm" else clear_cache,
        submit_answers=submit_answers(benchmark),
        teardown=teardown,
    )
