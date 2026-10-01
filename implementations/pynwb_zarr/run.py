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


"""Run a true PyNWB NWBZarrIO materialization benchmark against remote Zarr."""

from __future__ import annotations

import os
from collections.abc import Callable

import neurodatabench
import helpers

logger = neurodatabench.get_logger(__name__)


_DEFAULT_BENCHMARK = "dynamic_routing_nwb_zarr_v0"
_DEFAULT_IMPLEMENTATION_ID = "pynwb_hdmf_zarr_direct"


def setup(context: neurodatabench.RunContext) -> None:
    """Materialize each remote Zarr store as a PyNWB NWBFile."""
    backend = helpers.backend()
    if backend != "s3fs":
        raise ValueError(f"NWBZarrIO does not support configured backend {backend!r}.")
    logger.debug(
        "Opening %d Zarr stores through NWBZarrIO and s3fs.",
        len(context.benchmark.data_sources),
    )
    helpers.open_files(context.benchmark.data_sources)


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Declare that this implementation has no managed local cache."""
    logger.debug("No PyNWB Zarr cache to clear for %d paths.", len(context.benchmark.data_sources))


def submit_answers(benchmark_id: str) -> Callable:
    """Return the submit_answers function for this benchmark."""
    import change_detection, dynamic_routing, vr_foraging

    if benchmark_id.startswith("change_detection"):
        return change_detection.submit_answers
    elif benchmark_id.startswith("vr_foraging"):
        return vr_foraging.submit_answers
    return dynamic_routing.submit_answers


def teardown(context: neurodatabench.RunContext) -> None:
    """Close NWBZarrIO handles and clear materialized state."""
    logger.debug("Closing NWBZarrIO handles for %d paths.", len(context.benchmark.data_sources))
    helpers.close_files()


if __name__ == "__main__":
    benchmark = os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK)
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            f"{_DEFAULT_IMPLEMENTATION_ID}_{helpers.backend()}",
        ),
        implementation_nwb_interface="pynwb/NWBZarrIO",
        implementation_object_store_backend=helpers.backend(),
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=benchmark,
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers(benchmark),
        teardown=teardown,
    )
