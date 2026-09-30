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
# neurodatabench = { path = "../..", editable = true }
# ///

"""Runnable direct Zarr implementation for the packaged NWB benchmark."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
import zarr

import neurodatabench
import helpers

logger = neurodatabench.get_logger(__name__)


_DEFAULT_BENCHMARK = "dynamic_routing_nwb_zarr_v0"
_DEFAULT_IMPLEMENTATION_ID = "direct_zarr"


def setup(context: neurodatabench.RunContext) -> None:
    """Configure process-level settings before answering benchmark questions."""
    logger.debug(
        "Preparing direct Zarr/%s access for %d NWB stores.",
        helpers.backend(),
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
    logger.debug(
        "Direct Zarr benchmark teardown for %d NWB paths.",
        len(context.benchmark.data_sources),
    )


def _quiet_storage_debug_loggers() -> None:
    """Keep benchmark debug logs focused on implementation-level events."""
    for logger_name in ("aiobotocore", "botocore", "fsspec", "s3fs", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


if __name__ == "__main__":
    benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK)
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            f"{_DEFAULT_IMPLEMENTATION_ID}_{helpers.backend()}_zarr{zarr.__version__.split('.')[0]}",
        ),
        implementation_nwb_interface=None,
        implementation_object_store_backend=helpers.backend(),
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=benchmark,
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers(benchmark),
        teardown=teardown,
    )
