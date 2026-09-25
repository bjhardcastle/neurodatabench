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

import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any

import lazynwb
import numpy as np
import polars as pl

import neurodatabench

logger = neurodatabench.get_logger(__name__)

state: dict[str, Any] = {}

_DEFAULT_BACKEND = "obstore"
_DEFAULT_BENCHMARK = "dynamic_routing_nwb_hdf5_v0"
_ROI_BENCHMARK = "multiplane_ophys_roi_zarr_v0"
_ROI_PLANES = tuple(f"VISp_{index}" for index in range(8))
_ROI_TABLE_SUFFIX = "image_segmentation/roi_table"
_ROI_EXPECTED_TABLE_COUNT = 288
_ROI_EXPECTED_ROI_COUNT = 67_400
_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Remove the selected lazynwb catalog before a cold run."""
    if _local_cache() != "cold":
        logger.debug("Keeping lazynwb catalog for warm run.")
        return
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.data_sources),
    )
    cache_path = _set_catalog_cache_path()
    for path in (cache_path, Path(f"{cache_path}-shm"), Path(f"{cache_path}-wal")):
        path.unlink(missing_ok=True)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.data_sources))
    _set_catalog_cache_path()
    os.environ.setdefault("AWS_REGION", "us-west-2")

    lazynwb.config.anon = True
    _configure_backend(_backend())
    state.clear()

    if context.benchmark.id == _ROI_BENCHMARK:
        state["roi_tables"] = _scan_roi_tables(context)
        return

    state["units"] = lazynwb.scan_nwb(
        context.benchmark.data_sources,
        "/units",
        disable_progress=True,
        infer_schema_length=1,
    )
    state["trials"] = lazynwb.scan_nwb(
        context.benchmark.data_sources,
        "/intervals/trials",
        disable_progress=True,
    )
    state["facemap_side_camera"] = lazynwb.get_timeseries(
        context.benchmark.data_sources[0],
        "/processing/behavior/facemap_side_camera",
        exact_path=True,
    )


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "roi_table_read_summary":
                answer = _read_roi_tables(context)
            case "multisession_units_metadata_query":
                answer = int(
                    state["units"]
                    .filter(
                        pl.col("structure").eq("VISp"),
                        pl.col("default_qc"),
                    )
                    .select(pl.len().alias("count"))
                    .collect()
                    .item()
                )
            case "predicated_spike_times":
                answer = _longest_isi_for_fastest_visp_unit(state["units"])
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
    logger.debug("Clearing lazynwb state for %d NWB paths.", len(context.benchmark.data_sources))
    state.clear()


def _longest_isi_for_fastest_visp_unit(units: pl.LazyFrame) -> float:
    """Return the longest ISI while reading spikes only for the selected unit."""
    candidates = (
        units.filter(
            pl.col("structure").eq("VISp"),
            pl.col("firing_rate").is_not_null(),
        )
        .select(
            lazynwb.NWB_PATH_COLUMN_NAME,
            lazynwb.TABLE_INDEX_COLUMN_NAME,
            "firing_rate",
        )
        .collect()
    )
    if candidates.is_empty():
        raise ValueError("No VISp unit with a firing_rate was found.")

    fastest_unit = candidates.sort("firing_rate", descending=True).head(1)
    nwb_path = str(fastest_unit[lazynwb.NWB_PATH_COLUMN_NAME].item())
    table_index = int(fastest_unit[lazynwb.TABLE_INDEX_COLUMN_NAME].item())
    firing_rate = float(fastest_unit["firing_rate"].item())
    logger.debug(
        "Fetching spike_times for fastest VISp unit in %s at row %d "
        "(firing_rate=%s).",
        nwb_path,
        table_index,
        firing_rate,
    )

    selected_unit = (
        units.filter(
            pl.col(lazynwb.NWB_PATH_COLUMN_NAME).eq(nwb_path),
            pl.col(lazynwb.TABLE_INDEX_COLUMN_NAME).eq(table_index),
        )
        .select("spike_times")
        .collect()
    )
    if selected_unit.height != 1:
        raise ValueError(
            "Expected one unit at "
            f"{nwb_path!r} row {table_index}, found {selected_unit.height}."
        )
    spike_times = np.asarray(selected_unit["spike_times"].item(), dtype=np.float64)
    return float(np.diff(spike_times).max())


def _scan_roi_tables(
    context: neurodatabench.RunContext,
) -> list[tuple[str, str, pl.LazyFrame]]:
    """Create one lazy frame per session and imaging plane."""
    tables: list[tuple[str, str, pl.LazyFrame]] = []
    for data_source in context.benchmark.data_sources:
        for plane in _ROI_PLANES:
            table_path = f"/processing/{plane}/{_ROI_TABLE_SUFFIX}"
            logger.debug("Planning lazyNWB scan for %s at %s.", data_source, table_path)
            tables.append(
                (
                    data_source,
                    plane,
                    lazynwb.scan_nwb(
                        [data_source],
                        table_path,
                        disable_progress=True,
                    ),
                )
            )
    return tables


def _read_roi_tables(context: neurodatabench.RunContext) -> neurodatabench.JsonObject:
    """Collect every ROI-table column and summarize the resulting rows."""
    tables: list[tuple[str, str, pl.LazyFrame]] = state["roi_tables"]
    first_table_image_mask_mean: float | None = None
    roi_count = 0
    for index, (_, _, lazy_table) in enumerate(tables):
        logger.debug("Collecting ROI table %d/%d.", index + 1, len(tables))
        frame = lazy_table.collect()
        roi_count += frame.height
        if index == 0:
            if frame.is_empty():
                raise RuntimeError("The first ROI table did not produce any rows.")
            image_masks = np.stack(frame["image_mask"].to_list())
            first_table_image_mask_mean = float(
                np.mean(image_masks, dtype=np.float64)
            )

    table_count = len(tables)
    if table_count != _ROI_EXPECTED_TABLE_COUNT:
        raise RuntimeError(
            f"Expected {_ROI_EXPECTED_TABLE_COUNT} ROI tables, found {table_count}."
        )
    if roi_count != _ROI_EXPECTED_ROI_COUNT:
        raise RuntimeError(f"Expected {_ROI_EXPECTED_ROI_COUNT} ROIs, found {roi_count}.")

    if first_table_image_mask_mean is None:
        raise RuntimeError("No ROI tables were collected.")
    return {
        "session_count": len(context.benchmark.data_sources),
        "table_count": table_count,
        "roi_count": roi_count,
        "first_table_image_mask_mean": first_table_image_mask_mean,
    }


def _backend() -> str:
    """Return the requested lazynwb object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


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
    elif backend == "obstore":
        lazynwb.config.use_obstore = True
        lazynwb.config.use_remfile = False
    elif backend == "s3fs":
        lazynwb.config.use_obstore = False
        lazynwb.config.use_remfile = False
        lazynwb.config.fsspec_storage_options = {"anon": True}
    else:
        raise ValueError(f"Unsupported lazynwb {version} backend: {backend!r}.")


def _lazynwb_version() -> str:
    """Return the installed lazynwb distribution version."""
    return importlib.metadata.version("lazynwb")


def _set_catalog_cache_path() -> Path:
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


def _local_cache() -> neurodatabench.models.LocalCacheState:
    """Return local cache metadata declared for this run."""
    value = os.environ.get("NDB_LOCAL_CACHE", "cold")
    if value not in {"cold", "warm"}:
        raise ValueError("NDB_LOCAL_CACHE must be 'cold' or 'warm' for lazynwb.")
    return value  # type: ignore[return-value]


def _default_implementation_id() -> str:
    """Return an ID containing the installed lazynwb version and backend."""
    version = _lazynwb_version().replace(".", "_").replace("+", "_")
    return f"lazynwb_{version}_{_backend()}"


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            _default_implementation_id(),
        ),
        implementation_nwb_interface="lazynwb",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=_local_cache(),
        implementation_remote_cache=False,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=None if _local_cache() == "warm" else clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
