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


"""lazynwb answers for the dynamic routing benchmark questions."""

import logging

import lazynwb
import numpy as np
import polars as pl

import neurodatabench
import helpers


_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128


logger = logging.getLogger(__name__)


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _count_visp_default_qc(_units(context.benchmark.data_sources))
            case "predicated_spike_times":
                answer = _longest_isi_for_fastest_visp_unit(_units(context.benchmark.data_sources))
            case "multisession_table_query":
                answer = _multisession_table_query(helpers.trials())
            case "large_array":
                answer = _large_array(context.benchmark.data_sources[0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _units(data_sources: list[str]) -> pl.LazyFrame:
    """Return the cached multi-session units LazyFrame."""
    if "units" not in helpers.state:
        helpers.state["units"] = lazynwb.scan_nwb(
            data_sources,
            "/units",
            disable_progress=True,
            infer_schema_length=1,
        )
    return helpers.state["units"]


def _count_visp_default_qc(units: pl.LazyFrame) -> int:
    """Count VISp units passing default QC across all sessions."""
    return int(
        units
        .filter(
            pl.col("structure").eq("VISp"),
            pl.col("default_qc"),
        )
        .select(pl.len().alias("count"))
        .collect()
        .item()
    )


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


def _multisession_table_query(trials: pl.LazyFrame) -> float:
    """Compute the mean trial duration across all sessions."""
    return float(
        trials
        .select(
            (pl.col("stop_time") - pl.col("start_time")).mean().alias("mean_length")
        )
        .collect()
        .item()
    )


def _large_array(nwb_path: str) -> float:
    """Return the mean of a 6.6 MB facemap data block from the first session."""
    facemap = lazynwb.get_timeseries(
        nwb_path,
        "/processing/behavior/facemap_side_camera",
        exact_path=True,
    )
    data = np.asarray(
        facemap.data[:_FACEMAP_DOWNLOAD_ROWS, :_FACEMAP_DOWNLOAD_COLUMNS],
        dtype=np.float32,
    )
    return float(np.mean(data, dtype=np.float64))
