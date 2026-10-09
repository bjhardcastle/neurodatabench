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


"""PyNWB NWBZarrIO answers for the dynamic routing benchmark questions."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import pydantic_settings
from hdmf_zarr import NWBZarrIO

import neurodatabench

logger = neurodatabench.get_logger(__name__)


_SOURCE_TYPE = "zarr"
_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128


class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the PyNWB Zarr runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True, cli_kebab_case=True, cli_parse_args=True, env_prefix="NDB_"
    )

    benchmark: Literal["dynamic_routing_nwb_v0"] = "dynamic_routing_nwb_v0"
    object_store_backend: Literal["s3fs", "obstore"] = "s3fs"
    implementation_id: str = "pynwb"


settings: Settings | None = None


def _settings() -> Settings:
    """Return the resolved implementation settings."""
    if settings is None:
        raise RuntimeError("Settings must be initialized before running the implementation.")
    return settings


state: dict[str, Any] = {}


#region Dynamic Routing solutions
def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    files = state["files"]
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _count_visp_default_qc(files)
            case "predicated_spike_times":
                answer = _longest_isi_for_fastest_visp_unit(files)
            case "multisession_table_query":
                answer = _multisession_table_query(files)
            case "large_array":
                answer = _large_array(files)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _unit_metrics(file_record: dict[str, Any]) -> pd.DataFrame:
    """Return the cached units DataFrame without heavy ragged columns."""
    if "unit_metrics" not in file_record:
        nwb_file = file_record["nwb_file"]
        if nwb_file.units is None:
            raise ValueError(f"NWBFile {nwb_file.identifier} does not contain units.")
        file_record["units"] = nwb_file.units
        file_record["unit_metrics"] = nwb_file.units.to_dataframe(
            exclude={
                "spike_times",
                "spike_amplitudes",
                "waveform_mean",
                "waveform_std",
            },
        )
    return file_record["unit_metrics"]


def _count_visp_default_qc(file_records: list[dict[str, Any]]) -> int:
    """Count VISp units passing default QC across opened PyNWB NWBFiles."""
    count = 0
    for file_record in file_records:
        if "unit_metrics" not in file_record:
            nwb_file = file_record["nwb_file"]
            if nwb_file.units is None:
                raise ValueError(f"NWBFile {nwb_file.identifier} does not contain units.")
            file_record["units"] = nwb_file.units
            file_record["unit_metrics"] = nwb_file.units.to_dataframe(
                exclude={
                    "spike_times",
                    "spike_amplitudes",
                    "waveform_mean",
                    "waveform_std",
                },
            )
        units = file_record["unit_metrics"]
        structure = _string_array(units["structure"])
        default_qc = np.asarray(units["default_qc"], dtype=np.bool_)
        count += int(np.count_nonzero((structure == "VISp") & default_qc))
    return count


def _longest_isi_for_fastest_visp_unit(file_records: list[dict[str, Any]]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_file_record: dict[str, Any] | None = None
    top_row = -1
    top_firing_rate = -np.inf
    for file_record in file_records:
        nwb_file = file_record["nwb_file"]
        if "unit_metrics" not in file_record:
            if nwb_file.units is None:
                raise ValueError(f"NWBFile {nwb_file.identifier} does not contain units.")
            file_record["units"] = nwb_file.units
            file_record["unit_metrics"] = nwb_file.units.to_dataframe(
                exclude={
                    "spike_times",
                    "spike_amplitudes",
                    "waveform_mean",
                    "waveform_std",
                },
            )
        units = file_record["unit_metrics"]
        structure = _string_array(units["structure"])
        firing_rate = np.asarray(units["firing_rate"], dtype=np.float64)
        candidate_rows = np.flatnonzero((structure == "VISp") & ~np.isnan(firing_rate))
        if candidate_rows.size == 0:
            logger.debug("No VISp units with firing_rate in %s.", nwb_file.identifier)
            continue
        local_row = int(candidate_rows[np.argmax(firing_rate[candidate_rows])])
        local_rate = float(firing_rate[local_row])
        if local_rate > top_firing_rate:
            top_file_record = file_record
            top_row = local_row
            top_firing_rate = local_rate
    if top_file_record is None:
        raise ValueError("No VISp unit with a finite firing_rate was found.")
    spike_times = np.asarray(
        top_file_record["units"].get_unit_spike_times(top_row),
        dtype=np.float64,
    )
    return float(np.diff(spike_times).max())



def _multisession_table_query(file_records: list[dict[str, Any]]) -> float:
    """Compute the mean trial duration across opened PyNWB NWBFiles."""
    total_duration = 0.0
    total_trials = 0
    for file_record in file_records:
        trials = _get_trials_table(file_record)
        start_time = np.asarray(trials["start_time"], dtype=np.float64)
        stop_time = np.asarray(trials["stop_time"], dtype=np.float64)
        total_duration += float(np.sum(stop_time - start_time))
        total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


def _large_array(file_records: list[dict[str, Any]]) -> float:
    """Return the mean of a 6.6 MB facemap block from the first NWBFile."""
    if not file_records:
        raise ValueError("At least one NWBFile record is required.")
    nwb_file = file_records[0]["nwb_file"]
    facemap = nwb_file.processing["behavior"]["facemap_side_camera"]
    data = np.asarray(
        facemap.data[:_FACEMAP_DOWNLOAD_ROWS, :_FACEMAP_DOWNLOAD_COLUMNS],
        dtype=np.float32,
    )
    return float(np.mean(data, dtype=np.float64))


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
