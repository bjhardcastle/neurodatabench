# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "h5py",
#   "numpy",
#   "obstore",
#   "pandas",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "pynwb",
#   "remfile",
#   "s3fs",
# ]
# ///

"""Runnable PyNWB/HDF5 NWBFile implementation for the packaged NWB benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.exists():
    sys.path.insert(0, _REPO_SRC.as_posix())

import h5py
import neurodatabench
import numpy as np
import pynwb
import remfile

logger = logging.getLogger(__name__)

state: dict[str, Any] = {}

_DEFAULT_BACKEND = "remfile"
_DEFAULT_BENCHMARK = "dynamic_routing_hdf5_v0"
_DEFAULT_IMPLEMENTATION_ID = "pynwb_hdf5_nwbfile"
_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128


def setup(context: neurodatabench.RunContext) -> None:
    """Open every benchmark NWB file as a PyNWB NWBFile and store it in state."""
    logger.debug(
        "Opening %d NWB files through PyNWB NWBHDF5IO.",
        len(context.benchmark.data_sources),
    )
    _quiet_storage_debug_loggers()
    state.clear()
    state["files"] = []

    try:
        for nwb_path in context.benchmark.data_sources:
            logger.debug("Opening PyNWB NWBFile for %s.", nwb_path)
            if _backend() == "ros":
                if not h5py.get_config().ros3:
                    raise RuntimeError("This h5py build does not include the ROS3 driver.")
                h5_file = h5py.File(_to_https_url(nwb_path).encode(), mode="r", driver="ros3")
            else:
                file_obj = _open_binary_file(nwb_path)
                h5_file = h5py.File(file_obj, mode="r")
            nwb_io = pynwb.NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)
            state["files"].append({"nwb_file": nwb_io.read()})
    except Exception:
        logger.debug("Clearing partially opened PyNWB files after setup failure.")
        state.clear()
        raise


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear implementation-managed caches before timed benchmark phases."""
    logger.debug(
        "No local PyNWB/remfile disk cache to clear for %d NWB paths.",
        len(context.benchmark.data_sources),
    )


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _count_visp_default_qc(state["files"])
            case "predicated_spike_times":
                answer = _longest_isi_for_fastest_visp_unit(state["files"])
            case "multisession_table_query":
                answer = _multisession_table_query(state["files"])
            case "large_array":
                answer = _large_array(state["files"])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Release PyNWB objects opened during setup."""
    logger.debug(
        "Clearing PyNWB NWBFile state for %d NWB paths.",
        len(context.benchmark.data_sources),
    )
    state.clear()


def _quiet_storage_debug_loggers() -> None:
    """Keep benchmark debug logs focused on implementation-level events."""
    for logger_name in ("aiobotocore", "botocore", "fsspec", "requests", "s3fs", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _backend() -> str:
    """Return the requested PyNWB HDF5 object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def _open_binary_file(nwb_path: str) -> Any:
    """Open one NWB path as a seekable file-like object for h5py."""
    backend = _backend()
    if backend == "remfile":
        return remfile.File(_to_https_url(nwb_path))
    if backend == "s3fs":
        import s3fs

        return s3fs.S3FileSystem(anon=True).open(nwb_path, mode="rb")
    if backend == "ros":
        raise RuntimeError("ROS3 is opened directly as an h5py.File in setup().")
    if backend == "obstore":
        import obstore
        from obstore.store import S3Store

        bucket, key = _split_s3_uri(nwb_path)
        store = S3Store(
            bucket=bucket,
            config={"region": os.environ.get("AWS_REGION", "us-west-2")},
            skip_signature=True,
        )
        return obstore.open_reader(store, key)
    raise ValueError(f"Unsupported PyNWB HDF5 backend: {backend}")


def _to_https_url(nwb_path: str) -> str:
    """Convert a public S3 URI to the HTTPS URL expected by remfile."""
    if nwb_path.startswith("https://") or nwb_path.startswith("http://"):
        return nwb_path
    if not nwb_path.startswith("s3://"):
        raise ValueError(f"Unsupported remote NWB path: {nwb_path}")
    bucket, key = nwb_path.removeprefix("s3://").split("/", maxsplit=1)
    return f"https://{bucket}.s3.amazonaws.com/{quote(key)}"


def _split_s3_uri(nwb_path: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and key components."""
    if not nwb_path.startswith("s3://"):
        raise ValueError(f"Unsupported remote NWB path: {nwb_path}")
    bucket, key = nwb_path.removeprefix("s3://").split("/", maxsplit=1)
    return bucket, key


def _count_visp_default_qc(file_records: list[dict[str, Any]]) -> int:
    """Count VISp units passing default QC across opened PyNWB NWBFiles."""
    count = 0
    for file_record in file_records:
        if "units_frame" not in file_record:
            nwb_file = file_record["nwb_file"]
            if nwb_file.units is None:
                raise ValueError(
                    f"NWBFile {nwb_file.identifier} does not contain units.",
                )
            file_record["units_table"] = nwb_file.units
            file_record["units_frame"] = nwb_file.units.to_dataframe(
                exclude={"spike_times"},
            )
        units = file_record["units_frame"]
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
        if "units_frame" not in file_record:
            if nwb_file.units is None:
                raise ValueError(
                    f"NWBFile {nwb_file.identifier} does not contain units.",
                )
            file_record["units_table"] = nwb_file.units
            file_record["units_frame"] = nwb_file.units.to_dataframe(
                exclude={"spike_times"},
            )
        units = file_record["units_frame"]
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

    top_nwb_file = top_file_record["nwb_file"]
    logger.debug(
        "Fetching spike_times for fastest VISp unit in %s at row %d.",
        top_nwb_file.identifier,
        top_row,
    )
    spike_times = np.asarray(
        top_file_record["units_table"].get_unit_spike_times(top_row),
        dtype=np.float64,
    )
    return float(np.diff(spike_times).max())


def _multisession_table_query(file_records: list[dict[str, Any]]) -> float:
    """Compute the mean trial duration across opened PyNWB NWBFiles."""
    total_duration = 0.0
    total_trials = 0
    for file_record in file_records:
        if "trials_frame" not in file_record:
            nwb_file = file_record["nwb_file"]
            if nwb_file.trials is None:
                raise ValueError(
                    f"NWBFile {nwb_file.identifier} does not contain trials.",
                )
            file_record["trials_table"] = nwb_file.trials
            file_record["trials_frame"] = nwb_file.trials.to_dataframe()
        trials = file_record["trials_frame"]
        start_time = np.asarray(trials["start_time"], dtype=np.float64)
        stop_time = np.asarray(trials["stop_time"], dtype=np.float64)
        total_duration += float(np.sum(stop_time - start_time))
        total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


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


def _large_array(file_records: list[dict[str, Any]]) -> float:
    """Return the mean of a 6.6 MB facemap data block from the first NWBFile."""
    if not file_records:
        raise ValueError("At least one NWBFile record is required.")
    nwb_file = file_records[0]["nwb_file"]
    facemap = nwb_file.processing["behavior"]["facemap_side_camera"]
    data = np.asarray(
        facemap.data[:_FACEMAP_DOWNLOAD_ROWS, :_FACEMAP_DOWNLOAD_COLUMNS],
        dtype=np.float32,
    )
    return float(np.mean(data, dtype=np.float64))


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            f"{_DEFAULT_IMPLEMENTATION_ID}_{_backend()}",
        ),
        implementation_nwb_interface="pynwb",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=None,
        implementation_remote_cache=None,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
