# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "hdmf-zarr",
#   "numpy",
#   "pandas",
#   "pynwb",
#   "s3fs",
#   "zarr<3",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///


"""Shared PyNWB NWBZarrIO state and table helpers for the benchmark modules."""

import os
from typing import Any

import numpy as np
import pandas as pd
from hdmf_zarr import NWBZarrIO

import neurodatabench


_DEFAULT_BACKEND = "s3fs"
logger = neurodatabench.get_logger(__name__)

state: dict[str, Any] = {}


def backend() -> str:
    """Return the requested NWBZarrIO object-store backend."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def open_files(data_sources: list[str]) -> None:
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
        close_files()
        raise


def close_files() -> None:
    """Close every NWBZarrIO handle and clear implementation state."""
    for file_record in reversed(state.get("files", [])):
        file_record["nwb_io"].close()
    state.clear()


def files() -> list[dict[str, Any]]:
    """Return the opened file records."""
    return state["files"]


def trials_frame(file_record: dict[str, Any]) -> pd.DataFrame:
    """Return the cached trials DataFrame for one opened NWBFile."""
    if "trials_frame" not in file_record:
        nwb_file = file_record["nwb_file"]
        if nwb_file.trials is None:
            raise ValueError(f"NWBFile {nwb_file.identifier} does not contain trials.")
        file_record["trials_frame"] = nwb_file.trials.to_dataframe()
    return file_record["trials_frame"]


def string_array(values: Any) -> np.ndarray:
    """Convert a table column to a NumPy string array."""
    raw_values = np.asarray(values)
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_values
        ],
        dtype=str,
    )
