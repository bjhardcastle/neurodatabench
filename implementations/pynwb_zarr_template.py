# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "hdmf-zarr",
#   "h5py",
#   "numpy",
#   "pandas",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "pynwb",
#   "remfile",
#   "s3fs",
#   "zarr<3",
# ]
# ///

"""Run a true PyNWB NWBZarrIO materialization benchmark against remote Zarr."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, (_REPO_ROOT / "src").as_posix())
sys.path.insert(0, _REPO_ROOT.as_posix())

import neurodatabench
from hdmf_zarr import NWBZarrIO

import implementations.pynwb_hdf5_template as answer_helpers

logger = logging.getLogger(__name__)
state: dict[str, Any] = {}


def setup(context: neurodatabench.RunContext) -> None:
    """Materialize each remote Zarr store as a PyNWB NWBFile."""
    logger.debug("Opening %d Zarr stores through NWBZarrIO.", len(context.benchmark.nwb_paths))
    state.clear()
    state["files"] = []
    for nwb_path in context.benchmark.nwb_paths:
        nwb_io = NWBZarrIO(
            path=nwb_path,
            mode="r",
            load_namespaces=True,
            storage_options={"anon": True},
        )
        state["files"].append({"nwb_io": nwb_io, "nwb_file": nwb_io.read()})


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Declare that this implementation has no managed local cache."""
    logger.debug("No PyNWB Zarr cache to clear for %d paths.", len(context.benchmark.nwb_paths))


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Answer questions through the materialized PyNWB NWBFile objects."""
    files = state["files"]
    for question in context.benchmark.questions:
        match question.id:
            case "multisession_units_metadata_query":
                answer = answer_helpers._count_visp_default_qc(files)
            case "predicated_spike_times":
                answer = answer_helpers._longest_isi_for_fastest_visp_unit(files)
            case "multisession_table_query":
                answer = answer_helpers._multisession_table_query(files)
            case "large_array":
                answer = answer_helpers._large_array(files)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Close NWBZarrIO handles and clear materialized state."""
    for file_record in state.get("files", []):
        file_record["nwb_io"].close()
    state.clear()


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id="pynwb_nwbzarrio_s3fs",
        implementation_nwb_interface="pynwb/NWBZarrIO",
        implementation_object_store_backend="s3fs",
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark="dynamic_routing_zarr_v0",
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
