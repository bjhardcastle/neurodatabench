# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "arro3-core",
#   "icechunk",
#   "numpy",
#   "obstore",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "s3fs",
#   "virtualizarr",
#   "zarr",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///

"""Run the VirtualiZarr reader entry point for the remote ROI benchmark."""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Any


def main() -> None:
    """Delegate to the shared ROI reader with VirtualiZarr selected."""
    os.environ.setdefault("NDB_ZARR_METHOD", "virtualizarr")
    implementation = Path(__file__).with_name("roi_zarr_template.py")
    namespace: dict[str, Any] = runpy.run_path(str(implementation))
    namespace["main"]()


if __name__ == "__main__":
    main()
