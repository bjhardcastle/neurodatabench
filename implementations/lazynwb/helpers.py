"""Shared lazynwb configuration, cache, and state helpers for the benchmark modules."""

import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any

import lazynwb
import polars as pl

import neurodatabench


_DEFAULT_BACKEND = "obstore"
logger = neurodatabench.get_logger(__name__)

state: dict[str, Any] = {}


def backend() -> str:
    """Return the requested lazynwb object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def configure_backend(backend: str) -> None:
    """Configure backend switches exposed by the installed lazynwb version."""
    version = lazynwb_version()
    logger.debug("Configuring lazynwb %s backend %s.", version, backend)
    if version.split(".", maxsplit=1)[0] == "0":
        if backend not in {"obstore", "remfile", "s3fs"}:
            raise ValueError(f"Unsupported lazynwb pre-1.0 backend: {backend}")
        lazynwb.config.use_obstore = backend == "obstore"
        lazynwb.config.use_remfile = backend == "remfile"
        lazynwb.config.fsspec_storage_options = {"anon": True}
    elif backend != "obstore":
        raise ValueError(f"lazynwb {version} uses obstore; got backend {backend!r}.")


def lazynwb_version() -> str:
    """Return the installed lazynwb distribution version."""
    return importlib.metadata.version("lazynwb")


def set_catalog_cache_path() -> Path:
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


def local_cache() -> neurodatabench.models.LocalCacheState:
    """Return local cache metadata declared for this run."""
    value = os.environ.get("NDB_LOCAL_CACHE", "cold")
    if value not in {"cold", "warm"}:
        raise ValueError("NDB_LOCAL_CACHE must be 'cold' or 'warm' for lazynwb.")
    return value  # type: ignore[return-value]


def default_implementation_id() -> str:
    """Return an ID containing the installed lazynwb version and backend."""
    version = lazynwb_version().replace(".", "_").replace("+", "_")
    return f"lazynwb_{version}_{backend()}"


def trials() -> pl.LazyFrame:
    """Return the multi-session trials LazyFrame scanned during setup."""
    return state["trials"]
