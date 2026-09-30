import zarr
import os
import neurodatabench
import numpy as np

from typing import Any, Iterator, MutableMapping


_DEFAULT_BACKEND = "s3fs"
logger = neurodatabench.get_logger(__name__)


def open_store(nwb_path: str) -> Any:
    """Open one remote NWB Zarr store as a read-only Zarr group."""
    backend = os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)
    logger.debug("Opening NWB Zarr store %s through %s.", nwb_path, backend)
    if backend == "s3fs":
        if _is_zarr_v3():
            return zarr.open_group(
                nwb_path,
                mode="r",
                storage_options={"anon": True},
                use_consolidated=False,
            )
        return zarr.open(nwb_path, mode="r", storage_options={"anon": True})
    if backend == "obstore":
        from obstore import fsspec as obstore_fsspec

        os.environ.setdefault("AWS_SKIP_SIGNATURE", "true")
        if _is_zarr_v3():
            obstore_fsspec.register("s3")
            return zarr.open_group(nwb_path, mode="r", use_consolidated=False)
        bucket, key = _split_s3_uri(nwb_path)
        fs = obstore_fsspec.FsspecStore("s3", config={"skip_signature": True})
        return zarr.open(_ObstoreZarrV2Store(fs, f"{bucket}/{key}"), mode="r")
    if backend in {"remfile", "ros"}:
        raise RuntimeError(f"{backend} is a file backend and cannot open directory Zarr stores.")
    raise ValueError(f"Unsupported direct Zarr backend: {backend}")


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


def backend() -> str:
    """Return the requested direct Zarr object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def _is_zarr_v3() -> bool:
    """Return whether the active zarr-python runtime is major version 3."""
    return zarr.__version__.split(".", maxsplit=1)[0] == "3"


def _split_s3_uri(nwb_path: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and key components."""
    if not nwb_path.startswith("s3://"):
        raise ValueError(f"Unsupported remote NWB path: {nwb_path}")
    bucket, key = nwb_path.removeprefix("s3://").split("/", maxsplit=1)
    return bucket, key


class _ObstoreZarrV2Store(MutableMapping[str, bytes]):
    """Read-only Zarr v2 mapping backed by obstore's fsspec adapter."""

    def __init__(self, fs: Any, root_path: str) -> None:
        """Create a store rooted at a bucket-relative object prefix."""
        self._fs = fs
        self._root_path = root_path.rstrip("/")

    def __getitem__(self, key: str) -> bytes:
        """Return one Zarr metadata or chunk object."""
        try:
            return bytes(self._fs.cat_file(self._path_for(key)))
        except Exception as exc:
            if isinstance(exc, self._missing_exceptions()):
                raise KeyError(key) from exc
            raise

    def __setitem__(self, key: str, value: bytes) -> None:
        """Reject writes because benchmark stores are read-only."""
        raise TypeError(f"{type(self).__name__} is read-only")

    def __delitem__(self, key: str) -> None:
        """Reject deletes because benchmark stores are read-only."""
        raise TypeError(f"{type(self).__name__} is read-only")

    def __iter__(self) -> Iterator[str]:
        """Yield Zarr object keys relative to the store root."""
        prefix = f"{self._root_path}/"
        for path in self._fs.find(self._root_path):
            yield path.removeprefix(prefix)

    def __len__(self) -> int:
        """Return the number of objects below the store root."""
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        """Return whether a Zarr key exists in the store."""
        if not isinstance(key, str):
            return False
        try:
            return bool(self._fs.exists(self._path_for(key)))
        except Exception:
            return False

    def _path_for(self, key: str) -> str:
        """Return the bucket-relative object path for a Zarr key."""
        stripped = key.lstrip("/")
        if not stripped:
            return self._root_path
        return f"{self._root_path}/{stripped}"

    def _missing_exceptions(self) -> tuple[type[BaseException], ...]:
        """Return filesystem exceptions that should behave like missing keys."""
        missing = getattr(self._fs, "missing_exceptions", (FileNotFoundError,))
        if isinstance(missing, tuple):
            return missing
        return tuple(missing)