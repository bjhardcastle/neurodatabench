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

"""Benchmark remote Zarr ROI-table reads with current access methods.

The implementation keeps the data on S3 and reads each ROI table directly from
the object store. Direct Zarr reads use either s3fs or obstore. VirtualiZarr
manifest references and Icechunk repositories are local metadata caches whose
referenced chunks remain on S3. Cold cache runs build those metadata caches in
the timed setup phase; warm runs load a cache prepared by an earlier run.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import zarr

import neurodatabench

logger = neurodatabench.get_logger(__name__)

TableRef: TypeAlias = tuple[str, str, str]

_DEFAULT_BENCHMARK = "multiplane_ophys_roi_zarr_v0"
_DEFAULT_METHOD = "zarr-obstore"
_CACHE_VERSION = 1
_EXPECTED_TABLE_COUNT = 288
_OBJECT_STORE_BUCKET = "aind-open-data"
_OBJECT_STORE_REGION = "us-west-2"
_VIRTUAL_CHUNK_PREFIX = f"s3://{_OBJECT_STORE_BUCKET}/"
_ROI_SUFFIX = "image_segmentation/roi_table"

state: dict[str, Any] = {}


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Remove the selected local metadata cache before a cold measured run."""
    if _local_cache() != "cold":
        logger.debug("Keeping metadata cache for warm %s run.", _method())
        return
    cache_path = _cache_path()
    logger.debug("Clearing cold %s cache at %s.", _method(), cache_path)
    if cache_path.is_dir():
        shutil.rmtree(cache_path)
    elif cache_path.exists():
        cache_path.unlink()


def setup(context: neurodatabench.RunContext) -> None:
    """Prepare remote stores and build or load any selected metadata cache."""
    method = _method()
    logger.debug(
        "Preparing %s for %d remote NWB stores.",
        method,
        len(context.benchmark.data_sources),
    )
    os.environ.setdefault("AWS_REGION", _OBJECT_STORE_REGION)
    _quiet_storage_loggers()

    discovery_backend = (
        method if method in {"zarr-s3fs", "zarr-obstore"} else "zarr-s3fs"
    )
    discovery_stores = _build_direct_stores(
        context.benchmark.data_sources,
        backend=discovery_backend,
    )
    tables = _discover_tables(context.benchmark.data_sources, discovery_stores)
    if len(tables) != _EXPECTED_TABLE_COUNT:
        raise RuntimeError(
            f"Expected {_EXPECTED_TABLE_COUNT} ROI tables, discovered {len(tables)}."
        )
    state.clear()
    state["tables"] = tables

    if method in {"zarr-s3fs", "zarr-obstore"}:
        state["stores"] = (
            discovery_stores
            if discovery_backend == method
            else _build_direct_stores(context.benchmark.data_sources, backend=method)
        )
    elif method == "virtualizarr":
        state["registry"] = _new_virtualizarr_registry()
        state["manifests"] = _prepare_virtualizarr_manifests(
            context.benchmark,
            tables,
            state["registry"],
        )
    elif method == "icechunk":
        state["icechunk_session"] = _prepare_icechunk_session(
            context.benchmark,
            tables,
        )
    else:
        raise ValueError(f"Unsupported NDB_ZARR_METHOD: {method}")
    logger.debug("Prepared %d ROI tables for %s.", len(tables), method)


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Read every ROI table and submit the benchmark summary."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        if question.id != "roi_table_read_summary":
            raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, _read_all_tables())


def teardown(context: neurodatabench.RunContext) -> None:
    """Release references to remote stores and cached metadata."""
    logger.debug("Releasing %d cached benchmark objects.", len(state))
    state.clear()


def _quiet_storage_loggers() -> None:
    """Keep dependency logging from obscuring benchmark diagnostics."""
    for logger_name in (
        "aiobotocore",
        "botocore",
        "fsspec",
        "s3fs",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _method() -> str:
    """Return the matrix-selected Zarr access method."""
    return os.environ.get("NDB_ZARR_METHOD", _DEFAULT_METHOD)


def _local_cache() -> neurodatabench.models.LocalCacheState | None:
    """Return the local-cache state for the selected access method."""
    if _method() in {"zarr-s3fs", "zarr-obstore"}:
        return None
    value = os.environ.get("NDB_LOCAL_CACHE", "cold")
    if value not in {"cold", "warm"}:
        raise ValueError("NDB_LOCAL_CACHE must be 'cold' or 'warm'.")
    return value  # type: ignore[return-value]


def _cache_path() -> Path:
    """Return the matrix-provided cache directory."""
    value = os.environ.get("NDB_ZARR_CACHE_PATH")
    if not value:
        raise RuntimeError(
            "NDB_ZARR_CACHE_PATH is required for VirtualiZarr and Icechunk runs."
        )
    return Path(value)


def _build_direct_stores(
    data_sources: Iterable[str],
    *,
    backend: str,
) -> dict[str, Any]:
    """Create one read-only Zarr store per S3 data source."""
    stores: dict[str, Any] = {}
    for data_source in data_sources:
        if backend == "zarr-s3fs":
            from zarr.storage import FsspecStore

            store = FsspecStore.from_url(
                data_source,
                storage_options={
                    "anon": True,
                    "client_kwargs": {"region_name": _OBJECT_STORE_REGION},
                },
                read_only=True,
            )
        elif backend == "zarr-obstore":
            from obstore import fsspec as obstore_fsspec
            from zarr.storage import FsspecStore

            filesystem = obstore_fsspec.FsspecStore(
                "s3",
                config={
                    "region": _OBJECT_STORE_REGION,
                    "skip_signature": True,
                },
                asynchronous=True,
            )
            store = FsspecStore(
                fs=filesystem,
                path=data_source.removeprefix("s3://"),
                read_only=True,
            )
        else:
            raise ValueError(f"Unsupported direct Zarr backend: {backend}")
        stores[data_source] = store
    return stores


def _discover_tables(
    data_sources: Iterable[str],
    stores: dict[str, Any],
) -> list[TableRef]:
    """Discover ROI tables below each benchmark data source."""
    tables: list[TableRef] = []
    for data_source in data_sources:
        processing = zarr.open_group(
            stores[data_source],
            path="processing",
            mode="r",
            zarr_format=2,
        )
        for plane in sorted(processing.group_keys()):
            group = f"processing/{plane}/{_ROI_SUFFIX}"
            try:
                zarr.open_group(
                    stores[data_source],
                    path=group,
                    mode="r",
                    zarr_format=2,
                )
            except (FileNotFoundError, KeyError, zarr.errors.GroupNotFoundError):
                logger.debug("Skipping %s without an ROI table.", group)
                continue
            tables.append((data_source, plane, group))
    return tables


def _new_virtualizarr_registry() -> Any:
    """Create an anonymous obstore registry for VirtualiZarr manifests."""
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import S3Store

    store = S3Store(
        bucket=_OBJECT_STORE_BUCKET,
        config={
            "region": _OBJECT_STORE_REGION,
            "skip_signature": True,
        },
    )
    return ObjectStoreRegistry({_VIRTUAL_CHUNK_PREFIX.rstrip("/"): store})


def _prepare_virtualizarr_manifests(
    benchmark: neurodatabench.models.Benchmark,
    tables: list[TableRef],
    registry: Any,
) -> dict[tuple[str, str], Any]:
    """Build a cold VirtualiZarr manifest cache or load a warm one."""
    if _local_cache() == "warm":
        return _load_manifest_cache(benchmark, tables, registry)

    from virtualizarr.parsers import ZarrParser

    cache_path = _cache_path()
    cache_path.mkdir(parents=True, exist_ok=True)
    manifests: dict[tuple[str, str], Any] = {}
    entries: list[dict[str, str]] = []
    for index, (data_source, _, group) in enumerate(tables):
        logger.debug("Building VirtualiZarr manifest %d/%d.", index + 1, len(tables))
        manifest = ZarrParser(group=group)(data_source, registry)
        filename = f"manifest-{index:04d}.json"
        _write_json(cache_path / filename, _manifest_to_payload(manifest))
        manifests[(data_source, group)] = manifest
        entries.append({"data_source": data_source, "group": group, "file": filename})
    _write_cache_metadata(cache_path, benchmark, tables, entries)
    return manifests


def _load_manifest_cache(
    benchmark: neurodatabench.models.Benchmark,
    tables: list[TableRef],
    registry: Any,
) -> dict[tuple[str, str], Any]:
    """Load and validate a persisted VirtualiZarr manifest cache."""
    from virtualizarr.manifests import ManifestArray, ManifestGroup, ManifestStore
    from zarr.core.metadata.v3 import ArrayV3Metadata

    cache_path = _cache_path()
    metadata = _read_cache_metadata(cache_path, benchmark, tables)
    manifests: dict[tuple[str, str], Any] = {}
    for entry in metadata["entries"]:
        filename = str(entry["file"])
        payload = json.loads((cache_path / filename).read_text(encoding="utf-8"))
        group = ManifestGroup(
            arrays={
                name: ManifestArray(
                    ArrayV3Metadata.from_dict(array_payload["metadata"]),
                    array_payload["manifest"],
                )
                for name, array_payload in payload["arrays"].items()
            },
            attributes=payload.get("attributes", {}),
        )
        data_source = str(entry["data_source"])
        group_path = str(entry["group"])
        manifests[(data_source, group_path)] = ManifestStore(
            group,
            registry=registry,
        )
    return manifests


def _manifest_to_payload(manifest: Any) -> dict[str, Any]:
    """Convert a VirtualiZarr manifest into a JSON-serializable payload."""
    group = manifest._group
    return {
        "attributes": group.metadata.to_dict()["attributes"],
        "arrays": {
            name: {
                "metadata": array.metadata.to_dict(),
                "manifest": array.manifest.dict(),
            }
            for name, array in group.arrays.items()
        },
    }


def _prepare_icechunk_session(
    benchmark: neurodatabench.models.Benchmark,
    tables: list[TableRef],
) -> Any:
    """Build a cold Icechunk repository or open a warm repository session."""
    import icechunk
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import ZarrParser

    storage = icechunk.local_filesystem_storage(str(_cache_path()))
    credentials = icechunk.containers_credentials(
        {_VIRTUAL_CHUNK_PREFIX: icechunk.s3_anonymous_credentials()}
    )
    if _local_cache() == "warm":
        _read_cache_metadata(_cache_path(), benchmark, tables)
    if _local_cache() == "cold":
        config = icechunk.RepositoryConfig.default()
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                _VIRTUAL_CHUNK_PREFIX,
                icechunk.s3_store(
                    region=_OBJECT_STORE_REGION,
                    anonymous=True,
                ),
            )
        )
        repository = icechunk.Repository.open_or_create(
            storage,
            config=config,
            authorize_virtual_chunk_access=credentials,
        )
        writable = repository.writable_session("main")
        registry = _new_virtualizarr_registry()
        for index, (data_source, _, group) in enumerate(tables):
            logger.debug("Building Icechunk metadata %d/%d.", index + 1, len(tables))
            virtual_dataset = open_virtual_dataset(
                data_source,
                registry=registry,
                parser=ZarrParser(group=group),
                loadable_variables=[],
            )
            virtual_dataset.vz.to_icechunk(
                writable.store,
                group=f"{_source_id(data_source)}/{group}",
                validate_containers=False,
            )
        writable.commit(f"cache {benchmark.id}")
        _write_cache_metadata(
            _cache_path(),
            benchmark,
            tables,
            [
                {"data_source": data_source, "group": group}
                for data_source, _, group in tables
            ],
        )

    repository = icechunk.Repository.open(
        storage,
        authorize_virtual_chunk_access=credentials,
    )
    return repository.readonly_session("main")


def _read_all_tables() -> neurodatabench.models.JsonObject:
    """Read every array in every ROI table and return the benchmark summary."""
    tables: list[TableRef] = state["tables"]
    first_image_mask_mean: float | None = None
    total_roi_count = 0

    for index, (data_source, _, group_path) in enumerate(tables):
        logger.debug("Reading ROI table %d/%d.", index + 1, len(tables))
        group = _open_table(data_source, group_path)
        table_roi_count: int | None = None
        for array_name in sorted(group.array_keys()):
            values = np.asarray(group[array_name][:])
            if array_name == "id":
                table_roi_count = int(values.shape[0])
            if index == 0 and array_name == "image_mask":
                first_image_mask_mean = float(np.mean(values, dtype=np.float64))
        if table_roi_count is None:
            raise RuntimeError(f"ROI table {group_path} has no id array.")
        total_roi_count += table_roi_count

    if first_image_mask_mean is None:
        raise RuntimeError("The first ROI table has no image_mask array.")
    return {
        "session_count": len({data_source for data_source, _, _ in tables}),
        "table_count": len(tables),
        "roi_count": total_roi_count,
        "first_table_image_mask_mean": first_image_mask_mean,
    }


def _open_table(data_source: str, group_path: str) -> Any:
    """Open one ROI table through the selected access method."""
    method = _method()
    if method in {"zarr-s3fs", "zarr-obstore"}:
        return zarr.open_group(
            state["stores"][data_source],
            path=group_path,
            mode="r",
            zarr_format=2,
        )
    if method == "virtualizarr":
        return zarr.open_group(
            state["manifests"][(data_source, group_path)],
            mode="r",
            zarr_format=3,
        )
    if method == "icechunk":
        return zarr.open_group(
            state["icechunk_session"].store,
            path=f"{_source_id(data_source)}/{group_path}",
            mode="r",
            zarr_format=3,
        )
    raise ValueError(f"Unsupported NDB_ZARR_METHOD: {method}")


def _source_id(data_source: str) -> str:
    """Return the session directory name represented by an S3 URI."""
    prefix = f"s3://{_OBJECT_STORE_BUCKET}/"
    if not data_source.startswith(prefix) or not data_source.endswith(
        "/pophys.nwb.zarr"
    ):
        raise ValueError(f"Unexpected benchmark data source: {data_source}")
    return data_source.removeprefix(prefix).removesuffix("/pophys.nwb.zarr")


def _write_cache_metadata(
    cache_path: Path,
    benchmark: neurodatabench.models.Benchmark,
    tables: list[TableRef],
    entries: list[dict[str, str]],
) -> None:
    """Write cache identity and table order after a successful cache build."""
    _write_json(
        cache_path / "cache.json",
        {
            "version": _CACHE_VERSION,
            "method": _method(),
            "benchmark_id": benchmark.id,
            "data_sources": benchmark.data_sources,
            "tables": [list(table) for table in tables],
            "entries": entries,
        },
    )


def _read_cache_metadata(
    cache_path: Path,
    benchmark: neurodatabench.models.Benchmark,
    tables: list[TableRef],
) -> dict[str, Any]:
    """Read and validate cache identity before using a warm cache."""
    metadata_path = cache_path / "cache.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Warm {_method()} cache is missing at {cache_path}; run the cold row first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("version") != _CACHE_VERSION
        or metadata.get("method") != _method()
        or metadata.get("benchmark_id") != benchmark.id
        or metadata.get("data_sources") != benchmark.data_sources
        or metadata.get("tables") != [list(table) for table in tables]
    ):
        raise RuntimeError(
            f"Warm {_method()} cache at {cache_path} does not match the benchmark."
        )
    return metadata


def _write_json(path: Path, value: object) -> None:
    """Write JSON to a temporary sibling and atomically publish it."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(value, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _object_store_backend() -> str:
    """Return the object-store backend label recorded in run metadata."""
    return "s3fs" if _method() == "zarr-s3fs" else "obstore"


def _default_implementation_id() -> str:
    """Return the default implementation identifier."""
    return _method()


def main() -> None:
    """Run the shared ROI-table implementation selected by the environment."""
    local_cache = _local_cache()
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            _default_implementation_id(),
        ),
        implementation_nwb_interface=_method(),
        implementation_object_store_backend=_object_store_backend(),
        implementation_local_cache=local_cache,
        implementation_remote_cache=False,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=clear_cache if local_cache is not None else None,
        submit_answers=submit_answers,
        teardown=teardown,
    )


if __name__ == "__main__":
    main()
