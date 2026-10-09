# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "altair",
#   "escapewheel-reader",
#   "numpy",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# escapewheel-reader = { git = "https://github.com/galenlynch/escapewheel", subdirectory = "escapewheel-reader" }
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///

"""Escapewheel implementation of the dynamic-foraging data-access benchmark."""

from __future__ import annotations

import escapewheel_reader
import numpy as np
import pydantic_settings

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_SOURCE_TYPE = "escapewheel"
_SPIKE_TIME_DOWNLOAD_COUNT = 1_000_000
_UNIT_SELECTORS = (("671", "45883-2"), ("5", "46101"))
_NWB_FIRST_PROBE = "46110"


class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the Escapewheel runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_ignore_unknown_args=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_",
    )

    benchmark: str = "dynamic_foraging_nwb_v0"
    implementation_id: str = "escapewheel-reader"


settings: Settings | None = None
_packets: list[escapewheel_reader.Packet] = []


def _data_sources(context: neurodatabench.RunContext) -> list[str]:
    """Return the Escapewheel packet sources declared by the benchmark."""
    try:
        data_sources = context.benchmark.data_sources[_SOURCE_TYPE]
    except KeyError as exc:
        raise ValueError(
            f"Benchmark {context.benchmark.id!r} has no {_SOURCE_TYPE!r} data sources."
        ) from exc
    if not data_sources:
        raise ValueError(
            f"Benchmark {context.benchmark.id!r} has an empty {_SOURCE_TYPE!r} source list."
        )
    return data_sources


def _opened_packets() -> list[escapewheel_reader.Packet]:
    """Return packet handles initialized during setup."""
    if not _packets:
        raise RuntimeError(
            "Escapewheel packets must be opened before answering questions."
        )
    return _packets


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every dynamic-foraging benchmark question."""
    packets = _opened_packets()
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _table_counts(packets, "core.units", "unit_id")
            case "predicated_spike_times":
                answer = _selected_spike_counts(packets)
            case "multisession_table_query":
                answer = _table_counts(
                    packets,
                    "dynamic_foraging.trials",
                    "trial_id",
                )
            case "large_array":
                answer = _large_array(packets[0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def _table_counts(
    packets: list[escapewheel_reader.Packet],
    resource_id: str,
    identifier: str,
) -> list[int]:
    """Return row counts for one projected table resource in packet order."""
    return [
        packet.table(resource_id, columns=[identifier]).num_rows for packet in packets
    ]


def _selected_spike_counts(
    packets: list[escapewheel_reader.Packet],
) -> list[int]:
    """Return spike counts for the benchmark's selected unit in each packet."""
    if len(packets) != len(_UNIT_SELECTORS):
        raise ValueError(
            f"Expected {len(_UNIT_SELECTORS)} dynamic-foraging packets, "
            f"found {len(packets)}."
        )

    counts: list[int] = []
    for packet, (unit_label, probe_name) in zip(
        packets,
        _UNIT_SELECTORS,
        strict=True,
    ):
        probes = packet.table(
            "core.probes",
            columns=["probe_id"],
            probe_name=probe_name,
        )
        if probes.num_rows != 1:
            raise ValueError(
                f"Expected one probe named {probe_name!r} in {packet.location}, "
                f"found {probes.num_rows}."
            )
        units = packet.table(
            "core.units",
            columns=["n_spikes"],
            local_unit_label=unit_label,
            probe_id=int(probes["probe_id"][0].as_py()),
        )
        if units.num_rows != 1:
            raise ValueError(
                f"Expected one unit labeled {unit_label!r} on probe "
                f"{probe_name!r} in {packet.location}, found {units.num_rows}."
            )
        n_spikes = units["n_spikes"][0].as_py()
        if n_spikes is None:
            raise ValueError(
                f"Unit {unit_label!r} on probe {probe_name!r} has no n_spikes value."
            )
        counts.append(int(n_spikes))
    return counts


def _large_array(packet: escapewheel_reader.Packet) -> float:
    """Return the NWB-coordinate sum for its first million ragged spike values."""
    logger.debug(
        "Reading the Escapewheel equivalent of the first %d NWB spike times from %s.",
        _SPIKE_TIME_DOWNLOAD_COUNT,
        packet.location,
    )
    unit_ids = _large_array_unit_ids(packet)
    units = packet.table("core.units", columns=["unit_id"], unit_id=unit_ids)
    if units["unit_id"].to_pylist() != unit_ids:
        raise ValueError(
            "Escapewheel unit storage order does not reproduce the NWB ragged-array order."
        )
    stored = escapewheel_reader.spike_times(
        packet,
        units=units,
        clock=escapewheel_reader.STORED,
    )
    reference_ticks = escapewheel_reader.on_clock(packet, stored)
    if reference_ticks.num_rows < _SPIKE_TIME_DOWNLOAD_COUNT:
        raise ValueError(
            f"Expected {_SPIKE_TIME_DOWNLOAD_COUNT} spike ticks in {packet.location}, "
            f"found {reference_ticks.num_rows}."
        )
    ticks = reference_ticks["t_tick"].slice(0, _SPIKE_TIME_DOWNLOAD_COUNT)
    return float(
        np.sum(
            ticks.to_numpy(zero_copy_only=False).astype(np.float64)
            / _reference_tick_rate(packet),
            dtype=np.float64,
        )
    )


def _large_array_unit_ids(packet: escapewheel_reader.Packet) -> list[int]:
    """Return source-ordered unit IDs covering the NWB spike-array prefix."""
    probes = packet.table(
        "core.probes",
        columns=["probe_id"],
        probe_name=_NWB_FIRST_PROBE,
    )
    if probes.num_rows != 1:
        raise ValueError(
            f"Expected one probe named {_NWB_FIRST_PROBE!r} in {packet.location}, "
            f"found {probes.num_rows}."
        )
    rows = packet.table(
        "core.units",
        columns=["unit_id", "local_unit_label", "n_spikes"],
        probe_id=int(probes["probe_id"][0].as_py()),
    ).to_pylist()
    try:
        rows.sort(key=lambda row: int(row["local_unit_label"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Probe {_NWB_FIRST_PROBE!r} has a non-numeric local unit label."
        ) from exc

    selected: list[int] = []
    spike_count = 0
    for row in rows:
        n_spikes = row["n_spikes"]
        if n_spikes is None:
            raise ValueError(f"Unit {row['unit_id']} has no n_spikes value.")
        selected.append(int(row["unit_id"]))
        spike_count += int(n_spikes)
        if spike_count >= _SPIKE_TIME_DOWNLOAD_COUNT:
            return selected
    raise ValueError(
        f"Probe {_NWB_FIRST_PROBE!r} has only {spike_count} spikes; "
        f"expected at least {_SPIKE_TIME_DOWNLOAD_COUNT}."
    )


def _reference_tick_rate(packet: escapewheel_reader.Packet) -> float:
    """Return ticks per second for the packet's reference clock."""
    sessions = packet.table("core.sessions", columns=["reference_clock_id"])
    if sessions.num_rows != 1:
        raise ValueError(
            f"Expected one session in {packet.location}, found {sessions.num_rows}."
        )
    clocks = packet.table(
        "core.clocks",
        columns=["tick_rate_num", "tick_rate_den"],
        clock_id=int(sessions["reference_clock_id"][0].as_py()),
    )
    if clocks.num_rows != 1:
        raise ValueError(
            f"Expected one reference clock in {packet.location}, found {clocks.num_rows}."
        )
    numerator = clocks["tick_rate_num"][0].as_py()
    denominator = clocks["tick_rate_den"][0].as_py()
    if numerator is None or denominator in (None, 0):
        raise ValueError(
            f"Reference clock in {packet.location} has no rational tick rate."
        )
    return float(numerator) / float(denominator)


def setup(context: neurodatabench.RunContext) -> None:
    """Open every Escapewheel packet before measured question execution."""
    data_sources = _data_sources(context)
    logger.debug("Opening %d Escapewheel packets.", len(data_sources))
    _packets.clear()
    _packets.extend(escapewheel_reader.open_packet(source) for source in data_sources)


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Declare that the Escapewheel reader has no managed local cache."""
    logger.debug(
        "No Escapewheel local cache to clear for %d packets.",
        len(_data_sources(context)),
    )


def teardown(context: neurodatabench.RunContext) -> None:
    """Release packet references after benchmark execution."""
    logger.debug(
        "Releasing %d Escapewheel packet handles.",
        len(_data_sources(context)),
    )
    _packets.clear()


if __name__ == "__main__":
    settings = Settings()
    neurodatabench.main(
        implementation_id=settings.implementation_id,
        implementation_source_type=_SOURCE_TYPE,
        implementation_nwb_interface=None,
        implementation_object_store_backend="pyarrow",
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=settings.benchmark,
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
