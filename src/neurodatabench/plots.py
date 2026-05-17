"""Altair dashboard generation for NeuroDataBench result artifacts."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import altair as alt

import neurodatabench.models

alt.data_transformers.disable_max_rows()

logger = logging.getLogger(__name__)

_LEADERBOARD_PLOT_MAX_SECONDS = 20.0
_DASHBOARD_COLORS: dict[str, str] = {
    "timing": "#d97706",
    "memory": "#e11d48",
    "cpu_process": "#059669",
    "cpu_system": "#7c3aed",
    "network_received": "#0f766e",
    "event_setup": "#64748b",
    "event_answer": "#f97316",
    "empty_state_text": "#374151",
}


def _write_result_dashboard(
    *,
    out_dir: Path,
    metadata: neurodatabench.models.JsonObject,
    timings: neurodatabench.models.RunTimings,
    profile_samples: list[neurodatabench.models.JsonObject],
    profile_summary: neurodatabench.models.JsonObject,
    run_start_wall_time_ns: int,
) -> None:
    """Write the Altair dashboard for one benchmark result directory."""
    logger.debug("Writing result dashboard to %s.", out_dir)
    dashboard = _result_dashboard_chart(
        metadata=metadata,
        timings=timings,
        profile_samples=profile_samples,
        profile_summary=profile_summary,
        run_start_wall_time_ns=run_start_wall_time_ns,
    )
    _save_altair_chart(out_dir / "dashboard.html", dashboard)


def _write_leaderboard_plot(
    *,
    results_dir: Path,
    rows: list[neurodatabench.models.JsonObject],
) -> None:
    """Write an aggregate leaderboard plot for a results directory."""
    logger.debug("Writing leaderboard plot to %s.", results_dir)
    if not rows:
        return

    chart = _leaderboard_plot_chart(rows)
    _save_altair_chart(results_dir / "leaderboard.html", chart)


def _leaderboard_plot_chart(
    rows: list[neurodatabench.models.JsonObject],
) -> alt.LayerChart:
    """Return the aggregate leaderboard chart with long runtimes capped."""
    plot_rows = _leaderboard_plot_rows(rows)
    y_sort = [str(row["leaderboard_label"]) for row in plot_rows]
    y_encoding = alt.Y(
        "leaderboard_label:N",
        title=None,
        sort=y_sort,
    )
    base = alt.Chart(alt.Data(values=plot_rows))
    completed_chart = (
        base.transform_filter("!datum.timed_out")
        .mark_bar()
        .encode(
            x=alt.X(
                "plot_total_seconds:Q",
                title="total seconds (capped at 20 s)",
                scale=alt.Scale(domain=[0.0, _LEADERBOARD_PLOT_MAX_SECONDS]),
            ),
            y=y_encoding,
            color=alt.Color(
                "benchmark_id:N",
                title="benchmark",
                scale=alt.Scale(
                    range=[
                        _DASHBOARD_COLORS["cpu_process"],
                        _DASHBOARD_COLORS["timing"],
                        _DASHBOARD_COLORS["memory"],
                        _DASHBOARD_COLORS["cpu_system"],
                        _DASHBOARD_COLORS["network_received"],
                    ]
                ),
            ),
            tooltip=_leaderboard_tooltips(include_timeout_seconds=False),
        )
    )
    timeout_chart = (
        base.transform_filter(alt.datum.timed_out)
        .mark_bar()
        .encode(
            x=alt.X(
                "plot_total_seconds:Q",
                title="total seconds (capped at 20 s)",
                scale=alt.Scale(domain=[0.0, _LEADERBOARD_PLOT_MAX_SECONDS]),
            ),
            y=y_encoding,
            color=alt.Color(
                "benchmark_id:N",
                title="benchmark",
                scale=alt.Scale(
                    range=[
                        _DASHBOARD_COLORS["cpu_process"],
                        _DASHBOARD_COLORS["timing"],
                        _DASHBOARD_COLORS["memory"],
                        _DASHBOARD_COLORS["cpu_system"],
                        _DASHBOARD_COLORS["network_received"],
                    ]
                ),
            ),
            tooltip=_leaderboard_tooltips(include_timeout_seconds=True),
        )
    )
    truncated_labels = (
        base.transform_filter(alt.datum.total_seconds_truncated)
        .mark_text(
            align="right",
            baseline="middle",
            color="#ffffff",
            dx=-6,
            fontSize=11,
        )
        .encode(
            x=alt.X("plot_total_seconds:Q"),
            y=y_encoding,
            text=alt.Text("plot_total_seconds_label:N"),
        )
    )
    timeout_labels = (
        base.transform_filter(alt.datum.timed_out)
        .mark_text(
            align="left",
            baseline="middle",
            color="#991b1b",
            dx=6,
            fontSize=11,
            fontWeight="bold",
        )
        .encode(
            x=alt.X("plot_total_seconds:Q"),
            y=y_encoding,
            text=alt.Text("timed_out_label:N"),
        )
    )
    cap_rule = (
        alt.Chart(
            alt.Data(values=[{"cap_seconds": _LEADERBOARD_PLOT_MAX_SECONDS}])
        )
        .mark_rule(color="#374151", strokeDash=[5, 4], strokeWidth=1.5)
        .encode(x=alt.X("cap_seconds:Q"))
    )
    return (
        alt.layer(
            completed_chart,
            timeout_chart,
            cap_rule,
            truncated_labels,
            timeout_labels,
        )
        .properties(
            title=alt.TitleParams(
                text="NeuroDataBench Leaderboard",
                anchor="start",
            ),
            width=904,
            height=max(90, min(28 * len(rows), 720)),
        )
    )


def _leaderboard_tooltips(
    *,
    include_timeout_seconds: bool,
) -> list[alt.Tooltip]:
    """Return leaderboard tooltip fields for completed or timed-out bars."""
    tooltips = [
        alt.Tooltip("implementation_id:N", title="Implementation"),
        alt.Tooltip("nwb_interface:N", title="NWB interface"),
        alt.Tooltip("object_store_backend:N", title="Object store backend"),
        alt.Tooltip("benchmark_id:N", title="Benchmark"),
        alt.Tooltip("run_status:N", title="Run status"),
        alt.Tooltip("datetime_utc:N", title="Run UTC"),
        alt.Tooltip("local_cache:N", title="Local cache"),
        alt.Tooltip("remote_cache:N", title="Remote cache"),
        alt.Tooltip("total_seconds:Q", title="Total seconds", format=",.3f"),
        alt.Tooltip("setup_seconds:Q", title="Setup seconds", format=",.3f"),
        alt.Tooltip(
            "submit_answers_seconds:Q",
            title="Submit seconds",
            format=",.3f",
        ),
        alt.Tooltip(
            "peak_rss_delta_mib:Q",
            title="Peak RSS delta MiB",
            format=",.1f",
        ),
        alt.Tooltip(
            "peak_rss_mib:Q",
            title="Raw peak RSS MiB",
            format=",.1f",
        ),
        alt.Tooltip(
            "network_received_mib:Q",
            title="Network received MiB",
            format=",.1f",
        ),
    ]
    if include_timeout_seconds:
        tooltips.insert(
            5,
            alt.Tooltip(
                "timeout_seconds:Q",
                title="Timeout seconds",
                format=",.3f",
            ),
        )
    return tooltips


def _leaderboard_plot_rows(
    rows: list[neurodatabench.models.JsonObject],
) -> list[neurodatabench.models.JsonObject]:
    """Return chart rows with capped display seconds and truncation labels."""
    plot_rows: list[neurodatabench.models.JsonObject] = []
    for row in rows:
        plot_row = dict(row)
        total_seconds = _leaderboard_number_at(row, "total_seconds") or 0.0
        is_truncated = total_seconds > _LEADERBOARD_PLOT_MAX_SECONDS
        timed_out = _leaderboard_timed_out(row)
        timeout_seconds = _leaderboard_number_at(row, "timeout_seconds")
        plot_row["total_seconds"] = total_seconds
        plot_row["plot_total_seconds"] = min(
            total_seconds,
            _LEADERBOARD_PLOT_MAX_SECONDS,
        )
        plot_row["run_status"] = row.get("run_status") or (
            "timed out" if timed_out else "correct"
        )
        plot_row["timed_out"] = timed_out
        plot_row["timed_out_label"] = "timed out" if timed_out else ""
        if not timed_out:
            plot_row.pop("timeout_seconds", None)
        elif timeout_seconds is not None:
            plot_row["timeout_seconds"] = timeout_seconds
        plot_row["total_seconds_truncated"] = is_truncated
        plot_row["plot_total_seconds_label"] = (
            f"{total_seconds:,.1f} s" if is_truncated else ""
        )
        plot_rows.append(plot_row)
    return sorted(
        plot_rows,
        key=lambda row: (
            str(row.get("benchmark_id", "")),
            _leaderboard_number_at(row, "total_seconds") or 0.0,
            str(row.get("implementation_id", "")),
            str(row.get("datetime_utc", "")),
            str(row.get("result_dir", "")),
        ),
    )


def _leaderboard_timed_out(row: neurodatabench.models.JsonObject) -> bool:
    """Return whether a leaderboard row represents a timed-out run."""
    timed_out = row.get("timed_out")
    if isinstance(timed_out, bool):
        return timed_out
    if isinstance(timed_out, str):
        return timed_out.strip().lower() == "true"
    run_status = row.get("run_status")
    return isinstance(run_status, str) and run_status.strip().lower() == "timed out"


def _leaderboard_number_at(
    row: neurodatabench.models.JsonObject,
    key: str,
) -> float | None:
    """Return a leaderboard number from JSON-native or CSV-like row data."""
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return float(stripped_value)
        except ValueError:
            return None
    return None


def _result_dashboard_chart(
    *,
    metadata: neurodatabench.models.JsonObject,
    timings: neurodatabench.models.RunTimings,
    profile_samples: list[neurodatabench.models.JsonObject],
    profile_summary: neurodatabench.models.JsonObject,
    run_start_wall_time_ns: int,
) -> alt.VConcatChart:
    """Return a single dashboard chart for benchmark timing and profiling."""
    time_domain = _elapsed_time_domain(
        timings=timings,
        profile_samples=profile_samples,
        run_start_wall_time_ns=run_start_wall_time_ns,
    )
    dashboard = alt.vconcat(
        _timing_summary_chart(timings, time_domain=time_domain),
        _network_profile_chart(
            profile_samples=profile_samples,
            timings=timings,
            run_start_wall_time_ns=run_start_wall_time_ns,
            time_domain=time_domain,
        ),
        _memory_profile_chart(
            profile_samples=profile_samples,
            profile_summary=profile_summary,
            timings=timings,
            run_start_wall_time_ns=run_start_wall_time_ns,
            time_domain=time_domain,
        ),
        _cpu_profile_chart(
            profile_samples=profile_samples,
            timings=timings,
            run_start_wall_time_ns=run_start_wall_time_ns,
            time_domain=time_domain,
        ),
        spacing=10,
    ).resolve_scale(
        x="shared",
        color="independent",
    ).properties(
        title=_dashboard_title(metadata)
    )
    return dashboard


def _dashboard_title(metadata: neurodatabench.models.JsonObject) -> alt.TitleParams:
    """Return a dashboard title that identifies the benchmark run."""
    implementation_id = _metadata_string_at(
        metadata,
        ("implementation", "id"),
        default="unknown implementation",
    )
    benchmark_id = _metadata_string_at(
        metadata,
        ("benchmark", "id"),
        default="unknown benchmark",
    )
    subtitle_parts = [
        _metadata_label_value(metadata, "datetime_utc", "UTC"),
        _metadata_label_value(metadata, "hostname", "host"),
        _metadata_label_value(metadata, "benchmark_harness_version", "harness"),
        _metadata_label_value(metadata, ("benchmark", "nwb_format"), "format"),
        _metadata_label_value(
            metadata,
            ("implementation", "nwb_interface"),
            "NWB interface",
        ),
        _metadata_label_value(
            metadata,
            ("implementation", "object_store_backend"),
            "object store",
        ),
        _metadata_label_value(
            metadata,
            ("implementation", "local_cache"),
            "local cache",
        ),
        _metadata_label_value(
            metadata,
            ("implementation", "remote_cache"),
            "remote cache",
        ),
    ]
    subtitle = " | ".join(part for part in subtitle_parts if part is not None)
    return alt.TitleParams(
        text=f"NeuroDataBench: {implementation_id} / {benchmark_id}",
        subtitle=subtitle,
        anchor="start",
        fontSize=20,
        offset=16,
    )


def _metadata_label_value(
    metadata: neurodatabench.models.JsonObject,
    key_path: str | tuple[str, ...],
    label: str,
) -> str | None:
    """Return one compact metadata label/value string for the dashboard title."""
    value = _metadata_string_at(metadata, key_path)
    if value is None:
        return None
    return f"{label} {value}"


def _metadata_string_at(
    metadata: neurodatabench.models.JsonObject,
    key_path: str | tuple[str, ...],
    *,
    default: str | None = None,
) -> str | None:
    """Return a nested metadata value as a string, if present."""
    keys = (key_path,) if isinstance(key_path, str) else key_path
    value: object = metadata
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    if value is None:
        return default
    return str(value)


def _timing_summary_chart(
    timings: neurodatabench.models.RunTimings,
    *,
    time_domain: tuple[float, float],
) -> alt.Chart:
    """Return an Altair horizontal start/stop chart of benchmark timing phases."""
    rows = _timing_summary_rows(timings)
    phase_order = list(dict.fromkeys(str(row["phase"]) for row in rows))
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(
            color=_DASHBOARD_COLORS["timing"],
            size=8,
            stroke="#ffffff",
            strokeWidth=0.8,
        )
        .encode(
            x=alt.X(
                "start_seconds:Q",
                title="elapsed seconds",
                scale=alt.Scale(domain=list(time_domain)),
            ),
            x2="stop_seconds:Q",
            y=alt.Y("phase:N", title=None, sort=phase_order),
            tooltip=[
                alt.Tooltip("phase:N", title="Phase"),
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("start_seconds:Q", title="Start", format=",.3f"),
                alt.Tooltip("stop_seconds:Q", title="Stop", format=",.3f"),
                alt.Tooltip("duration_seconds:Q", title="Duration", format=",.3f"),
            ],
        )
        .properties(title="Benchmark Timing", width=904, height=48)
    )
    return chart


def _timing_summary_rows(
    timings: neurodatabench.models.RunTimings,
) -> list[dict[str, object]]:
    """Return phase timing rows, splitting submit answers by question submission."""
    rows: list[dict[str, object]] = []
    for phase_timing in timings.phase_timings:
        if phase_timing.phase == "total":
            continue
        if phase_timing.phase != "submit_answers":
            rows.append(_phase_timing_row(phase_timing, segment=phase_timing.phase))
            continue

        submit_rows = _submit_answer_timing_rows(phase_timing, timings)
        rows.extend(submit_rows)

    return rows


def _phase_timing_row(
    phase_timing: neurodatabench.models.RunPhaseTiming,
    *,
    segment: str,
) -> dict[str, object]:
    """Return a chart row for one timing segment."""
    return {
        "phase": phase_timing.phase,
        "segment": segment,
        "start_seconds": phase_timing.start_seconds,
        "stop_seconds": phase_timing.stop_seconds,
        "duration_seconds": phase_timing.duration_seconds,
    }


def _submit_answer_timing_rows(
    phase_timing: neurodatabench.models.RunPhaseTiming,
    timings: neurodatabench.models.RunTimings,
) -> list[dict[str, object]]:
    """Return submit-answer phase rows split by observed answer submissions."""
    submissions = sorted(
        (
            submission
            for submission in timings.answer_submissions
            if submission.submitted_elapsed_seconds is not None
        ),
        key=lambda submission: float(submission.submitted_elapsed_seconds or 0.0),
    )
    if not submissions:
        return [_phase_timing_row(phase_timing, segment=phase_timing.phase)]

    rows: list[dict[str, object]] = []
    previous_stop_seconds = phase_timing.start_seconds
    for submission in submissions:
        submitted_elapsed_seconds = float(submission.submitted_elapsed_seconds or 0.0)
        stop_seconds = min(
            max(submitted_elapsed_seconds, previous_stop_seconds),
            phase_timing.stop_seconds,
        )
        rows.append(
            {
                "phase": phase_timing.phase,
                "segment": submission.question_id,
                "start_seconds": previous_stop_seconds,
                "stop_seconds": stop_seconds,
                "duration_seconds": stop_seconds - previous_stop_seconds,
            }
        )
        previous_stop_seconds = stop_seconds

    if previous_stop_seconds < phase_timing.stop_seconds:
        rows.append(
            {
                "phase": phase_timing.phase,
                "segment": "after final submission",
                "start_seconds": previous_stop_seconds,
                "stop_seconds": phase_timing.stop_seconds,
                "duration_seconds": phase_timing.stop_seconds
                - previous_stop_seconds,
            }
        )
    return rows


def _memory_profile_chart(
    *,
    profile_samples: list[neurodatabench.models.JsonObject],
    profile_summary: neurodatabench.models.JsonObject,
    timings: neurodatabench.models.RunTimings,
    run_start_wall_time_ns: int,
    time_domain: tuple[float, float],
) -> alt.Chart | alt.LayerChart:
    """Return an Altair line chart of baseline-subtracted memory usage."""
    baseline_rss = _json_number_at(
        profile_summary,
        ("baseline_process_plus_children_rss_bytes",),
    )
    rows: list[dict[str, object]] = []
    for sample, elapsed_seconds in _profile_samples_with_elapsed_time(
        profile_samples,
        run_start_wall_time_ns,
    ):
        process_rss = _json_number_at(sample, ("process", "rss_bytes"))
        child_rss = _json_number_at(sample, ("children", "rss_bytes")) or 0.0
        if process_rss is None:
            continue
        total_rss = process_rss + child_rss
        if baseline_rss is None:
            baseline_rss = total_rss
        rows.append(
            {
                "elapsed_seconds": elapsed_seconds,
                "metric": "RSS delta (process + children)",
                "value_mib": max(total_rss - baseline_rss, 0.0) / 1_048_576,
            }
        )

    chart = _profile_line_chart(
        title="Memory Profile (baseline-subtracted)",
        rows=rows,
        y_field="value_mib",
        y_title="MiB above baseline",
        color_domain=["RSS delta (process + children)"],
        color_range=[_DASHBOARD_COLORS["memory"]],
        event_markers=_profile_event_marker_rows(timings),
        time_domain=time_domain,
        width=904,
        height=125,
    )
    return chart


def _cpu_profile_chart(
    *,
    profile_samples: list[neurodatabench.models.JsonObject],
    timings: neurodatabench.models.RunTimings,
    run_start_wall_time_ns: int,
    time_domain: tuple[float, float],
) -> alt.Chart | alt.LayerChart:
    """Return an Altair line chart of sampled CPU usage."""
    rows: list[dict[str, object]] = []
    for sample, elapsed_seconds in _profile_samples_with_elapsed_time(
        profile_samples,
        run_start_wall_time_ns,
    ):
        process_cpu = _json_number_at(sample, ("process", "cpu_percent"))
        child_cpu = _json_number_at(sample, ("children", "cpu_percent")) or 0.0
        system_cpu = _json_number_at(sample, ("system", "cpu_percent"))
        if process_cpu is not None:
            rows.append(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "metric": "process + children",
                    "value_percent": process_cpu + child_cpu,
                }
            )
        if system_cpu is not None:
            rows.append(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "metric": "system",
                    "value_percent": system_cpu,
                }
            )

    chart = _profile_line_chart(
        title="CPU Profile",
        rows=rows,
        y_field="value_percent",
        y_title="percent",
        color_domain=["process + children", "system"],
        color_range=[
            _DASHBOARD_COLORS["cpu_process"],
            _DASHBOARD_COLORS["cpu_system"],
        ],
        event_markers=_profile_event_marker_rows(timings),
        time_domain=time_domain,
        width=904,
        height=125,
    )
    return chart


def _network_profile_chart(
    *,
    profile_samples: list[neurodatabench.models.JsonObject],
    timings: neurodatabench.models.RunTimings,
    run_start_wall_time_ns: int,
    time_domain: tuple[float, float],
) -> alt.Chart | alt.LayerChart:
    """Return an Altair line chart of sampled system network usage."""
    baseline_received = None
    rows: list[dict[str, object]] = []
    for sample, elapsed_seconds in _profile_samples_with_elapsed_time(
        profile_samples,
        run_start_wall_time_ns,
    ):
        bytes_received = _json_number_at(sample, ("io", "net", "bytes_recv"))
        if bytes_received is None:
            continue
        if baseline_received is None:
            baseline_received = bytes_received
        received_delta = max(bytes_received - baseline_received, 0.0)
        rows.append(
            {
                "elapsed_seconds": elapsed_seconds,
                "metric": "received",
                "value_mib": received_delta / 1_048_576,
            }
        )

    chart = _profile_line_chart(
        title="Network Profile (system)",
        rows=rows,
        y_field="value_mib",
        y_title="MiB received since first sample",
        color_domain=["received"],
        color_range=[_DASHBOARD_COLORS["network_received"]],
        event_markers=_profile_event_marker_rows(timings),
        time_domain=time_domain,
        width=904,
        height=125,
    )
    return chart


def _profile_line_chart(
    *,
    title: str,
    rows: list[dict[str, object]],
    y_field: str,
    y_title: str,
    color_domain: Sequence[str],
    color_range: Sequence[str],
    event_markers: list[dict[str, object]],
    time_domain: tuple[float, float],
    width: int = 440,
    height: int = 300,
) -> alt.Chart | alt.LayerChart:
    """Return an Altair line chart for sampled profile rows."""
    if not rows:
        return (
            alt.Chart(alt.Data(values=[{"message": "No profile samples collected"}]))
            .mark_text(size=16, color=_DASHBOARD_COLORS["empty_state_text"])
            .encode(text="message:N")
            .properties(title=title, width=width, height=height)
        )

    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "elapsed_seconds:Q",
                title="elapsed seconds",
                scale=alt.Scale(domain=list(time_domain)),
            ),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=list(color_domain), range=list(color_range)),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("elapsed_seconds:Q", title="Elapsed seconds", format=",.3f"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=",.3f"),
            ],
        )
        .properties(title=title, width=width, height=height)
    )
    return _annotate_profile_chart(
        chart,
        event_markers=event_markers,
    )


def _annotate_profile_chart(
    chart: alt.Chart,
    *,
    event_markers: list[dict[str, object]],
) -> alt.Chart | alt.LayerChart:
    """Add hover-only event markers to a profile chart."""
    if not event_markers:
        return chart

    marker_source = alt.Data(values=event_markers)
    rules = (
        alt.Chart(marker_source)
        .mark_rule(strokeDash=[5, 4], strokeWidth=1.5)
        .encode(
            x=alt.X("elapsed_seconds:Q"),
            color=alt.Color(
                "event:N",
                legend=None,
                scale=alt.Scale(
                    domain=["setup complete", "answer submitted"],
                    range=[
                        _DASHBOARD_COLORS["event_setup"],
                        _DASHBOARD_COLORS["event_answer"],
                    ],
                ),
            ),
            tooltip=[
                alt.Tooltip("event:N", title="Event"),
                alt.Tooltip("detail:N", title="Detail"),
                alt.Tooltip("elapsed_seconds:Q", title="Elapsed seconds", format=",.3f"),
            ],
        )
    )
    return alt.layer(chart, rules).resolve_scale(color="independent")


def _elapsed_time_domain(
    *,
    timings: neurodatabench.models.RunTimings,
    profile_samples: list[neurodatabench.models.JsonObject],
    run_start_wall_time_ns: int,
) -> tuple[float, float]:
    """Return the shared elapsed-time domain for all dashboard panes."""
    elapsed_values = [
        max(0.0, phase_timing.stop_seconds)
        for phase_timing in timings.phase_timings
    ]
    elapsed_values.extend(
        elapsed_seconds
        for _, elapsed_seconds in _profile_samples_with_elapsed_time(
            profile_samples,
            run_start_wall_time_ns,
        )
    )
    max_elapsed = max(elapsed_values, default=0.0)
    return 0.0, max(max_elapsed, 0.001)


def _profile_event_marker_rows(
    timings: neurodatabench.models.RunTimings,
) -> list[dict[str, object]]:
    """Return setup and answer submission marker rows for profile charts."""
    rows: list[dict[str, object]] = []
    for phase_timing in timings.phase_timings:
        if phase_timing.phase != "setup":
            continue
        rows.append(
            {
                "event": "setup complete",
                "detail": "setup",
                "elapsed_seconds": phase_timing.stop_seconds,
            }
        )
        break
    for answer_submission in timings.answer_submissions:
        elapsed_seconds = answer_submission.submitted_elapsed_seconds
        if elapsed_seconds is None:
            continue
        rows.append(
            {
                "event": "answer submitted",
                "detail": answer_submission.question_id,
                "elapsed_seconds": elapsed_seconds,
            }
        )
    return rows


def _save_altair_chart(path: Path, chart: alt.TopLevelMixin) -> None:
    """Save an Altair chart as an HTML artifact."""
    chart.configure_axis(
        labelColor="#4b5563",
        titleColor="#374151",
    ).configure_view(stroke=None).save(path.as_posix())


def _profile_samples_with_elapsed_time(
    profile_samples: Sequence[neurodatabench.models.JsonObject],
    run_start_wall_time_ns: int,
) -> list[tuple[neurodatabench.models.JsonObject, float]]:
    """Return profile samples paired with elapsed seconds from run start."""
    rows: list[tuple[neurodatabench.models.JsonObject, float]] = []
    for sample in profile_samples:
        time_ns = _json_number_at(sample, ("time_ns",))
        if time_ns is None:
            continue
        elapsed_seconds = (time_ns - run_start_wall_time_ns) / 1_000_000_000
        if elapsed_seconds < 0:
            continue
        rows.append((sample, elapsed_seconds))
    return rows


def _json_number_at(value: object, keys: Sequence[str]) -> float | None:
    """Return a nested JSON number, if present."""
    current: object = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)
