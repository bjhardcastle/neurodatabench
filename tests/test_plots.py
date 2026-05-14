"""Tests for NeuroDataBench plot generation helpers."""

from __future__ import annotations

import unittest

import neurodatabench.plots


class PlotTests(unittest.TestCase):
    """Exercise chart-specific data shaping behavior."""

    def test_leaderboard_plot_rows_cap_runs_longer_than_twenty_seconds(self) -> None:
        """Leaderboard plot rows should cap visible bars without changing totals."""
        rows = [
            {
                "implementation_id": "fast",
                "nwb_interface": "lazynwb",
                "object_store_backend": "s3fs",
                "benchmark_id": "benchmark",
                "leaderboard_label": "fast",
                "total_seconds": 2.5,
            },
            {
                "implementation_id": "slow",
                "nwb_interface": "pynwb",
                "object_store_backend": "remfile",
                "benchmark_id": "benchmark",
                "leaderboard_label": "slow",
                "total_seconds": 24.25,
            },
            {
                "implementation_id": "timeout",
                "nwb_interface": "pynwb",
                "object_store_backend": "remfile",
                "benchmark_id": "benchmark",
                "leaderboard_label": "timeout",
                "timed_out": True,
                "total_seconds": 60.0,
                "timeout_seconds": 60.0,
            },
        ]

        plot_rows = neurodatabench.plots._leaderboard_plot_rows(rows)

        self.assertEqual(plot_rows[0]["plot_total_seconds"], 2.5)
        self.assertFalse(plot_rows[0]["total_seconds_truncated"])
        self.assertNotIn("timeout_seconds", plot_rows[0])
        self.assertEqual(plot_rows[1]["total_seconds"], 24.25)
        self.assertEqual(plot_rows[1]["plot_total_seconds"], 20.0)
        self.assertTrue(plot_rows[1]["total_seconds_truncated"])
        self.assertEqual(plot_rows[1]["plot_total_seconds_label"], "24.2 s")
        self.assertEqual(plot_rows[2]["plot_total_seconds"], 20.0)
        self.assertEqual(plot_rows[2]["run_status"], "timed out")
        self.assertEqual(plot_rows[2]["timed_out_label"], "timed out")
        self.assertEqual(plot_rows[2]["timeout_seconds"], 60.0)

    def test_leaderboard_plot_rows_include_csv_like_timeouts(self) -> None:
        """Leaderboard plot rows should coerce CSV-like timeout row values."""
        rows = [
            {
                "implementation_id": "timeout",
                "nwb_interface": "pynwb",
                "object_store_backend": "remfile",
                "benchmark_id": "benchmark",
                "leaderboard_label": "timeout",
                "timed_out": "True",
                "run_status": "timed out",
                "total_seconds": "60.0",
                "timeout_seconds": "60.0",
            }
        ]

        plot_rows = neurodatabench.plots._leaderboard_plot_rows(rows)

        self.assertEqual(plot_rows[0]["plot_total_seconds"], 20.0)
        self.assertEqual(plot_rows[0]["total_seconds"], 60.0)
        self.assertTrue(plot_rows[0]["timed_out"])
        self.assertEqual(plot_rows[0]["timed_out_label"], "timed out")
        self.assertEqual(plot_rows[0]["timeout_seconds"], 60.0)

    def test_leaderboard_plot_rows_sort_by_time_within_benchmark(self) -> None:
        """Leaderboard plot rows should keep benchmark groups sorted by runtime."""
        rows = [
            {
                "implementation_id": "b-slow",
                "benchmark_id": "b",
                "leaderboard_label": "b-slow",
                "total_seconds": 9.0,
            },
            {
                "implementation_id": "a-slow",
                "benchmark_id": "a",
                "leaderboard_label": "a-slow",
                "total_seconds": 8.0,
            },
            {
                "implementation_id": "b-fast",
                "benchmark_id": "b",
                "leaderboard_label": "b-fast",
                "total_seconds": 1.0,
            },
            {
                "implementation_id": "a-fast",
                "benchmark_id": "a",
                "leaderboard_label": "a-fast",
                "total_seconds": 2.0,
            },
        ]

        plot_rows = neurodatabench.plots._leaderboard_plot_rows(rows)

        self.assertEqual(
            [row["leaderboard_label"] for row in plot_rows],
            ["a-fast", "a-slow", "b-fast", "b-slow"],
        )

    def test_leaderboard_plot_chart_caps_twenty_second_axis(self) -> None:
        """Leaderboard chart axis should cap runtimes at twenty seconds."""
        chart = neurodatabench.plots._leaderboard_plot_chart(
            [
                {
                    "implementation_id": "slow",
                    "nwb_interface": "pynwb",
                    "object_store_backend": "remfile",
                    "benchmark_id": "benchmark",
                    "leaderboard_label": "slow",
                    "total_seconds": 24.25,
                }
            ]
        )

        spec = chart.to_dict()

        self.assertNotIn("subtitle", spec["title"])
        self.assertEqual(
            spec["layer"][0]["encoding"]["x"]["title"],
            "total seconds (capped at 20 s)",
        )
        self.assertEqual(
            spec["layer"][0]["encoding"]["x"]["scale"]["domain"],
            [0.0, 20.0],
        )
        tooltip_fields = [
            tooltip["field"]
            for tooltip in spec["layer"][0]["encoding"]["tooltip"]
        ]
        self.assertIn("nwb_interface", tooltip_fields)
        self.assertIn("object_store_backend", tooltip_fields)
        self.assertIn("run_status", tooltip_fields)
        self.assertNotIn("rank", tooltip_fields)
        self.assertNotIn("timeout_seconds", tooltip_fields)
        self.assertNotIn("total_seconds_truncated_label", tooltip_fields)

    def test_leaderboard_plot_chart_shows_timeout_seconds_for_timeouts(self) -> None:
        """Timed-out leaderboard bars should include timeout seconds in tooltips."""
        chart = neurodatabench.plots._leaderboard_plot_chart(
            [
                {
                    "implementation_id": "timeout",
                    "nwb_interface": "pynwb",
                    "object_store_backend": "remfile",
                    "benchmark_id": "benchmark",
                    "leaderboard_label": "timeout",
                    "timed_out": True,
                    "total_seconds": 60.0,
                    "timeout_seconds": 60.0,
                }
            ]
        )

        spec = chart.to_dict()
        tooltip_fields = [
            tooltip["field"]
            for tooltip in spec["layer"][1]["encoding"]["tooltip"]
        ]
        self.assertIn("timeout_seconds", tooltip_fields)

    def test_dashboard_title_includes_stack_metadata_when_present(self) -> None:
        """Run dashboard subtitles should include implementation stack metadata."""
        title = neurodatabench.plots._dashboard_title(
            {
                "datetime_utc": "2026-05-13T00:00:00Z",
                "hostname": "host",
                "benchmark_harness_version": "1.2.3",
                "benchmark": {
                    "id": "benchmark",
                    "nwb_format": "hdf5",
                },
                "implementation": {
                    "id": "implementation",
                    "nwb_interface": "pynwb",
                    "object_store_backend": "remfile",
                    "local_cache": None,
                    "remote_cache": False,
                },
            }
        ).to_dict()

        self.assertEqual(
            title["text"],
            "NeuroDataBench: implementation / benchmark",
        )
        self.assertIn("NWB interface pynwb", title["subtitle"])
        self.assertIn("object store remfile", title["subtitle"])


if __name__ == "__main__":
    unittest.main()
