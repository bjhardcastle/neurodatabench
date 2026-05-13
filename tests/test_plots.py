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
                "rank": 1,
                "implementation_id": "fast",
                "nwb_interface": "lazynwb",
                "object_store_backend": "s3fs",
                "benchmark_id": "benchmark",
                "leaderboard_label": "1. fast",
                "total_seconds": 2.5,
            },
            {
                "rank": 2,
                "implementation_id": "slow",
                "nwb_interface": "pynwb",
                "object_store_backend": "remfile",
                "benchmark_id": "benchmark",
                "leaderboard_label": "2. slow",
                "total_seconds": 24.25,
            },
        ]

        plot_rows = neurodatabench.plots._leaderboard_plot_rows(rows)

        self.assertEqual(plot_rows[0]["plot_total_seconds"], 2.5)
        self.assertFalse(plot_rows[0]["total_seconds_truncated"])
        self.assertEqual(plot_rows[0]["total_seconds_truncated_label"], "no")
        self.assertEqual(plot_rows[1]["total_seconds"], 24.25)
        self.assertEqual(plot_rows[1]["plot_total_seconds"], 20.0)
        self.assertTrue(plot_rows[1]["total_seconds_truncated"])
        self.assertEqual(plot_rows[1]["total_seconds_truncated_label"], "yes")
        self.assertEqual(plot_rows[1]["plot_total_seconds_label"], "> 20 s")

    def test_leaderboard_plot_chart_caps_twenty_second_axis(self) -> None:
        """Leaderboard chart axis should cap runtimes at twenty seconds."""
        chart = neurodatabench.plots._leaderboard_plot_chart(
            [
                {
                    "rank": 1,
                    "implementation_id": "slow",
                    "nwb_interface": "pynwb",
                    "object_store_backend": "remfile",
                    "benchmark_id": "benchmark",
                    "leaderboard_label": "1. slow",
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
