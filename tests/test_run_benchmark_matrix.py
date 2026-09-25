"""Tests for the benchmark matrix helper script."""

from __future__ import annotations

import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BenchmarkMatrixScriptTests(unittest.TestCase):
    """Exercise script behavior that affects generated matrix artifacts."""

    def test_dry_run_does_not_write_status_jsonl(self) -> None:
        """Dry-run previews commands without creating status output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / "nested" / "matrix_status.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_benchmark_matrix.py",
                    "--dry-run",
                    "--limit",
                    "1",
                    "--status-jsonl",
                    str(status_path),
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(status_path.exists())
            self.assertFalse(status_path.parent.exists())

    def test_timeout_flag_targets_runner_supervisor_not_helper(self) -> None:
        """Matrix timeouts should wrap the uv command instead of helper main()."""
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark_matrix.py",
                "--dry-run",
                "--limit",
                "1",
                "--timeout-seconds",
                "12",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("neurodatabench.runner supervise", output)
        self.assertIn("--timeout-seconds 12", output)
        self.assertIn("-- uv run", output)
        self.assertNotIn("uv run --timeout-seconds", output)

    def test_timeout_disabled_flag_is_forwarded(self) -> None:
        """The timeout-disabled flag should bypass the supervisor."""
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark_matrix.py",
                "--dry-run",
                "--limit",
                "1",
                "--timeout-disabled",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            f"DRY RUN: uv run --python {sys.version_info.major}.{sys.version_info.minor}",
            output,
        )
        self.assertNotIn("neurodatabench.runner supervise", output)
        self.assertIn("implementations/lazynwb_template.py", output)

    def test_timeout_enabled_by_default(self) -> None:
        """Omitting timeout-disabled should keep the supervisor enabled."""
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark_matrix.py",
                "--dry-run",
                "--limit",
                "1",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("neurodatabench.runner supervise", output)

    def test_timeout_disabled_rejects_explicit_value(self) -> None:
        """The timeout-disabled option should be a presence-only flag."""
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark_matrix.py",
                "--dry-run",
                "--limit",
                "1",
                "--timeout-disabled",
                "true",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)

    def test_snake_case_matrix_args_are_accepted(self) -> None:
        """Matrix inputs should accept snake case as well as canonical kebab case."""
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark_matrix.py",
                "--dry_run",
                "--limit",
                "1",
                "--profile_interval_ms",
                "250",
                "--timeout_disabled",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--profile-interval-ms 250", output)
        self.assertNotIn("neurodatabench.runner supervise", output)

    def test_out_flag_sets_per_run_storage_root(self) -> None:
        """Matrix --out should forward unique helper output directories under a root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "matrix-results"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_benchmark_matrix.py",
                    "--dry-run",
                    "--limit",
                    "1",
                    "--out",
                    str(output_root),
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(f"--out {output_root}", output)
            self.assertIn(
                str(
                    Path("dynamic_routing_nwb_hdf5_v0")
                    / "lazynwb_0_2_91_obstore_hdf5"
                ),
                output,
            )
            self.assertIn("--timeout-profile-out", output)
            self.assertIn("-- uv run", output)
            self.assertFalse(output_root.exists())

    def test_pynwb_zarr_row_is_selectable(self) -> None:
        """The PyNWB/HDMF-Zarr matrix row should use its real s3fs backend."""
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark_matrix.py",
                "--dry-run",
                "--only",
                "pynwb-zarr",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Starting 1 matrix run(s).", output)
        self.assertIn("pynwb-zarr-v2-s3fs", output)
        self.assertNotIn("pynwb-zarr-v2-obstore", output)
        self.assertIn("implementations/pynwb_zarr_template.py", output)
        self.assertIn("--with zarr<3", output)

    def test_roi_zarr_matrix_has_direct_and_cached_rows(self) -> None:
        """The remote ROI matrix should expose each reader and lazyNWB variants."""
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark_matrix.py",
                "--dry-run",
                "--only",
                "roi-",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Starting 16 matrix run(s).", output)
        self.assertIn("roi-zarr-s3fs", output)
        self.assertIn("roi-zarr-obstore", output)
        self.assertIn("roi-virtualizarr-cold", output)
        self.assertIn("roi-virtualizarr-warm", output)
        self.assertIn("roi-icechunk-cold", output)
        self.assertIn("roi-icechunk-warm", output)
        self.assertIn("implementations/roi_virtualizarr_template.py", output)
        self.assertIn("implementations/roi_icechunk_template.py", output)
        self.assertIn("roi-lazynwb-v0.2.91-s3fs-cold", output)
        self.assertIn("roi-lazynwb-v0.2.91-obstore-cold", output)
        self.assertIn("roi-lazynwb-v0.2.91-remfile-cold", output)
        self.assertIn("roi-lazynwb-1.0.0dev3-s3fs-cold", output)
        self.assertIn("roi-lazynwb-dev-s3fs-cold", output)
        self.assertIn("--benchmark multiplane_ophys_roi_zarr_v0", output)

    def test_roi_zarr_matrix_reuses_cache_between_cold_and_warm_rows(self) -> None:
        """Cold and warm cache rows should resolve to one shared matrix cache."""
        import neurodatabench.matrix

        runs = [
            run
            for run in runpy.run_path(
                "scripts/run_benchmark_matrix.py",
                run_name="matrix_test_module",
            )["default_matrix"]()
            if run.implementation_id
            in {"roi-virtualizarr-cold", "roi-virtualizarr-warm"}
        ]
        self.assertEqual(len(runs), 2)
        cold_environment = neurodatabench.matrix._environment_for(
            runs[0], output_root=Path("results")
        )
        warm_environment = neurodatabench.matrix._environment_for(
            runs[1], output_root=Path("results")
        )
        self.assertEqual(
            cold_environment["NDB_ZARR_CACHE_PATH"],
            warm_environment["NDB_ZARR_CACHE_PATH"],
        )

    def test_roi_lazynwb_matrix_reuses_cache_between_cold_and_warm_rows(self) -> None:
        """Stable and development lazyNWB rows should each share their cache pair."""
        import neurodatabench.matrix

        runs = [
            run
            for run in runpy.run_path(
                "scripts/run_benchmark_matrix.py",
                run_name="matrix_test_module",
            )["default_matrix"]()
            if run.implementation_id.startswith("roi-lazynwb-")
        ]
        self.assertEqual(len(runs), 10)
        for prefix in (
            "roi-lazynwb-v0.2.91-obstore",
            "roi-lazynwb-v0.2.91-remfile",
            "roi-lazynwb-v0.2.91-s3fs",
            "roi-lazynwb-1.0.0dev3-s3fs",
            "roi-lazynwb-dev-s3fs",
        ):
            pair = [run for run in runs if run.implementation_id.startswith(prefix)]
            self.assertEqual(len(pair), 2)
            paths = {
                neurodatabench.matrix._environment_for(
                    run,
                    output_root=Path("results"),
                )["NDB_LAZYNWB_CACHE_PATH"]
                for run in pair
            }
            self.assertEqual(len(paths), 1)

    def test_default_matrix_omits_unsupported_ros3_rows(self) -> None:
        """The self-contained matrix should not require a nonstandard h5py build."""
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark_matrix.py",
                "--dry-run",
                "--only=ros",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("No matrix runs selected.", output)

    def test_default_matrix_omits_parquet_components_row(self) -> None:
        """The default matrix should omit the parquet component implementation."""
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark_matrix.py",
                "--dry-run",
                "--only",
                "parquet-components",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("No matrix runs selected.", output)

    def test_implementation_scripts_have_closed_inline_metadata(self) -> None:
        """Every implementation script should contain validly delimited inline metadata."""
        implementations_dir = Path(__file__).resolve().parents[1] / "implementations"
        for script_path in implementations_dir.glob("*.py"):
            with self.subTest(script=script_path.name):
                script = script_path.read_text(encoding="utf-8")
                markers = re.findall(r"^# ///(?: script)?$", script, flags=re.MULTILINE)
                self.assertEqual(markers, ["# /// script", "# ///"])

    def test_runnable_scripts_install_neurodatabench_from_github(self) -> None:
        """Benchmark processes should not shadow the declared GitHub dependency."""
        repo_root = Path(__file__).resolve().parents[1]
        script_paths = [
            repo_root / "scripts" / "run_benchmark_matrix.py",
            *(repo_root / "implementations").glob("*.py"),
        ]

        for script_path in script_paths:
            with self.subTest(script=script_path.name):
                script = script_path.read_text(encoding="utf-8")
                self.assertIn(
                    'neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }',
                    script,
                )
                self.assertNotIn("sys.path.insert", script)


if __name__ == "__main__":
    unittest.main()
