"""Tests for the benchmark matrix helper script."""

from __future__ import annotations

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
                "lazynwb_pre1_obstore_hdf5_dynamic_routing_hdf5_v0_",
                output,
            )
            self.assertIn("-- uv run", output)
            self.assertFalse(output_root.exists())

    def test_pynwb_zarr_rows_are_selectable(self) -> None:
        """PyNWB/HDMF-Zarr matrix rows should cover s3fs and obstore."""
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
        self.assertIn("Starting 2 matrix run(s).", output)
        self.assertIn("pynwb-zarr-v2-s3fs", output)
        self.assertIn("pynwb-zarr-v2-obstore", output)
        self.assertIn("examples/pynwb_zarr_template.py", output)
        self.assertIn("--with zarr<3", output)


if __name__ == "__main__":
    unittest.main()
