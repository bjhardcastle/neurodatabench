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


if __name__ == "__main__":
    unittest.main()
