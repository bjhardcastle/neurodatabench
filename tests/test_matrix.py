"""Tests for benchmark matrix orchestration."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import neurodatabench.matrix
import neurodatabench.runner


class MatrixTests(unittest.TestCase):
    """Exercise matrix result classification and status artifacts."""

    def test_timeout_is_recorded_as_benchmark_outcome(self) -> None:
        """A supervised timeout should not be reported as an orchestration failure."""
        run = neurodatabench.matrix.MatrixRun(
            label="slow-reader",
            implementation="implementation.py",
            benchmark="benchmark",
            implementation_id="slow_reader",
        )
        completed = subprocess.CompletedProcess(
            args=["ignored"],
            returncode=neurodatabench.runner.SUPERVISOR_TIMEOUT_EXIT_CODE,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            with (
                unittest.mock.patch.object(
                    neurodatabench.matrix.subprocess,
                    "run",
                    return_value=completed,
                ),
                self.assertLogs("neurodatabench.matrix", level="WARNING") as logs,
            ):
                return_code = neurodatabench.matrix.run_matrix(
                    [run],
                    repo_root=repo_root,
                )

            status_path = repo_root / "results" / "matrix_status.jsonl"
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 0)
        self.assertEqual(status["returncode"], 124)
        self.assertEqual(status["outcome"], "timed_out")
        self.assertIn("Timed out slow-reader.", "\n".join(logs.output))

    def test_non_timeout_failure_still_fails_matrix(self) -> None:
        """Implementation crashes should retain a failing matrix exit status."""
        run = neurodatabench.matrix.MatrixRun(
            label="broken-reader",
            implementation="implementation.py",
            benchmark="benchmark",
            implementation_id="broken_reader",
        )
        completed = subprocess.CompletedProcess(args=["ignored"], returncode=2)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            unittest.mock.patch.object(
                neurodatabench.matrix.subprocess,
                "run",
                return_value=completed,
            ),
            self.assertLogs("neurodatabench.matrix", level="ERROR"),
        ):
            return_code = neurodatabench.matrix.run_matrix(
                [run],
                repo_root=Path(tmpdir),
            )

        self.assertEqual(return_code, 1)


if __name__ == "__main__":
    unittest.main()
