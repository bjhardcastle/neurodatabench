"""Tests for the focused dynamic-foraging benchmark matrix."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class DynamicForagingMatrixScriptTests(unittest.TestCase):
    """Exercise selection and commands in the dynamic-foraging matrix."""

    def test_dry_run_contains_all_implementations(self) -> None:
        """The matrix should run both LazyNWB formats and Escapewheel."""
        completed = self._dry_run()
        output = completed.stdout + completed.stderr

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Starting 3 matrix run(s).", output)
        self.assertEqual(
            output.count("implementations/lazynwb/dynamic_foraging.py"),
            2,
        )
        self.assertIn("implementations/escapewheel/dynamic_foraging.py", output)
        self.assertIn("lazynwb-1.0.0dev3-obstore-hdf5-cold", output)
        self.assertIn("lazynwb-1.0.0dev3-obstore-zarr-cold", output)
        self.assertIn("escapewheel-reader", output)
        self.assertEqual(output.count("--with-editable ."), 3)
        self.assertEqual(
            output.count(
                "--benchmark "
                "src/neurodatabench/benchmarks/dynamic_foraging_nwb_v0.json"
            ),
            3,
        )
        self.assertEqual(output.count("--timeout-profile-out"), 3)

    def test_only_selects_escapewheel(self) -> None:
        """The standard matrix selector should support a single reader run."""
        completed = self._dry_run("--only", "escapewheel-reader")
        output = completed.stdout + completed.stderr

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Starting 1 matrix run(s).", output)
        self.assertIn("implementations/escapewheel/dynamic_foraging.py", output)
        self.assertNotIn("implementations/lazynwb/dynamic_foraging.py", output)

    @staticmethod
    def _dry_run(*args: str) -> subprocess.CompletedProcess[str]:
        """Run the matrix script in preview mode and capture its output."""
        return subprocess.run(
            [
                sys.executable,
                "scripts/run_dynamic_foraging_matrix.py",
                "--dry-run",
                *args,
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
