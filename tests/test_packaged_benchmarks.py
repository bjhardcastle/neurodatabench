"""Tests for packaged benchmark JSON resources."""

from __future__ import annotations

import json
import unittest
from importlib import resources
from typing import Any

import neurodatabench.benchmarks
import neurodatabench.models


def _packaged_benchmark_paths() -> list[Any]:
    """Return all packaged benchmark JSON resources."""
    benchmark_root = resources.files(neurodatabench.benchmarks)
    return sorted(
        (
            resource
            for resource in benchmark_root.iterdir()
            if resource.name.endswith(".json")
            and resource.is_file()
        ),
        key=lambda resource: resource.name,
    )


class PackagedBenchmarkTests(unittest.TestCase):
    """Validate every packaged benchmark specification."""

    def test_packaged_benchmark_jsons_validate_against_schema(self) -> None:
        """All packaged benchmark JSON files should match the Benchmark schema."""
        benchmark_paths = _packaged_benchmark_paths()

        self.assertGreater(
            len(benchmark_paths),
            0,
            "Expected at least one packaged benchmark JSON resource.",
        )

        for benchmark_path in benchmark_paths:
            with self.subTest(benchmark=benchmark_path.name):
                data = json.loads(benchmark_path.read_text(encoding="utf-8"))
                benchmark = neurodatabench.models.Benchmark.model_validate(data)

                self.assertEqual(
                    benchmark.id,
                    benchmark_path.name.removesuffix(".json"),
                )

    def test_packaged_benchmark_jsons_can_be_loaded(self) -> None:
        """All packaged benchmark JSON files should be readable package resources."""
        for benchmark_path in _packaged_benchmark_paths():
            with self.subTest(benchmark=benchmark_path.name):
                raw_benchmark = benchmark_path.read_text(encoding="utf-8")
                loaded_benchmark: Any = json.loads(raw_benchmark)

                self.assertIsInstance(loaded_benchmark, dict)
