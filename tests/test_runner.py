"""Tests for the NeuroDataBench runner harness."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import unittest.mock
import zipfile
from datetime import datetime
from pathlib import Path

import pydantic

import neurodatabench
import neurodatabench.models
import neurodatabench.runner
import neurodatabench.validation


class RunnerTests(unittest.TestCase):
    """Exercise the public runner API and result artifacts."""

    def test_package_root_exposes_public_api(self) -> None:
        """The package root should expose main and model types."""
        self.assertIs(neurodatabench.main, neurodatabench.runner.main)
        self.assertTrue(hasattr(neurodatabench, "models"))
        self.assertTrue(hasattr(neurodatabench, "validation"))
        self.assertIs(neurodatabench.RunContext, neurodatabench.models.RunContext)
        self.assertIs(neurodatabench.Benchmark, neurodatabench.models.Benchmark)
        self.assertIs(neurodatabench.RunPhaseTiming, neurodatabench.models.RunPhaseTiming)
        self.assertIs(
            neurodatabench.BenchmarkValidationError,
            neurodatabench.validation.BenchmarkValidationError,
        )

    def test_main_runs_packaged_benchmark_by_name(self) -> None:
        """A call-first run should load a packaged benchmark and write artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            seen_contexts: list[neurodatabench.models.RunContext] = []

            def setup(context: neurodatabench.models.RunContext) -> None:
                """Record setup context."""
                seen_contexts.append(context)

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit every expected answer from the packaged benchmark."""
                seen_contexts.append(context)
                for question in context.benchmark.questions:
                    context.submit_answer(question.id, question.answer)

            with self.assertLogs("neurodatabench.runner", level="INFO") as logs:
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_nwb_interface="pynwb",
                    implementation_object_store_backend="s3fs",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark="dynamic_routing_zarr_v0",
                    out=tmpdir,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )

            self.assertEqual(len(seen_contexts), 2)
            self.assertIs(seen_contexts[0], seen_contexts[1])
            self.assertEqual(
                seen_contexts[0].benchmark.id,
                "dynamic_routing_zarr_v0",
            )
            self.assertTrue((out_dir / "benchmark.json").exists())
            self.assertTrue((out_dir / "run_metadata.json").exists())
            self.assertTrue((out_dir / "timings.json").exists())
            self.assertTrue((out_dir / "validation.json").exists())
            self.assertTrue((out_dir / "profile_samples.jsonl").exists())
            self.assertTrue((out_dir / "profile_summary.json").exists())
            self.assertTrue((out_dir / "dashboard.html").exists())
            self.assertTrue((out_dir / "implementation.py").exists())
            self.assertFalse((out_dir / "timing_summary.html").exists())
            self.assertFalse((out_dir / "memory_profile.html").exists())
            self.assertFalse((out_dir / "cpu_profile.html").exists())
            self.assertTrue((out_dir / "requirements.txt").exists())
            self.assertTrue((out_dir / "results_bundle.zip").exists())
            self.assertFalse((out_dir / "answers.jsonl").exists())
            self.assertRegex(
                "\n".join(logs.output),
                r"Benchmark run completed in \d+\.\d{3} s",
            )

            validation = _read_json(out_dir / "validation.json")
            self.assertTrue(validation["correct"])
            self.assertNotIn("timed_out", validation)
            metadata = _read_json(out_dir / "run_metadata.json")
            self.assertNotIn("timeout_seconds", metadata)
            self.assertNotIn("timed_out", metadata)
            self.assertEqual(metadata["implementation"]["id"], "test-implementation")
            self.assertEqual(metadata["implementation"]["nwb_interface"], "pynwb")
            self.assertEqual(
                metadata["implementation"]["object_store_backend"],
                "s3fs",
            )
            self.assertEqual(
                metadata["implementation_script"],
                {
                    "artifact": "implementation.py",
                    "source": str(Path(__file__).resolve()),
                },
            )
            profile_summary = _read_json(out_dir / "profile_summary.json")
            self.assertIsInstance(profile_summary["baseline_process_rss_bytes"], int)
            self.assertIsInstance(
                profile_summary["baseline_process_plus_children_rss_bytes"],
                int,
            )
            self.assertIsInstance(profile_summary["peak_process_rss_delta_bytes"], int)
            self.assertIsInstance(
                profile_summary["peak_process_plus_children_rss_delta_bytes"],
                int,
            )
            timings = _read_json(out_dir / "timings.json")
            self.assertEqual(
                [row["phase"] for row in timings["phase_timings"]],
                ["setup", "submit_answers", "total"],
            )
            for row in timings["phase_timings"]:
                self.assertLessEqual(row["start_seconds"], row["stop_seconds"])
                self.assertAlmostEqual(
                    row["duration_seconds"],
                    row["stop_seconds"] - row["start_seconds"],
                )
            submitted_ids = [
                row["question_id"] for row in timings["answer_submissions"]
            ]
            self.assertEqual(
                submitted_ids,
                [question.id for question in seen_contexts[0].benchmark.questions],
            )
            for row in timings["answer_submissions"]:
                datetime.fromisoformat(row["submitted_at"])
                self.assertIsInstance(row["submitted_elapsed_seconds"], float)
                self.assertGreaterEqual(row["submitted_elapsed_seconds"], 0.0)

            with zipfile.ZipFile(out_dir / "results_bundle.zip") as bundle:
                bundle_names = set(bundle.namelist())
            self.assertIn("dashboard.html", bundle_names)
            self.assertIn("implementation.py", bundle_names)
            self.assertNotIn("timing_summary.html", bundle_names)
            self.assertNotIn("memory_profile.html", bundle_names)
            self.assertNotIn("cpu_profile.html", bundle_names)

    def test_local_cache_false_is_rejected(self) -> None:
        """False is not a valid local cache metadata state."""
        with self.assertRaisesRegex(
            ValueError,
            "local_cache must be 'cold', 'warm', or None",
        ):
            neurodatabench.models.Implementation(
                id="test-implementation",
                nwb_interface=None,
                object_store_backend=None,
                local_cache=False,
                remote_cache=False,
            )

    def test_clear_cache_runs_before_profiled_setup(self) -> None:
        """The optional cache hook should run before measured phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "cache-hook.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "cache-hook",
                        [{"id": "q", "text": "Q", "answer": 1}],
                    )
                ),
                encoding="utf-8",
            )
            events: list[str] = []
            seen_contexts: list[neurodatabench.models.RunContext] = []

            def clear_cache(context: neurodatabench.models.RunContext) -> None:
                """Record that cache clearing ran before measured work."""
                events.append("clear_cache")
                seen_contexts.append(context)

            def setup(context: neurodatabench.models.RunContext) -> None:
                """Record setup execution."""
                events.append("setup")
                seen_contexts.append(context)

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                events.append("submit_answers")
                seen_contexts.append(context)
                context.submit_answer("q", 1)

            original_start = neurodatabench.runner._Profiler.start

            def profiler_start(profiler: neurodatabench.runner._Profiler) -> None:
                """Record the profiler boundary before starting it."""
                events.append("profiler_start")
                original_start(profiler)

            with unittest.mock.patch.object(
                neurodatabench.runner._Profiler,
                "start",
                profiler_start,
            ):
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache="cold",
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    out=Path(tmpdir) / "results",
                    setup=setup,
                    clear_cache=clear_cache,
                    submit_answers=submit_answers,
                    argv=(),
                )

            self.assertEqual(
                events,
                ["clear_cache", "profiler_start", "setup", "submit_answers"],
            )
            self.assertTrue(
                all(context is seen_contexts[0] for context in seen_contexts)
            )

    def test_main_runs_filesystem_benchmark_path(self) -> None:
        """A run should load benchmark JSON from a direct filesystem path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "custom.json"
            benchmark_path.write_text(
                json.dumps(_benchmark_json("custom", [{"id": "q", "text": "Q", "answer": 1}])),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                context.submit_answer("q", 1)

            neurodatabench.runner.main(
                implementation_id="test-implementation",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=benchmark_path,
                out=Path(tmpdir) / "results",
                setup=setup,
                submit_answers=submit_answers,
                argv=(),
            )

            validation = _read_json(Path(tmpdir) / "results" / "validation.json")
            self.assertTrue(validation["correct"])

    def test_main_uses_default_output_directory(self) -> None:
        """A run should create a named timestamped result dir when out is omitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "custom.json"
            benchmark_path.write_text(
                json.dumps(_benchmark_json("custom", [{"id": "q", "text": "Q", "answer": 1}])),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                context.submit_answer("q", 1)

            previous_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                neurodatabench.runner.main(
                    implementation_id="default-out-test",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )
            finally:
                os.chdir(previous_cwd)

            result_dirs = sorted(
                path for path in (Path(tmpdir) / "results").iterdir() if path.is_dir()
            )
            self.assertEqual(len(result_dirs), 1)
            self.assertRegex(
                result_dirs[0].name,
                r"^default-out-test_custom_\d{8}T\d{6}Z$",
            )
            self.assertTrue((Path(tmpdir) / "results" / "leaderboard.json").exists())
            self.assertTrue((Path(tmpdir) / "results" / "leaderboard.csv").exists())
            self.assertTrue((Path(tmpdir) / "results" / "leaderboard.html").exists())
            validation = _read_json(result_dirs[0] / "validation.json")
            self.assertTrue(validation["correct"])

    def test_results_leaderboard_updates_for_successful_runs(self) -> None:
        """Runs under a results directory should refresh aggregate leaderboard files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "custom.json"
            benchmark_path.write_text(
                json.dumps(_benchmark_json("custom", [{"id": "q", "text": "Q", "answer": 1}])),
                encoding="utf-8",
            )
            results_dir = Path(tmpdir) / "results"
            (results_dir / "partial-run").mkdir(parents=True)

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def slow_setup(context: neurodatabench.models.RunContext) -> None:
                """Make one run observably slower."""
                time.sleep(0.01)

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                context.submit_answer("q", 1)

            neurodatabench.runner.main(
                implementation_id="fast",
                implementation_nwb_interface="lazynwb",
                implementation_object_store_backend="s3fs",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=benchmark_path,
                out=results_dir / "fast-run",
                setup=setup,
                submit_answers=submit_answers,
                argv=(),
            )
            neurodatabench.runner.main(
                implementation_id="slow",
                implementation_nwb_interface="pynwb",
                implementation_object_store_backend="remfile",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=benchmark_path,
                out=results_dir / "slow-run",
                setup=slow_setup,
                submit_answers=submit_answers,
                argv=(),
            )

            leaderboard = json.loads(
                (results_dir / "leaderboard.json").read_text(encoding="utf-8")
            )
            self.assertIsInstance(leaderboard, list)
            self.assertEqual([row["implementation_id"] for row in leaderboard], ["fast", "slow"])
            self.assertEqual(
                [row["nwb_interface"] for row in leaderboard],
                ["lazynwb", "pynwb"],
            )
            self.assertEqual(
                [row["object_store_backend"] for row in leaderboard],
                ["s3fs", "remfile"],
            )
            self.assertTrue(all("rank" not in row for row in leaderboard))
            self.assertTrue(all("timed_out" not in row for row in leaderboard))
            self.assertTrue(all("timeout_seconds" not in row for row in leaderboard))
            self.assertTrue((results_dir / "leaderboard.csv").exists())
            self.assertTrue((results_dir / "leaderboard.html").exists())
            leaderboard_csv = (results_dir / "leaderboard.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "implementation_id,nwb_interface,object_store_backend,benchmark_id",
                leaderboard_csv,
            )
            self.assertNotIn("rank", leaderboard_csv.splitlines()[0])
            self.assertNotIn("timed_out", leaderboard_csv.splitlines()[0])
            self.assertNotIn("timeout_seconds", leaderboard_csv.splitlines()[0])
            self.assertTrue(
                all("peak_rss_delta_mib" in row for row in leaderboard)
            )

    def test_cli_args_override_call_defaults(self) -> None:
        """Settings CLI overrides should replace benchmark and output defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = Path(tmpdir) / "default.json"
            override_path = Path(tmpdir) / "override.json"
            override_out = Path(tmpdir) / "override-results"
            default_path.write_text(
                json.dumps(_benchmark_json("default", [{"id": "default", "text": "Q", "answer": 0}])),
                encoding="utf-8",
            )
            override_path.write_text(
                json.dumps(_benchmark_json("override", [{"id": "override", "text": "Q", "answer": 7}])),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the answer for the override benchmark."""
                self.assertEqual(context.benchmark.id, "override")
                context.submit_answer("override", 7)

            neurodatabench.runner.main(
                implementation_id="test-implementation",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=default_path,
                out=Path(tmpdir) / "default-results",
                setup=setup,
                submit_answers=submit_answers,
                argv=("--benchmark", str(override_path), "--out", str(override_out)),
            )

            self.assertTrue((override_out / "validation.json").exists())
            self.assertFalse((Path(tmpdir) / "default-results").exists())

    def test_settings_environment_supplies_missing_config(self) -> None:
        """Settings environment variables should supply omitted run config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "env.json"
            out_dir = Path(tmpdir) / "env-results"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json("env", [{"id": "q", "text": "Q", "answer": 1}])
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                context.submit_answer("q", 1)

            with unittest.mock.patch.dict(
                os.environ,
                {
                    "NDB_BENCHMARK": str(benchmark_path),
                    "NDB_OUT": str(out_dir),
                    "NDB_PROFILE_INTERVAL_MS": "10",
                },
            ):
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )

            self.assertTrue((out_dir / "validation.json").exists())
            summary = _read_json(out_dir / "profile_summary.json")
            self.assertEqual(summary["sample_interval_seconds"], 0.01)

    def test_settings_resolves_log_level(self) -> None:
        """Log level should resolve from env, call defaults, and CLI overrides."""
        with unittest.mock.patch.dict(os.environ, {"NDB_LOG_LEVEL": "error"}):
            env_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level=None,
                implementation_id="test-implementation",
                argv=(),
            )
            call_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level="info",
                implementation_id="test-implementation",
                argv=(),
            )
            cli_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level="info",
                implementation_id="test-implementation",
                argv=("--log-level", "debug"),
            )

        self.assertEqual(env_config.log_level, "ERROR")
        self.assertEqual(call_config.log_level, "INFO")
        self.assertEqual(cli_config.log_level, "DEBUG")

    def test_fail_fast_config_resolves_from_call_cli_and_environment(self) -> None:
        """Fail-fast validation should resolve from all settings inputs."""
        with unittest.mock.patch.dict(os.environ, {"NDB_FAIL_FAST": "true"}):
            env_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level=None,
                implementation_id="test-implementation",
                argv=(),
            )
            call_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level=None,
                default_fail_fast=False,
                implementation_id="test-implementation",
                argv=(),
            )
            cli_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level=None,
                default_fail_fast=False,
                implementation_id="test-implementation",
                argv=("--fail-fast",),
            )

        self.assertTrue(env_config.fail_fast)
        self.assertFalse(call_config.fail_fast)
        self.assertTrue(cli_config.fail_fast)

    def test_timeout_config_resolves_from_call_cli_environment_and_no_timeout(
        self,
    ) -> None:
        """Timeout settings should support overrides and explicit disabling."""
        with unittest.mock.patch.dict(
            os.environ,
            {"NDB_TIMEOUT_SECONDS": "30", "NDB_NO_TIMEOUT": "false"},
        ):
            env_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level=None,
                implementation_id="test-implementation",
                argv=(),
            )
            call_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level=None,
                default_timeout_seconds=45,
                implementation_id="test-implementation",
                argv=(),
            )
            cli_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level=None,
                default_timeout_seconds=45,
                implementation_id="test-implementation",
                argv=("--timeout-seconds", "60"),
            )
            disabled_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level=None,
                default_timeout_seconds=45,
                default_no_timeout=True,
                implementation_id="test-implementation",
                argv=(),
            )
            cli_disabled_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level=None,
                implementation_id="test-implementation",
                argv=("--no-timeout",),
            )

        benchmark = neurodatabench.models.Benchmark.model_validate(
            _benchmark_json(
                "timeout-config",
                [{"id": "q", "text": "Q", "answer": 1}],
                timeout_seconds=120,
            )
        )
        self.assertEqual(env_config.timeout_seconds, 30)
        self.assertEqual(call_config.timeout_seconds, 45)
        self.assertEqual(cli_config.timeout_seconds, 60)
        self.assertFalse(env_config.disable_timeout)
        self.assertTrue(cli_disabled_config.disable_timeout)
        self.assertEqual(
            neurodatabench.runner._effective_timeout_seconds(
                config=disabled_config,
                benchmark=benchmark,
            ),
            None,
        )

    def test_timed_out_run_writes_result_artifacts(self) -> None:
        """A timed-out run should write inspectable artifacts and exit cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "timeout.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "timeout",
                        [{"id": "q", "text": "Q", "answer": 1}],
                        timeout_seconds=0.01,
                    )
                ),
                encoding="utf-8",
            )
            results_dir = Path(tmpdir) / "results"
            out_dir = results_dir / "timeout_run"

            def setup(context: neurodatabench.models.RunContext) -> None:
                """Sleep longer than the benchmark timeout."""
                time.sleep(0.05)

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """This should not run after setup times out."""
                context.submit_answer("q", 1)

            with self.assertRaises(SystemExit) as error:
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    out=out_dir,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )

            self.assertEqual(
                str(error.exception),
                "Benchmark timed out at 0.01 seconds",
            )
            self.assertTrue((out_dir / "run_metadata.json").exists())
            self.assertTrue((out_dir / "validation.json").exists())
            self.assertTrue((out_dir / "results_bundle.zip").exists())
            metadata = _read_json(out_dir / "run_metadata.json")
            validation = _read_json(out_dir / "validation.json")
            timings = _read_json(out_dir / "timings.json")
            self.assertEqual(metadata["timeout_seconds"], 0.01)
            self.assertTrue(metadata["timed_out"])
            self.assertFalse(validation["correct"])
            self.assertTrue(validation["timed_out"])
            self.assertGreater(timings["total_duration_ns"], 0)
            leaderboard = json.loads(
                (results_dir / "leaderboard.json").read_text(encoding="utf-8")
            )
            self.assertIsInstance(leaderboard, list)
            self.assertEqual(len(leaderboard), 1)
            self.assertTrue(leaderboard[0]["timed_out"])
            self.assertEqual(leaderboard[0]["run_status"], "timed out")
            self.assertEqual(leaderboard[0]["timeout_seconds"], 0.01)
            self.assertEqual(leaderboard[0]["result_dir"], "timeout_run")
            self.assertTrue((results_dir / "leaderboard.html").exists())

    def test_no_timeout_disables_benchmark_timeout(self) -> None:
        """The runner should allow explicit no-timeout runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "no-timeout.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "no-timeout",
                        [{"id": "q", "text": "Q", "answer": 1}],
                        timeout_seconds=0.01,
                    )
                ),
                encoding="utf-8",
            )
            out_dir = Path(tmpdir) / "results"

            def setup(context: neurodatabench.models.RunContext) -> None:
                """Sleep longer than the benchmark timeout."""
                time.sleep(0.02)

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                context.submit_answer("q", 1)

            neurodatabench.runner.main(
                implementation_id="test-implementation",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=benchmark_path,
                out=out_dir,
                setup=setup,
                submit_answers=submit_answers,
                no_timeout=True,
                argv=(),
            )

            metadata = _read_json(out_dir / "run_metadata.json")
            validation = _read_json(out_dir / "validation.json")
            self.assertNotIn("timeout_seconds", metadata)
            self.assertNotIn("timed_out", metadata)
            self.assertTrue(validation["correct"])
            self.assertNotIn("timed_out", validation)

    def test_invalid_log_level_raises(self) -> None:
        """Invalid log levels should fail during settings validation."""
        with self.assertRaises(pydantic.ValidationError):
            neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_zarr_v0",
                default_out=None,
                default_log_level="verbose",
                implementation_id="test-implementation",
                argv=(),
            )

    def test_mutable_identity_detection_does_not_apply_to_scalars(self) -> None:
        """Validation should flag reused mutable expected objects but not scalars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "identity.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "identity",
                        [
                            {"id": "list", "text": "Q", "answer": [1, 2]},
                            {"id": "int", "text": "Q", "answer": 1},
                        ],
                    )
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit expected answers directly to exercise diagnostics."""
                for question in context.benchmark.questions:
                    context.submit_answer(question.id, question.answer)

            with self.assertLogs("neurodatabench.runner", level="WARNING") as logs:
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    out=Path(tmpdir) / "results-logged",
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )

            self.assertEqual(len(logs.output), 1)
            self.assertIn("list", logs.output[0])

    def test_validation_failures_exit_with_all_errors(self) -> None:
        """Final validation should exit cleanly with every invalid answer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "bad.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "bad",
                        [
                            {"id": "missing", "text": "Q", "answer": 1},
                            {"id": "duplicate", "text": "Q", "answer": 2},
                            {"id": "wrong", "text": "Q", "answer": 3},
                        ],
                    )
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit intentionally invalid answers."""
                context.submit_answer("duplicate", 2)
                context.submit_answer("duplicate", 2)
                context.submit_answer("wrong", 4)
                context.submit_answer("unknown", 5)

            out_dir = Path(tmpdir) / "results"
            with self.assertRaises(SystemExit) as error:
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    out=out_dir,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
            )
            message = str(error.exception)
            self.assertIn(
                "Benchmark answers failed validation:",
                message,
            )
            self.assertIn(
                "- missing: missing_answer; submitted_answer=<missing>; actual_answer=1",
                message,
            )
            self.assertIn(
                "- duplicate: duplicate_answer; submitted_answer=[2, 2]; actual_answer=2",
                message,
            )
            self.assertIn(
                "- wrong: exact_int; submitted_answer=4; actual_answer=3",
                message,
            )
            self.assertIn(
                "- unknown: unknown_question_id; submitted_answer=5; "
                "actual_answer=<unknown_question_id>",
                message,
            )
            self.assertFalse((out_dir / "validation.json").exists())
            self.assertFalse(out_dir.exists())

    def test_validation_errors_include_submitted_and_actual_answers(self) -> None:
        """Validation diagnostics should include the compared answer values."""
        scenarios: list[
            tuple[str, dict[str, object], list[dict[str, object]], list[str]]
        ] = [
            (
                "missing",
                {"id": "missing", "text": "Q", "answer": 1},
                [],
                [
                    "missing: missing_answer",
                    "submitted_answer=<missing>",
                    "actual_answer=1",
                ],
            ),
            (
                "duplicate",
                {"id": "duplicate", "text": "Q", "answer": 2},
                [
                    {"question_id": "duplicate", "answer": 2},
                    {"question_id": "duplicate", "answer": 3},
                ],
                [
                    "duplicate: duplicate_answer",
                    "submitted_answer=[2, 3]",
                    "actual_answer=2",
                ],
            ),
            (
                "wrong",
                {"id": "wrong", "text": "Q", "answer": {"value": [1, 2]}},
                [{"question_id": "wrong", "answer": {"value": [1, 3]}}],
                [
                    "wrong: object_value_value_list_item_1_exact_int",
                    'submitted_answer={"value": [1, 3]}',
                    'actual_answer={"value": [1, 2]}',
                ],
            ),
            (
                "unknown",
                {"id": "known", "text": "Q", "answer": True},
                [
                    {"question_id": "known", "answer": True},
                    {"question_id": "unknown", "answer": False},
                ],
                [
                    "unknown: unknown_question_id",
                    "submitted_answer=false",
                    "actual_answer=<unknown_question_id>",
                ],
            ),
        ]

        for benchmark_id, question, submitted_answers, expected_parts in scenarios:
            with self.subTest(benchmark=benchmark_id):
                benchmark = neurodatabench.models.Benchmark.model_validate(
                    _benchmark_json(benchmark_id, [question])
                )
                with self.assertRaises(
                    neurodatabench.validation.BenchmarkValidationError
                ) as error:
                    neurodatabench.validation.validate_answers(
                        benchmark,
                        submitted_answers,
                    )

                message = str(error.exception)
                for expected_part in expected_parts:
                    self.assertIn(expected_part, message)

    def test_fail_fast_validation_raises_during_submission(self) -> None:
        """Fail-fast mode should validate answers as soon as they are submitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "bad-fast.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "bad-fast",
                        [
                            {"id": "first", "text": "Q", "answer": 1},
                            {"id": "second", "text": "Q", "answer": 2},
                        ],
                    )
                ),
                encoding="utf-8",
            )
            events: list[str] = []

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit one correct answer and then one wrong answer."""
                context.submit_answer("first", 1)
                events.append("after_first")
                context.submit_answer("second", 99)
                events.append("after_wrong")

            out_dir = Path(tmpdir) / "results"
            with self.assertLogs("neurodatabench.runner", level="WARNING") as logs:
                with self.assertRaisesRegex(
                    neurodatabench.runner.BenchmarkValidationError,
                    "second: exact_int; submitted_answer=99; actual_answer=2",
                ):
                    neurodatabench.runner.main(
                        implementation_id="test-implementation",
                        implementation_local_cache=None,
                        implementation_remote_cache=False,
                        benchmark=benchmark_path,
                        out=out_dir,
                        setup=setup,
                        submit_answers=submit_answers,
                        fail_fast=True,
                        argv=(),
                    )

            self.assertEqual(events, ["after_first"])
            self.assertFalse(out_dir.exists())
            self.assertIn("development only", "\n".join(logs.output))

    def test_user_code_failure_does_not_leave_empty_output_directory(self) -> None:
        """A failing implementation should not leave an empty result directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "bad-user-code.json"
            benchmark_path.write_text(
                json.dumps(_benchmark_json("bad-user-code", [{"id": "q", "text": "Q", "answer": 1}])),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Raise before any artifacts can be written."""
                raise RuntimeError("boom")

            out_dir = Path(tmpdir) / "empty-results"
            with self.assertRaisesRegex(RuntimeError, "boom"):
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    out=out_dir,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )
            self.assertFalse(out_dir.exists())

    def test_validation_supports_float_lists_and_objects(self) -> None:
        """Validation should compare supported JSON answer shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "shapes.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "shapes",
                        [
                            {"id": "float", "text": "Q", "answer": 1.0},
                            {"id": "list", "text": "Q", "answer": [1, 2]},
                            {"id": "object", "text": "Q", "answer": {"a": 1}},
                        ],
                    )
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit equivalent answers."""
                context.submit_answer("float", 1.0 + 1e-09)
                context.submit_answer("list", [1, 2])
                context.submit_answer("object", {"a": 1})

            neurodatabench.runner.main(
                implementation_id="test-implementation",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=benchmark_path,
                out=Path(tmpdir) / "results",
                setup=setup,
                submit_answers=submit_answers,
                argv=(),
            )

            validation = _read_json(Path(tmpdir) / "results" / "validation.json")
            self.assertTrue(validation["correct"])

    def test_environment_requirements_are_sorted(self) -> None:
        """The environment requirements artifact should contain sorted package pins."""
        requirements_text = neurodatabench.runner._environment_requirements_text()
        rows = [row for row in requirements_text.splitlines() if row]
        self.assertEqual(rows, sorted(rows, key=str.lower))
        self.assertTrue(all("==" in row for row in rows))


def _benchmark_json(
    benchmark_id: str,
    questions: list[dict[str, object]],
    timeout_seconds: float | None = None,
) -> dict[str, object]:
    """Return a minimal benchmark JSON object."""
    benchmark: dict[str, object] = {
        "id": benchmark_id,
        "nwb_paths": ["file:///tmp/test.nwb"],
        "nwb_format": "hdf5",
        "questions": questions,
    }
    if timeout_seconds is not None:
        benchmark["timeout_seconds"] = timeout_seconds
    return benchmark


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Expected JSON object")
    return value
