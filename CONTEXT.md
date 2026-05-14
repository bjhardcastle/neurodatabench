# NWB benchmark implementation

## Current Notes

* `examples/lazynwb_template.py` is runnable from the repo checkout with `uv run examples/lazynwb_template.py --out <dir>` by prepending local `src/` before importing `neurodatabench`. It depends on altair, `lazynwb[pynwb]==1.0.0dev2`, numpy, polars, and psutil in its inline script metadata.
* The lazynwb example imports lazynwb/numpy/polars at module scope, reads NWB paths from each hook's `RunContext` instead of storing a module-level path copy, keeps the per-question lazynwb calls inline in `submit_answers()` instead of hiding them behind question helper functions, points `LAZYNWB_CATALOG_CACHE_PATH` at a fresh temp `catalog.sqlite` path in `setup()` and `clear_cache()`, configures lazynwb in `setup()`, and answers the benchmark questions directly.
* The lazynwb template fetches `spike_times` for the selected unit through `lazynwb.tables.get_df(...)`.
* `examples/pynwb_zarr_template.py` is a quick PyNWB/HDMF-Zarr dependency-stack implementation for the same packaged benchmark. A full `NWBZarrIO(...).read()` on the public S3 stores was too slow for a quick baseline, so the example reads the needed DynamicTable Zarr arrays directly (`/units`, `/intervals/trials`) while keeping the PyNWB/HDMF-Zarr dependencies explicit. The direct spike-times helper mirrors PyNWB's `nwbfile.units.get_unit_spike_times(index)` ragged-column lookup without requiring full `NWBFile` materialization. It validated successfully with `uv run examples/pynwb_zarr_template.py --out /tmp/neurodatabench-pynwb-zarr-check --profile-interval-ms 1000` in about 24 seconds.
* `examples/direct_h5py_template.py` mirrors the direct-array PyNWB Zarr template for `dynamic_routing_hdf5_v0` by reading the needed HDF5 groups with `h5py` through `remfile.File` objects. The benchmark's public `s3://bucket/key` paths are converted to virtual-hosted S3 HTTPS URLs before opening because remfile expects HTTP(S). This used to live at `examples/pynwb_hdf5_template.py`, but it was renamed because it does not materialize PyNWB `NWBFile` objects. It previously validated successfully in about 13 seconds and can now be rerun with `uv run examples/direct_h5py_template.py --fail-fast --log-level INFO`; an earlier raw `s3fs` file-object version was slower because random HDF5 metadata/table access pulled large readahead blocks.
* `examples/pynwb_hdf5_template.py` is now the intentionally slow true PyNWB HDF5 baseline. Its setup opens every remote NWB path through a local `remfile.File` -> `h5py.File` -> `pynwb.NWBHDF5IO` chain, stores per-file state records around the resulting `pynwb.NWBFile` objects, and lazily adds the PyNWB units/trials tables and DataFrames to those records the first time a question needs them. PyNWB's namespace-loading path expects an `h5py.File`, so do not pass a raw `remfile.File` directly to `NWBHDF5IO(file=...)`. It excludes `spike_times` from the units DataFrame scans, then calls the cached units table's `get_unit_spike_times(...)` for the selected fastest VISp unit.
* Implementation examples prioritize readability, directness, and fair benchmark behavior over complete static typing. Do not add helper functions, casts, or abstractions solely to satisfy type checkers in example scripts.
* `dynamic_routing_zarr_v0.json` expected answers were replaced with values computed by the lazynwb template from the current public S3 Zarr data: `45`, `2.461157454828708`, and `5.529684329819269`.
* Validation errors now include both `submitted_answer=...` and `actual_answer=...` diagnostics for missing, duplicate, mismatched, and unknown-question submissions.
* Answer validation lives in `neurodatabench.validation`; `runner.py` calls it after timed execution by default. End-of-run validation aggregates all missing, duplicate, mismatched, and unknown-question failures into one clean `SystemExit` message so user scripts show diagnostics without a traceback. `main(fail_fast=True)`, CLI `--fail-fast`, or `NDB_FAIL_FAST=true` validates each submitted answer immediately and logs a warning that this mode is for development only, not benchmark runs.
* Successful runner executions log total/setup/submit-answer duration at INFO level. The default runner log level is INFO so finish timing is visible without a flag.
* Benchmarks may define `timeout_seconds` as the default fair-run wall-clock budget. `main(timeout_seconds=None)` means no runner override; `main(timeout_seconds=...)`, CLI `--timeout-seconds`, or `NDB_TIMEOUT_SECONDS` overrides the benchmark default. `main(no_timeout=True)`, CLI `--no-timeout`/`--disable-timeout`, `NDB_NO_TIMEOUT=true`, or `NDB_DISABLE_TIMEOUT=true` explicitly disables timeout enforcement. Successful result metadata and validation omit timeout fields; timeout exits store `timed_out=true` and the effective `timeout_seconds`, and write inspectable artifacts with `validation.correct=false`.
* Timeout enforcement uses an internal `BaseException` sentinel so broad dependency or implementation `except Exception`/`except TimeoutError` handlers cannot swallow the alarm before the runner writes timeout artifacts and refreshes the leaderboard.
* `timings.json` includes explicit `phase_timings` start/stop/duration records and per-answer `submitted_elapsed_seconds`, so the dashboard does not need to reverse-engineer the run structure.
* Successful result directories copy the inferred implementation hook source to `implementation.py`, record its original path in `run_metadata.json`, and include it in `results_bundle.zip`. Callers can pass `implementation_script=...`, `--implementation-script`, or `NDB_IMPLEMENTATION_SCRIPT` when the hook source is not the desired top-level script.
* Each successful result directory includes one Altair HTML dashboard, `dashboard.html`, with timing, memory, CPU, and system network panes sharing one elapsed-time domain. Profile panes use absolute units: process+children RSS in MiB, process+children/system CPU in percent, and system network received in MiB since first sample. The dashboard title uses run metadata, including implementation ID, benchmark ID, timestamp, host, harness version, format, NWB interface, object-store backend, and cache metadata. The timing pane omits the redundant total bar and uses thin phase bars; setup completion and answer submission annotations are hover-only vertical markers on the profile panes. Dashboard colors use a named palette in `neurodatabench.plots` that avoids a blue-heavy visual range. The dashboard artifact is included in `results_bundle.zip`, and dashboard rendering lives in `neurodatabench.plots`.
* Complete runs written under a directory named `results` refresh aggregate leaderboard artifacts in that parent directory: `leaderboard.json`, `leaderboard.csv`, and `leaderboard.html`. Leaderboard rows are built from complete correct runs and timed-out runs, sorted by benchmark ID and total runtime without materialized rank fields. Leaderboard JSON/CSV rows include implementation ID, NWB interface, object-store backend, NWB format, cache metadata, run status, timing, memory, and network summaries; `timed_out` and `timeout_seconds` are included only for timed-out rows. The leaderboard plot caps visible bars at 20 seconds, labels capped bars with their actual duration, groups by benchmark/time ordering, marks timed-out rows, and keeps exact totals in the tooltip/JSON/CSV.
* Leaderboard plotting normalizes the row values it depends on, so CSV-like strings for `timed_out`, `run_status`, `total_seconds`, and `timeout_seconds` still produce visible timed-out bars with numeric tooltips.
* `neurodatabench.runner.main()` now accepts dedicated implementation metadata fields (`implementation_id`, `implementation_nwb_interface`, `implementation_object_store_backend`, `implementation_local_cache`, `implementation_remote_cache`) instead of requiring callers to construct `models.Implementation`. If `out` and `--out` are omitted, the runner writes to a timestamped directory named `results/<implementation_id>_<benchmark>_<YYYYmmddTHHMMSSZ>`.
* Runner configuration now uses `pydantic-settings` rather than `argparse`. Call defaults can be overridden by Settings CLI flags such as `--benchmark`, `--out`, `--profile-interval-ms`, and `--log-level`, while missing values can come from `NDB_` environment variables such as `NDB_BENCHMARK`, `NDB_OUT`, and `NDB_LOG_LEVEL`.
* `RunContext` is owned by `neurodatabench.models`; examples and templates should annotate hook contexts through the package root as `neurodatabench.RunContext`, not `neurodatabench.runner.RunContext`.
* The public user API is available from the package root: prefer `import neurodatabench`, `neurodatabench.main(...)`, root model types such as `neurodatabench.RunContext`, and `neurodatabench.models` when the module namespace is useful.
* `neurodatabench.__version__` is read from installed package metadata via `importlib.metadata`, so `pyproject.toml` remains the single version source.
* Successful result directories write environment package pins to `requirements.txt`. The file uses requirements-style `name==version` lines, not a mixed `*.lock.txt` naming convention.
* `scripts/run_benchmark_matrix.py` orchestrates the current example helpers across lazynwb pre-1.0/1.0.0dev3, HDF5/Zarr benchmarks, storage backend labels, direct h5py, direct Zarr v2/v3 for Zarr-capable backends (`s3fs`, `obstore`), and true PyNWB HDF5 combinations. It forwards per-run metadata through `NDB_BENCHMARK`, `NDB_IMPLEMENTATION_ID`, `NDB_OBJECT_STORE_BACKEND`, `NDB_LOCAL_CACHE`, and optional Zarr pins, appending JSONL status to `results/matrix_status.jsonl` by default for real runs only. Lazynwb matrix entries get stable cache DB paths under `results/matrix_caches/`, so matching 1.0 cold/warm runs share a cache path. Run `uv run python scripts/run_benchmark_matrix.py --dry-run` to inspect the full command set without writing status.
* The dynamic routing HDF5 and Zarr packaged benchmarks currently use a 90-second fair-run timeout while matrix failures are being separated into slow scanning, file-opening timeout, and backend compatibility categories.
* With the 90-second timeout, the lazynwb pre-1.0 HDF5 remfile and s3fs matrix rows complete, while the obstore row still times out. Debug logs show obstore is not stuck before opening: it spends tens of seconds retrieving `/units` and `/intervals/trials` accessors in setup, then times out during the first answer collection.
* Lazynwb 0.2.90 Zarr failures appear to be anonymous S3 configuration loss, not fundamental lack of Zarr support. `lazynwb.file_io._open_file()` builds a `UPath` with resolved storage options, but the Zarr fallback calls `zarr.open(u.as_posix(), mode="r")`, dropping those options; debug logs then show signed `s3fs` `HeadObject` requests failing with `Unable to locate credentials`.
* `examples/direct_h5py_template.py` should open HDF5 over obstore through `obstore.fsspec.FsspecStore`, not `obstore.open_reader()`. `open_reader().read()` returns `obstore.Bytes`, which h5py rejects; the fsspec bridge returns built-in `bytes` and the direct h5py obstore row validated in about 56 seconds.
* Lazynwb 1.0/dev matrix runs must not receive `LAZYNWB_USE_OBSTORE` or `LAZYNWB_USE_REMFILE`; 1.0 implicitly uses obstore, and forcing those environment settings made HDF5 runs jump from about 5-6 seconds to about 13-15 seconds. Keep those backend-toggle env vars scoped to `examples/lazynwb_v0.py`.


## Design decisions

* Single user-edited script executes the benchmark.
* Packaged benchmark JSON resources are covered by unittest discovery and validated through the `Benchmark` Pydantic dataclass schema.
* Runtime code and tests should remain compatible with Python 3.10+.
* Runtime harness imports stay inside `if __name__ == "__main__":`.
* Implementation import time is not explicitly timed or recorded.
* Setup is timed separately and included in top-line measured total time.
* `answer_questions()` duration is timed separately from setup.
* Top-line `total_duration_ns` measures from immediately before `setup()` through `answer_questions()` completion.
* Per-question timings are not measured by the harness when the implementation answers the full list.
* By default, validation happens only after setup, answer, and total timing finish.
* Fail-fast validation is an opt-in development aid only; do not use it for benchmark runs because it checks answers during timed submission.
* `questions.json` contains dataset paths and answers.
* Do not pass expected answers to implementation.
* `dataset_paths` appears once at the top of `questions.json` as a list of local paths or remote URIs.
* There is no separate dataset manifest and no dataset ID indirection.
* Dataset files do not carry asset IDs, checksums, byte sizes, or separate URI fields.
* Question files do not require `schema_version`; add a format version only if the runner later supports multiple incompatible question-file shapes.
* Canonical benchmark questions are distributed as package data under `src/neurodatabench/questions/`.
* Packaged question filenames are benchmark names, e.g. `visual_coding_v1.json`, so installed users can run without locating repo files.
* No `order` field; list order is canonical.
* No `depends_on`.
* No `input_files`.
* No validation DSL in `questions.json`; validation behavior is defined by the harness.
* Float comparisons use `numpy.isclose` defaults.
* Answers may be JSON scalars, lists, or objects. Lists are valid when a question expects ordered multiple values.
* The implementation receives the full ordered question list without expected answers, fills each question's `answer` field, and returns the completed list.
* Prefer an explicit answer-submission interface over implicit mutation of question objects; the current `question.answer = ...` sketch is a placeholder if a clearer sink/callback API is adopted.
* A localhost answer server launched by `main()` is useful as a future out-of-process or cross-language adapter, but it is too much ceremony for the default in-process Python template.
* Default Python submissions should use an explicit in-process answer sink, e.g. `submit_answer(question_id, answer)`, so answers are visible and intentional without adding ports, background threads, request serialization, or server lifecycle failure modes.
* If answers are submitted through a sink, do not require canonical answer order. Capture each submitted answer's timestamp/order as observational detail, validate by question ID, and treat total measured time as the primary comparable metric. Per-answer timing/order is bonus diagnostic data that usually requires implementation context to interpret fairly.
* Submission state is module-owned. `setup()` must reset module-level state for each measured run; the runner must not inspect or pass state between hooks.
* `RunContext` carries dataset paths, output directory, scratch directory, and input provenance needed by implementation hooks.
* Use full type hints throughout the harness, submission template, tests, and packaging code.
* Keep structured models sparse. Prefer plain JSON objects for question specs and metadata records; use dataclasses only where they make the hook/result contract clearer.
* Do not write run status or captured error artifacts. User-code exceptions should bubble, and validation correctness is the pass/fail signal.
* Pydantic model creation has measurable overhead; avoid Pydantic in the user implementation top level or timed setup/answer path.
* The user script must not define a top-level `IMPLEMENTATION` metadata object. Pass implementation metadata as typed arguments to `main()` inside `if __name__ == "__main__":`.
* `main(questions=..., out=...)` can provide script-level defaults.
* CLI `--questions` and `--out` are mutually exclusive with matching `main()` defaults; raise a clear error if both are set, because silent overrides make benchmark provenance ambiguous.
* Dataset paths are available through `RunContext`.
* Local cache metadata is declared as `"cold"`, `"warm"`, or `None`; use `None` rather than `False` when an implementation has no local cache. `clear_cache()` is only an optional hook for clearing implementation-managed local caches.
* `clear_cache()` is untimed and called only before the first measured run; it is not called between repeated runs.
* No `pip freeze`.
* Environment package snapshots are written as requirements-style `requirements.txt` artifacts when captured.
* Use uv inline script metadata for declared dependencies.
* Harness records platform, datetime, Python, key package versions, CPU, memory, disk, network counters, profiling samples.
* Implementer declares only:

  * implementation ID
  * NWB interface/API, if any
  * object-store backend, if any
  * local and remote cache metadata

---

# Repo skeleton

```text
pyproject.toml
src/
  neurodatabench/
    __init__.py
    models.py
    runner.py
    schemas.py
    metadata.py
    profiling.py
    validation.py
    packaging.py
    questions/
      visual_coding_v1.json
templates/
  submission_template.py
tests/
  test_validation.py
  test_inline_metadata.py
  test_runner_smoke.py
```

---

# Package Data

Questions must be included in wheels and sdists. The exact `pyproject.toml`
syntax depends on the build backend, but the package-data contract is:

```toml
[tool.setuptools.package-data]
neurodatabench = ["py.typed", "questions/*.json"]
```

Rules:

```text
Packaged question resource names are benchmark names.
The runner accepts either a filesystem path or a packaged question name.
Example: --questions visual_coding_v1
Submission scripts may pass `questions="visual_coding_v1"` to `main()` as a default.
Raise an error if both CLI `--questions` and `main(questions=...)` are set.
The local smoke test should exercise packaged question resources, not only repo-relative files.
```

---

# Step-by-step implementation checklist

## 1. Define schemas

### `questions.json`

```json
{
  "benchmark_id": "nwb_access_benchmark_v1",
  "dataset_paths": [
    "/data/nwb/session_001.nwb"
  ],
  "questions": [
    {
      "id": "q001_units_VISp_default_qc",
      "text": "How many units are in VISp where default_qc is True?",
      "answer": 123
    },
    {
      "id": "q002_mean_firing_rate_previous_units",
      "text": "For the units identified previously, what is the mean firing rate?",
      "answer": 4.82
    },
    {
      "id": "q003_first_five_unit_ids",
      "text": "What are the first five unit IDs after applying the same filters?",
      "answer": [101, 204, 205, 301, 455]
    }
  ]
}
```

Rules:

```text
questions[*].answer defines expected type.
No answer_type.
Answers may be JSON scalars, lists, or objects.
No validation config or validation DSL in questions.json.
Floats are compared with numpy.isclose defaults.
No order.
No depends_on.
No input_files.
Dataset paths are local paths or remote URIs.
No dataset_id.
No separate dataset manifest.
No dataset file asset_id, uri, sha256, or size_bytes fields.
```

---

## 2. Submission template

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "neurodatabench",
#   "pynwb",
#   "h5py",
#   "numpy",
#   "psutil"
# ]
# ///

"""Submission template for a NeuroDataBench benchmark implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurodatabench.models import (
        QuestionForImplementation,
        RunContext,
    )


state: dict[str, object] | None = None

# Implementation dependency imports go here.
import numpy as np
import pynwb


def setup(context: RunContext) -> None:
    """
    Timed separately and included in total_duration_ns.

    The harness provides typed run context with dataset path, output directory,
    and scratch directory. Reset module-level state for this measured run.
    """
    global state

    state = {
        "nwb_file": pynwb.read_nwb(context.dataset_paths[0]),
        "scratch_dir": context.scratch_dir,
    }


def clear_cache(context: RunContext) -> None:
    """
    Optional. Untimed.

    Clear implementation-managed local caches before the first measured run.
    The harness must not call this between repeated runs.
    """
    return None


def answer_questions(
    questions: list[QuestionForImplementation],
) -> list[QuestionForImplementation]:
    """
    Timed separately and included in total_duration_ns.

    Questions do NOT contain expected answers.
    Question order is canonical.
    The harness calls this function once for the full ordered list, so local
    variables are fine for transient work. Use module-level state for setup
    artifacts, intermediates reused by later questions, or values needed by
    teardown. May mutate state and question.answer fields.
    Must return the completed question list.
    """
    current_state = state
    if current_state is None:
        raise RuntimeError("setup() must run before answer_questions()")

    for question in questions:
        qid = question.id

        if qid == "q001_units_VISp_default_qc":
            filtered_unit_ids = [101, 204, 205, 301, 455] + list(range(1000, 1118))
            current_state["filtered_unit_ids"] = filtered_unit_ids
            question.answer = len(filtered_unit_ids)
            continue

        if qid == "q002_mean_firing_rate_previous_units":
            filtered_unit_ids = current_state["filtered_unit_ids"]
            if filtered_unit_ids is None:
                raise RuntimeError("q002 requires q001 to run first")

            mean_firing_rate = 4.82
            current_state["mean_firing_rate"] = mean_firing_rate
            question.answer = mean_firing_rate
            continue

        if qid == "q003_first_five_unit_ids":
            filtered_unit_ids = current_state["filtered_unit_ids"]
            if filtered_unit_ids is None:
                raise RuntimeError("q003 requires q001 to run first")

            question.answer = filtered_unit_ids[:5]
            continue

        raise NotImplementedError(qid)

    return questions


def teardown() -> None:
    """
    Untimed.
    """
    global state

    state = None
    return None


if __name__ == "__main__":
    from neurodatabench.runner import main

    main(
        implementation_name="replace-me",
        implementation_cache_type="cold",
        implementation_notes="",
        questions="visual_coding_v1",
        out="results",
        clear_cache=clear_cache,
        setup=setup,
        answer_questions=answer_questions,
        teardown=teardown,
    )
```

---

## 3. CLI

```bash
python submission.py run \
  --questions visual_coding_v1 \
  --out results
```

```bash
python submission.py smoke \
  --questions visual_coding_v1
```

CLI options:

```text
run
smoke
--questions PATH_OR_PACKAGED_NAME
--out DIR
--profile-interval-ms 250
```

`--questions` and `--out` may be omitted only when the submission script passes
`questions=...` and `out=...` to `main()`. If both the CLI and `main()` provide
the same setting, raise a clear error instead of choosing one.

---

## 4. Typed models

```python
# src/neurodatabench/models.py

"""Typed data structures for NeuroDataBench inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class RunContext:
    """Runner-provided context passed to implementation hooks."""

    dataset_paths: list[str]
    out_dir: Path
    scratch_dir: Path
    questions_source: str
    submission_path: Path


@dataclass(slots=True)
class QuestionForImplementation:
    """Question passed to user code; implementation fills answer."""

    id: str
    text: str
    answer: JsonValue | None = None


@dataclass(frozen=True, slots=True)
class SubmittedAnswer:
    """One normalized answer returned by the implementation."""

    index: int
    question_id: str
    answer: JsonValue | None


@dataclass(frozen=True, slots=True)
class RunTimings:
    """Timing summary for a run."""

    setup_duration_ns: int | None
    answer_questions_duration_ns: int | None
    total_duration_ns: int | None
    setup_timed: bool = True


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Validation outcome for one question."""

    index: int
    question_id: str
    correct: bool
    reason: str
    actual_question_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Validation summary for all questions."""

    correct: bool
    num_questions: int
    num_submitted: int
    num_correct: int
    results: list[ValidationResult]


```

Rules:

```text
Use dataclasses for public hook/result records that benefit from typed attributes.
Keep question specs, run metadata, inline script metadata, and profiler records as plain JSON objects.
Do not create run status or captured error artifacts.
Avoid Pydantic model creation in the submission module top level and timed setup/answer path.
```

---

## 5. Runner lifecycle

```text
parse args
load questions
validate schema shape
build run context
prepare implementation questions with blank answer fields
collect pre-run metadata
if first measured run and clear_cache is defined:
    call clear_cache(context)       # untimed
start profiler
start total timer
start setup timer
call setup(context)
stop setup timer
start answer_questions timer
call answer_questions(questions)
stop answer_questions timer
stop total timer
normalize answered questions
stop profiler
call teardown()                     # untimed
write answers
write timings
validate answers                    # after timing only
write metadata
write profile summary
package result bundle
if validation.correct is false:
    exit nonzero
```

---

## 6. Runner skeleton

```python
# src/neurodatabench/runner.py

"""Command-line runner for executing one NeuroDataBench submission."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

from neurodatabench.metadata import collect_run_metadata
from neurodatabench.models import (
    JsonObject,
    QuestionForImplementation,
    RunContext,
    RunTimings,
    SubmittedAnswer,
)
from neurodatabench.packaging import write_result_bundle
from neurodatabench.profiling import Profiler
from neurodatabench.schemas import (
    dataset_paths_from_questions,
    question_rows,
    load_questions,
    prepare_questions_for_implementation,
)
from neurodatabench.validation import validate_answers


@dataclass(frozen=True)
class RunArgs:
    """Parsed CLI arguments for one benchmark run."""

    command: str
    questions: str
    out: Path
    profile_interval_ms: int


def main(
    *,
    implementation_name: str,
    implementation_cache_type: str,
    setup: Callable[[RunContext], None],
    answer_questions: Callable[
        [list[QuestionForImplementation]],
        list[QuestionForImplementation],
    ],
    questions: str | Path | None = None,
    out: str | Path | None = None,
    clear_cache: Callable[[RunContext], None] | None = None,
    teardown: Callable[[], None] | None = None,
    implementation_notes: str = "",
) -> None:
    """Run a benchmark submission and write result artifacts."""
    args = resolve_run_args(default_questions=questions, default_out=out)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    question_spec = load_questions(args.questions)

    dataset_paths = dataset_paths_from_questions(question_spec)

    questions_with_answers = question_rows(question_spec)
    questions_for_impl = prepare_questions_for_implementation(questions_with_answers)

    scratch_dir = out_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

    context = RunContext(
        dataset_paths=dataset_paths,
        out_dir=out_dir,
        scratch_dir=scratch_dir,
        questions_source=args.questions,
        submission_path=Path(sys.argv[0]),
    )

    metadata: JsonObject = collect_run_metadata(
        context=context,
        implementation_name=implementation_name,
        implementation_cache_type=implementation_cache_type,
        implementation_notes=implementation_notes,
    )

    answers: list[SubmittedAnswer] = []
    answered_questions: list[QuestionForImplementation] = []
    setup_duration_ns: int | None = None
    answer_questions_duration_ns: int | None = None
    total_duration_ns: int | None = None
    profiler: Profiler | None = None
    setup_started = False

    if clear_cache is not None:
        clear_cache(context)

    profiler = Profiler(interval_seconds=args.profile_interval_ms / 1000)
    profiler.start()

    try:
        total_start_ns = perf_counter_ns()
        try:
            setup_start_ns = perf_counter_ns()
            setup_started = True
            try:
                setup(context)
            finally:
                setup_duration_ns = perf_counter_ns() - setup_start_ns

            answer_questions_start_ns = perf_counter_ns()
            try:
                answered_questions = answer_questions(questions_for_impl)
            finally:
                answer_questions_duration_ns = (
                    perf_counter_ns() - answer_questions_start_ns
                )
        finally:
            total_duration_ns = perf_counter_ns() - total_start_ns

        answers = normalize_answers(answered_questions)

    finally:
        profiler.stop()

        if setup_started and teardown is not None:
            teardown()

    profile_samples = profiler.samples if profiler is not None else []
    profile_summary = profiler.summary() if profiler is not None else None

    validation = validate_answers(
        questions_with_answers=questions_with_answers,
        submitted_answers=answers,
    )

    timings = RunTimings(
        setup_duration_ns=setup_duration_ns,
        answer_questions_duration_ns=answer_questions_duration_ns,
        total_duration_ns=total_duration_ns,
    )

    write_result_bundle(
        out_dir=out_dir,
        questions=question_spec,
        metadata=metadata,
        timings=timings,
        answers=answers,
        validation=validation,
        profile_samples=profile_samples,
        profile_summary=profile_summary,
    )

    if not validation.correct:
        raise SystemExit(1)


def normalize_answers(
    questions: list[QuestionForImplementation],
) -> list[SubmittedAnswer]:
    """Convert answered implementation questions into result rows."""
    return [
        SubmittedAnswer(
            index=index,
            question_id=question.id,
            answer=question.answer,
        )
        for index, question in enumerate(questions)
    ]


def resolve_run_args(
    *,
    default_questions: str | Path | None,
    default_out: str | Path | None,
) -> RunArgs:
    """Resolve run arguments, rejecting ambiguous CLI/default conflicts."""
    parsed = parse_args()

    if parsed.questions is not None and default_questions is not None:
        raise SystemExit("error: pass questions either in main() or --questions, not both")
    if parsed.out is not None and default_out is not None:
        raise SystemExit("error: pass out either in main() or --out, not both")

    questions = parsed.questions if parsed.questions is not None else default_questions
    out = parsed.out if parsed.out is not None else default_out

    if questions is None:
        raise SystemExit("error: --questions is required unless main(questions=...) is set")
    if out is None:
        raise SystemExit("error: --out is required unless main(out=...) is set")

    return RunArgs(
        command=parsed.command,
        questions=str(questions),
        out=Path(out),
        profile_interval_ms=parsed.profile_interval_ms,
    )


def parse_args() -> RunArgs:
    """Parse command-line arguments for a run or smoke command."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--questions")
    run.add_argument("--out")
    run.add_argument("--profile-interval-ms", type=int, default=250)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--questions")
    smoke.add_argument("--out")
    smoke.add_argument("--profile-interval-ms", type=int, default=250)

    namespace = parser.parse_args()
    return RunArgs(
        command=namespace.command,
        questions=namespace.questions,
        out=Path(namespace.out),
        profile_interval_ms=namespace.profile_interval_ms,
    )
```

---

## 7. Schema utilities skeleton

```python
# src/neurodatabench/schemas.py

"""Schema loading and conversion utilities for benchmark inputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from neurodatabench.models import (
    JsonObject,
    QuestionForImplementation,
)


def load_questions(source: str | Path) -> JsonObject:
    """Load and validate questions from a path or packaged question name."""
    questions = require_json_object(json.loads(read_questions_text(source)))

    dataset_paths = questions.get("dataset_paths")
    if not isinstance(dataset_paths, list) or not dataset_paths:
        raise ValueError("questions.json must include non-empty dataset_paths list")

    if not all(isinstance(path, str) for path in dataset_paths):
        raise ValueError("questions.json dataset_paths entries must be strings")

    rows = questions.get("questions")
    if not isinstance(rows, list):
        raise ValueError("questions.json must include questions list")

    seen: set[str] = set()

    for raw_question in rows:
        question = require_json_object(raw_question)
        question_id = question.get("id")
        if not isinstance(question_id, str):
            raise ValueError("Each question must include string id")
        if question_id in seen:
            raise ValueError(f"Duplicate question id: {question_id}")
        seen.add(question_id)

    return questions


def read_questions_text(source: str | Path) -> str:
    """Read questions text from a filesystem path or package resource."""
    path = Path(source)

    if path.exists():
        return path.read_text()

    question_name = path.stem if path.suffix == ".json" else str(source)
    resource = files("neurodatabench").joinpath(
        "questions",
        f"{question_name}.json",
    )

    if not resource.is_file():
        raise FileNotFoundError(f"Questions not found: {source}")

    return resource.read_text()


def question_rows(question_spec: JsonObject) -> list[JsonObject]:
    """Return the ordered question objects from a loaded questions file."""
    rows = question_spec.get("questions")
    if not isinstance(rows, list):
        raise ValueError("questions.json must include questions list")
    return [require_json_object(row) for row in rows]


def dataset_paths_from_questions(question_spec: JsonObject) -> list[str]:
    """Return the dataset paths declared by a loaded questions file."""
    dataset_paths = question_spec.get("dataset_paths")
    if not isinstance(dataset_paths, list) or not dataset_paths:
        raise ValueError("questions.json must include non-empty dataset_paths list")

    if not all(isinstance(path, str) for path in dataset_paths):
        raise ValueError("questions.json dataset_paths entries must be strings")

    return dataset_paths


def prepare_questions_for_implementation(
    questions: Sequence[JsonObject],
) -> list[QuestionForImplementation]:
    """Return implementation-facing questions with blank answer fields."""
    prepared: list[QuestionForImplementation] = []

    for raw_question in questions:
        question = require_json_object(raw_question)
        question_id = question.get("id")
        question_text = question.get("text")
        if not isinstance(question_id, str) or not isinstance(question_text, str):
            raise ValueError("Each question must include string id and text")
        prepared.append(QuestionForImplementation(id=question_id, text=question_text))

    return prepared


def require_json_object(value: object) -> JsonObject:
    """Return value as a JSON object or raise a clear validation error."""
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value
```

---

## 8. Validation skeleton

```python
# src/neurodatabench/validation.py

"""Answer validation utilities for completed benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurodatabench.models import (
    JsonObject,
    JsonValue,
    SubmittedAnswer,
    ValidationResult,
    ValidationSummary,
)


@dataclass(frozen=True)
class ComparisonResult:
    """Recursive JSON comparison result."""

    correct: bool
    reason: str


def validate_answers(
    questions_with_answers: list[JsonObject],
    submitted_answers: list[SubmittedAnswer],
) -> ValidationSummary:
    """Validate submitted answers after timed execution finishes."""
    results: list[ValidationResult] = []
    all_correct = True

    if len(questions_with_answers) != len(submitted_answers):
        all_correct = False

    for index, question in enumerate(questions_with_answers):
        expected_id = question.get("id")
        expected_answer = question.get("answer")

        if not isinstance(expected_id, str):
            raise ValueError("Each question must include string id")

        if index >= len(submitted_answers):
            results.append(ValidationResult(
                index=index,
                question_id=expected_id,
                correct=False,
                reason="missing_answer",
            ))
            all_correct = False
            continue

        submitted = submitted_answers[index]
        actual_id = submitted.question_id
        actual_answer = submitted.answer

        if actual_id != expected_id:
            results.append(ValidationResult(
                index=index,
                question_id=expected_id,
                correct=False,
                reason="question_id_mismatch",
                actual_question_id=actual_id,
            ))
            all_correct = False
            continue

        comparison = compare_json_value(
            expected_answer,
            actual_answer,
        )

        results.append(ValidationResult(
            index=index,
            question_id=expected_id,
            correct=comparison.correct,
            reason=comparison.reason,
        ))

        if not comparison.correct:
            all_correct = False

    return ValidationSummary(
        correct=all_correct,
        num_questions=len(questions_with_answers),
        num_submitted=len(submitted_answers),
        num_correct=sum(1 for result in results if result.correct),
        results=results,
    )


def compare_json_value(expected: JsonValue, actual: JsonValue) -> ComparisonResult:
    """Compare expected and actual JSON values recursively."""
    if isinstance(expected, bool):
        return ComparisonResult(actual is expected, "exact_bool")

    if isinstance(expected, int) and not isinstance(expected, bool):
        return ComparisonResult(
            actual == expected and isinstance(actual, int),
            "exact_int",
        )

    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return ComparisonResult(False, "expected_float")
        return ComparisonResult(
            bool(np.isclose(float(actual), expected)),
            "float_close",
        )

    if isinstance(expected, str):
        return ComparisonResult(actual == expected, "exact_str")

    if expected is None:
        return ComparisonResult(actual is None, "exact_null")

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return ComparisonResult(False, "expected_list")
        if len(expected) != len(actual):
            return ComparisonResult(False, "list_length_mismatch")
        for e, a in zip(expected, actual):
            comparison = compare_json_value(e, a)
            if not comparison.correct:
                return ComparisonResult(False, f"list_item_{comparison.reason}")
        return ComparisonResult(True, "list_match")

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return ComparisonResult(False, "expected_object")
        if set(expected.keys()) != set(actual.keys()):
            return ComparisonResult(False, "object_keys_mismatch")
        for key in expected:
            comparison = compare_json_value(expected[key], actual[key])
            if not comparison.correct:
                return ComparisonResult(
                    False,
                    f"object_value_{key}_{comparison.reason}",
                )
        return ComparisonResult(True, "object_match")

    return ComparisonResult(False, "unsupported_expected_type")
```

---

## 9. Inline metadata and environment capture

```python
# src/neurodatabench/metadata.py

"""Run metadata and inline script metadata capture."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from neurodatabench.models import JsonObject, RunContext


def collect_run_metadata(
    context: RunContext,
    implementation_name: str,
    implementation_cache_type: str,
    implementation_notes: str,
) -> JsonObject:
    """Collect run metadata without shelling out to pip."""
    inline_metadata = parse_uv_inline_metadata(context.submission_path)
    key_packages = package_versions_from_inline_metadata(inline_metadata)

    return {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "benchmark_harness_version": get_harness_version(),
        "implementation": {
            "name": implementation_name,
            "cache_type": implementation_cache_type,
            "notes": implementation_notes,
        },
        "paths": {
            "questions": context.questions_source,
            "submission": str(context.submission_path),
        },
        "dataset": {
            "paths": context.dataset_paths,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "process": {
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "argv": sys.argv,
        },
        "inline_script_metadata": inline_metadata,
        "key_package_versions": key_packages,
    }


def parse_uv_inline_metadata(script_path: Path) -> JsonObject:
    """Parse PEP 723 inline metadata from a submission script."""
    lines = script_path.read_text().splitlines()

    in_block = False
    block_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped == "# /// script":
            in_block = True
            continue

        if in_block and stripped == "# ///":
            break

        if in_block:
            if line.startswith("#"):
                block_lines.append(line[1:].lstrip())
            else:
                block_lines.append(line)

    if not block_lines:
        return {}

    parsed = tomllib.loads("\n".join(block_lines))
    return parsed


def package_versions_from_inline_metadata(
    metadata: JsonObject,
) -> dict[str, str | None]:
    """Return installed versions for declared inline dependencies."""
    raw_deps = metadata.get("dependencies", [])
    deps = raw_deps if isinstance(raw_deps, list) else []
    versions: dict[str, str | None] = {}

    for dep in deps:
        if not isinstance(dep, str):
            continue
        name = normalize_requirement_name(dep)
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    return versions


def normalize_requirement_name(requirement: str) -> str:
    """Return a normalized package name from a dependency declaration."""
    # Keep minimal. Replace with packaging.requirements.Requirement if available.
    raw = requirement.split(";")[0].strip()
    raw = raw.split("[")[0]
    for sep in ["==", ">=", "<=", "~=", ">", "<", " @ "]:
        if sep in raw:
            raw = raw.split(sep)[0]
    return raw.strip().lower().replace("_", "-")


def get_harness_version() -> str:
    """Return the installed NeuroDataBench package version if available."""
    try:
        return importlib.metadata.version("neurodatabench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
```

Rules:

```text
Do not call pip freeze.
Do not shell out to pip.
Only record declared inline dependencies plus resolved installed versions.
```

---

## 10. Profiler skeleton

```python
# src/neurodatabench/profiling.py

"""Process and system resource profiling for benchmark runs."""

from __future__ import annotations

import os
import threading
import time
from typing import Protocol

import psutil

from neurodatabench.models import JsonObject


class CounterSnapshot(Protocol):
    """Protocol for psutil namedtuple counters."""

    _fields: tuple[str, ...]


class Profiler:
    """Background sampler for process and system resource usage."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        """Create a profiler with the requested sampling interval."""
        self.interval_seconds = interval_seconds
        self.samples: list[JsonObject] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())
        self._net_start: CounterSnapshot | None = None
        self._disk_start: CounterSnapshot | None = None
        self._net_end: CounterSnapshot | None = None
        self._disk_end: CounterSnapshot | None = None

    def start(self) -> None:
        """Start collecting profiler samples."""
        self._net_start = psutil.net_io_counters()
        self._disk_start = psutil.disk_io_counters()

        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop collecting profiler samples."""
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2)

        self._net_end = psutil.net_io_counters()
        self._disk_end = psutil.disk_io_counters()

    def _run(self) -> None:
        """Collect samples until stopped."""
        while not self._stop.is_set():
            self.samples.append(self._sample())
            time.sleep(self.interval_seconds)

    def _sample(self) -> JsonObject:
        """Collect one profiler sample as a JSON object."""
        mem = self._process.memory_info()

        child_rss = 0
        child_cpu_percent = 0.0

        for child in self._process.children(recursive=True):
            try:
                child_mem = child.memory_info()
                child_rss += child_mem.rss
                child_cpu_percent += child.cpu_percent(interval=None)
            except psutil.Error:
                pass

        vm = psutil.virtual_memory()

        return {
            "time_ns": time.time_ns(),
            "process": {
                "cpu_percent": self._process.cpu_percent(interval=None),
                "rss_bytes": mem.rss,
                "vms_bytes": mem.vms,
                "num_threads": self._process.num_threads(),
            },
            "children": {
                "rss_bytes": child_rss,
                "cpu_percent": child_cpu_percent,
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_total_bytes": vm.total,
                "memory_available_bytes": vm.available,
                "memory_percent": vm.percent,
            },
            "io": {
                "net": psutil.net_io_counters()._asdict(),
                "disk": psutil.disk_io_counters()._asdict(),
            },
        }

    def summary(self) -> JsonObject:
        """Summarize collected profiler samples."""
        peak_rss = max(
            [
                int(sample["process"]["rss_bytes"])
                for sample in self.samples
                if isinstance(sample.get("process"), dict)
            ],
            default=None,
        )

        peak_total_rss = max(
            [
                int(sample["process"]["rss_bytes"])
                + int(sample["children"]["rss_bytes"])
                for sample in self.samples
                if isinstance(sample.get("process"), dict)
                and isinstance(sample.get("children"), dict)
            ],
            default=None,
        )

        return {
            "sample_interval_seconds": self.interval_seconds,
            "num_samples": len(self.samples),
            "peak_process_rss_bytes": peak_rss,
            "peak_process_plus_children_rss_bytes": peak_total_rss,
            "network_delta": diff_counters(self._net_start, self._net_end),
            "disk_delta": diff_counters(self._disk_start, self._disk_end),
            "network_scope": "system_delta",
            "disk_scope": "system_delta",
        }


def diff_counters(
    start: CounterSnapshot | None,
    end: CounterSnapshot | None,
) -> dict[str, int] | None:
    """Return counter deltas for psutil namedtuple counters."""
    if start is None or end is None:
        return None

    out: dict[str, int] = {}
    for key in start._fields:
        out[key] = getattr(end, key) - getattr(start, key)

    return out
```

---

## 11. Result bundle writer

```python
# src/neurodatabench/packaging.py

"""Result artifact writer for NeuroDataBench benchmark bundles."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TypeAlias

from neurodatabench.models import (
    JsonObject,
    JsonValue,
    RunTimings,
    SubmittedAnswer,
    ValidationSummary,
)

JsonWritable: TypeAlias = JsonValue | JsonObject | RunTimings | ValidationSummary
JsonLineRow: TypeAlias = SubmittedAnswer | JsonObject


def write_result_bundle(
    out_dir: Path,
    questions: JsonObject,
    metadata: JsonObject,
    timings: RunTimings,
    answers: list[SubmittedAnswer],
    validation: ValidationSummary,
    profile_samples: list[JsonObject],
    profile_summary: JsonObject | None,
) -> None:
    """Write all result files and package them into a zip bundle."""
    write_json(out_dir / "questions.json", questions)
    write_json(out_dir / "run_metadata.json", metadata)
    write_json(out_dir / "timings.json", timings)
    write_json(out_dir / "validation.json", validation)
    write_json(out_dir / "profile_summary.json", profile_summary)

    write_jsonl(out_dir / "answers.jsonl", answers)
    write_jsonl(out_dir / "profile_samples.jsonl", profile_samples)

    bundle_path = out_dir / "results_bundle.zip"

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in out_dir.iterdir():
            if path.name == bundle_path.name:
                continue
            if path.is_file():
                z.write(path, arcname=path.name)


def write_json(path: Path, value: JsonWritable) -> None:
    """Write a dataclass or JSON value as pretty JSON."""
    if is_dataclass(value):
        path.write_text(json.dumps(asdict(value), indent=2, sort_keys=True))
        return

    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def write_jsonl(path: Path, rows: Sequence[JsonLineRow]) -> None:
    """Write typed model/dataclass rows as newline-delimited JSON."""
    with path.open("w") as f:
        for row in rows:
            if is_dataclass(row):
                f.write(json.dumps(asdict(row), sort_keys=True) + "\n")
            else:
                f.write(json.dumps(row, sort_keys=True) + "\n")
```

---

# Output files

```text
results/
  questions.json
  run_metadata.json
  timings.json
  answers.jsonl
  validation.json
  profile_samples.jsonl
  profile_summary.json
  results_bundle.zip
```

---

# `timings.json` shape

```json
{
  "setup_timed": true,
  "setup_duration_ns": 123456789,
  "answer_questions_duration_ns": 987654321,
  "total_duration_ns": 1111111110
}
```

---

# `answers.jsonl` shape

```json
{"index":0,"question_id":"q001_units_VISp_default_qc","answer":123}
```

---

# `run_metadata.json` required fields

```text
datetime_utc
hostname
benchmark_harness_version
implementation
implementation.cache_type
dataset
paths
python
platform
process
inline_script_metadata
key_package_versions
```

---

# Acceptance checklist

## Schema

* [ ] `questions.json` does not require `schema_version`.
* [ ] `questions.json` has top-level `dataset_paths`.
* [ ] `dataset_paths` is a non-empty list of local paths or remote URIs.
* [ ] `questions.json` has no dataset ID indirection.
* [ ] There is no separate dataset manifest.
* [ ] Each question has `id`, `text`, `answer`.
* [ ] No question has `order`.
* [ ] No question has `depends_on`.
* [ ] No question has `input_files`.
* [ ] Question list order is execution order.
* [ ] Dataset paths are available through `RunContext`.
* [ ] Packaged question filenames are benchmark names.

## Timing

* [ ] Harness import happens only inside `if __name__ == "__main__":`.
* [ ] Implementation import duration is not explicitly timed or recorded.
* [ ] `clear_cache()` is optional, untimed, and called only before the first measured run.
* [ ] Setup duration is captured separately.
* [ ] `answer_questions()` duration is captured separately.
* [ ] `total_duration_ns` includes setup and `answer_questions()` duration.
* [ ] Teardown duration is not captured.
* [ ] Per-question duration is not captured by the harness.
* [ ] Validation starts only after setup, answer, and total timers stop.
* [ ] Answers are written after the timed answer pass.

## Implementation interface

* [ ] User script does not define top-level `IMPLEMENTATION` metadata.
* [ ] User script passes implementation metadata to `main()` inside `if __name__ == "__main__":`.
* [ ] User script may pass default `questions=...` and `out=...` values to `main()`.
* [ ] CLI `--questions` and `--out` raise a clear error when the matching default is also passed to `main()`.
* [ ] User script defines `setup()`.
* [ ] User script defines `answer_questions()`.
* [ ] User script optionally defines `teardown()`.
* [ ] User script declares implementation cache type in `main()`.
* [ ] User script keeps submission-owned state in module-level variables.
* [ ] `setup()` resets module-level state for each measured run.
* [ ] The runner does not inspect or pass submission state between hooks.
* [ ] `answer_questions()` receives the full ordered question list without expected answers.
* [ ] `answer_questions()` fills each question's `answer` field and returns the completed list.
* [ ] There is no separate implementation `warmup()` hook.

## Typing

* [ ] Public harness functions have complete parameter and return type hints.
* [ ] Submission template functions have complete parameter and return type hints.
* [ ] Public hook/result records use dataclasses where attributes improve clarity.
* [ ] Question specs, run metadata, inline script metadata, and profiler records remain plain JSON objects.
* [ ] No run status enum, status string, status file, or captured error artifact.
* [ ] Pydantic models are not constructed in the user script top level or timed setup/answer path.

## Metadata

* [ ] No `pip freeze`.
* [ ] Parse uv inline script metadata.
* [ ] Record declared dependencies.
* [ ] Record installed versions for declared dependencies.
* [ ] Record implementation cache type.
* [ ] Record dataset paths.
* [ ] Record Python executable/version.
* [ ] Record platform info.
* [ ] Record datetime UTC.
* [ ] Do not hash dataset files or input files.

## Profiling

* [ ] Sample process CPU.
* [ ] Sample process RSS/VMS.
* [ ] Sample child process RSS/CPU.
* [ ] Record a synchronous pre-setup RSS baseline and derived peak RSS deltas for fair memory comparison while preserving raw peak RSS.
* [ ] Sample system CPU.
* [ ] Sample system memory.
* [ ] Record network counter delta.
* [ ] Record disk counter delta.
* [ ] Mark network/disk counters as system-scope deltas.

## Validation

* [ ] Validate by question index.
* [ ] Validate question ID match.
* [ ] Infer expected type from `answer`.
* [ ] Exact compare ints, strings, bools, null.
* [ ] No validation config or validation DSL in `questions.json`.
* [ ] Float compare with `numpy.isclose` defaults.
* [ ] Recursive compare lists/objects.
* [ ] Write `validation.json`.
* [ ] Any incorrect answer makes `validation.correct` false.
* [ ] CLI exits nonzero after writing result artifacts when `validation.correct` is false.

## Packaging

* [ ] Write all result files.
* [ ] Create `results_bundle.zip`.
* [ ] Include questions with answers.
* [ ] Include metadata.
* [ ] Include timings.
* [ ] Include answers.
* [ ] Include validation.
* [ ] Include profile samples.
* [ ] Include profile summary.
* [ ] Distribute `src/neurodatabench/questions/*.json` in wheels and sdists.
* [ ] Runner resolves `--questions` as either a filesystem path or packaged question name.
* [ ] Runner resolves `main(questions=...)` the same way when CLI `--questions` is omitted.
* [ ] Runner rejects runs where both CLI `--questions` and `main(questions=...)` are supplied.
* [ ] Runner rejects runs where both CLI `--out` and `main(out=...)` are supplied.
* [ ] Smoke tests can run locally with packaged question names.
