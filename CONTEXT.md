# NWB benchmark implementation

## Design decisions

* Single user-edited script executes the benchmark.
* Harness is imported only inside `if __name__ == "__main__":`
* Implementation import time is measured.
* Setup is not timed.
* Total ordered-question-loop time is measured.
* Per-question times are measured.
* Validation happens only after all timed questions finish.
* `questions.json` contains answers.
* Do not pass answers to implementation.
* `dataset_id` appears once at top of `questions.json`.
* No `order` field; list order is canonical.
* No `depends_on`.
* No `input_files`.
* All files for the dataset are passed to `setup()`.
* No `pip freeze`.
* Use uv inline script metadata for declared dependencies.
* Harness records platform, datetime, Python, key package versions, CPU, memory, disk, network counters, profiling samples.
* Implementer declares only:

  * implementation name
  * declared cache state: `cold`, `warm`, `dandi_pre_cache`, or `other`
  * optional notes

---

# Repo skeleton

```text
nwb_benchmark/
  __init__.py
  runner.py
  schemas.py
  metadata.py
  profiling.py
  validation.py
  packaging.py
  templates/
    submission_template.py
  examples/
    manifest.json
    questions.json
  tests/
    test_validation.py
    test_inline_metadata.py
    test_runner_smoke.py
```

---

# Step-by-step implementation checklist

## 1. Define schemas

### `manifest.json`

```json
{
  "schema_version": "0.3",
  "benchmark_id": "nwb_access_benchmark_v1",
  "datasets": [
    {
      "dataset_id": "visual_coding_v1",
      "name": "Visual coding NWB benchmark dataset",
      "description": "Benchmark dataset",
      "files": [
        {
          "asset_id": "session_001",
          "path": "/data/nwb/session_001.nwb",
          "uri": null,
          "sha256": null,
          "size_bytes": null
        }
      ]
    }
  ]
}
```

### `questions.json`

```json
{
  "schema_version": "0.3",
  "benchmark_id": "nwb_access_benchmark_v1",
  "dataset_id": "visual_coding_v1",
  "validation": {
    "float_abs_tol": 1e-9,
    "float_rel_tol": 1e-9
  },
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
    }
  ]
}
```

Rules:

```text
questions[*].answer defines expected type.
No answer_type.
No order.
No depends_on.
No input_files.
```

---

## 2. Submission template

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pynwb",
#   "h5py",
#   "numpy",
#   "psutil"
# ]
# ///

from time import perf_counter_ns as _perf_counter_ns

_IMPLEMENTATION_IMPORT_START_NS = _perf_counter_ns()

# Implementation imports go here.
# They are included in implementation import timing.
import numpy as np
import pynwb

IMPLEMENTATION = {
    "name": "replace-me",
    "declared_cache_state": "cold",  # cold | warm | dandi_pre_cache | other
    "notes": ""
}


def setup(dataset, scratch_dir, context):
    """
    Untimed.

    dataset:
        {
          "dataset_id": "...",
          "files": [...]
        }

    scratch_dir:
        Path-like temporary directory for this run.

    context:
        runner-provided metadata.
    """
    return {
        "dataset": dataset,
        "scratch_dir": scratch_dir
    }


def warmup(state, questions):
    """
    Optional. Untimed.
    Return value ignored.
    """
    return None


def answer_question(question, state):
    """
    Timed.

    question does NOT contain answer.
    Question order is canonical.
    May mutate state.
    Must return JSON-serializable answer.
    """
    qid = question["id"]

    if qid == "q001_units_VISp_default_qc":
        return 123

    raise NotImplementedError(qid)


def teardown(state):
    """
    Untimed.
    """
    return None


_IMPLEMENTATION_IMPORT_END_NS = _perf_counter_ns()


if __name__ == "__main__":
    from nwb_benchmark.runner import main

    main(
        implementation=IMPLEMENTATION,
        setup=setup,
        warmup=warmup,
        answer_question=answer_question,
        teardown=teardown,
        implementation_import_duration_ns=(
            _IMPLEMENTATION_IMPORT_END_NS - _IMPLEMENTATION_IMPORT_START_NS
        ),
    )
```

---

## 3. CLI

```bash
python submission.py run \
  --manifest manifest.json \
  --questions questions.json \
  --out results \
  --warmup none
```

```bash
python submission.py run \
  --manifest manifest.json \
  --questions questions.json \
  --out results \
  --warmup full \
  --declared-cache-state warm
```

```bash
python submission.py smoke \
  --manifest manifest.json \
  --questions questions.json
```

CLI options:

```text
run
smoke
--manifest PATH
--questions PATH
--out DIR
--warmup none|full
--declared-cache-state cold|warm|dandi_pre_cache|other
--profile-interval-ms 250
--stop-on-error
```

---

## 4. Runner lifecycle

```text
parse args
load manifest
load questions
validate schema shape
select dataset using questions.dataset_id
strip answer fields before implementation sees questions
collect pre-run metadata
call setup()                       # untimed
run warmup if requested             # untimed
start profiler
start total question-loop timer
for question in ordered questions:
    start per-question timer
    call answer_question()
    stop per-question timer
    store answer in memory
stop total question-loop timer
stop profiler
call teardown()                     # untimed
write answers
write timings
validate answers                    # after timing only
write metadata
write profile summary
package result bundle
```

---

## 5. Runner skeleton

```python
# nwb_benchmark/runner.py

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable

from .metadata import collect_run_metadata
from .profiling import Profiler
from .schemas import load_manifest, load_questions, select_dataset, strip_answers
from .validation import validate_answers
from .packaging import write_result_bundle


def main(
    implementation: dict[str, Any],
    setup: Callable,
    warmup: Callable | None,
    answer_question: Callable,
    teardown: Callable | None,
    implementation_import_duration_ns: int,
) -> None:
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.manifest))
    question_spec = load_questions(Path(args.questions))

    dataset_id = question_spec["dataset_id"]
    dataset = select_dataset(manifest, dataset_id)

    questions_with_answers = question_spec["questions"]
    questions_for_impl = strip_answers(questions_with_answers)

    declared_cache_state = (
        args.declared_cache_state
        or implementation.get("declared_cache_state")
        or "other"
    )

    metadata = collect_run_metadata(
        implementation=implementation,
        declared_cache_state=declared_cache_state,
        manifest_path=Path(args.manifest),
        questions_path=Path(args.questions),
        submission_path=Path(sys.argv[0]),
        implementation_import_duration_ns=implementation_import_duration_ns,
        warmup=args.warmup,
    )

    scratch_dir = out_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

    context = {
        "dataset_id": dataset_id,
        "declared_cache_state": declared_cache_state,
        "warmup": args.warmup,
        "out_dir": str(out_dir),
    }

    state = None
    answers = []
    per_question_timings = []
    total_questions_duration_ns = None
    status = "ok"
    error = None

    try:
        state = setup(dataset, scratch_dir, context)

        if args.warmup == "full":
            if warmup is not None:
                warmup(state, questions_for_impl)
            else:
                for q in questions_for_impl:
                    answer_question(q, state)

        profiler = Profiler(interval_seconds=args.profile_interval_ms / 1000)
        profiler.start()

        total_start_ns = perf_counter_ns()

        for index, question in enumerate(questions_for_impl):
            q_start_ns = perf_counter_ns()

            try:
                answer = answer_question(question, state)
                q_status = "ok"
                q_error = None
            except Exception as exc:
                answer = None
                q_status = "error"
                q_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc()
                }
                if args.stop_on_error:
                    raise

            q_end_ns = perf_counter_ns()

            answers.append({
                "index": index,
                "question_id": question["id"],
                "answer": answer,
                "status": q_status,
                "error": q_error
            })

            per_question_timings.append({
                "index": index,
                "question_id": question["id"],
                "duration_ns": q_end_ns - q_start_ns,
                "status": q_status
            })

        total_end_ns = perf_counter_ns()
        total_questions_duration_ns = total_end_ns - total_start_ns

        profiler.stop()

    except Exception as exc:
        status = "error"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()
        }

    finally:
        if state is not None and teardown is not None:
            teardown(state)

    profile_samples = profiler.samples if "profiler" in locals() else []
    profile_summary = profiler.summary() if "profiler" in locals() else {}

    validation = validate_answers(
        questions_with_answers=questions_with_answers,
        submitted_answers=answers,
        validation_config=question_spec.get("validation", {}),
    )

    timings = {
        "implementation_import_duration_ns": implementation_import_duration_ns,
        "setup_timed": False,
        "total_questions_duration_ns": total_questions_duration_ns,
        "per_question": per_question_timings
    }

    write_result_bundle(
        out_dir=out_dir,
        manifest=manifest,
        questions=question_spec,
        metadata=metadata,
        timings=timings,
        answers=answers,
        validation=validation,
        profile_samples=profile_samples,
        profile_summary=profile_summary,
        status=status,
        error=error,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--questions", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--warmup", choices=["none", "full"], default="none")
    run.add_argument(
        "--declared-cache-state",
        choices=["cold", "warm", "dandi_pre_cache", "other"],
        default=None,
    )
    run.add_argument("--profile-interval-ms", type=int, default=250)
    run.add_argument("--stop-on-error", action="store_true")

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--manifest", required=True)
    smoke.add_argument("--questions", required=True)
    smoke.add_argument("--out", default="smoke_results")
    smoke.add_argument("--warmup", choices=["none", "full"], default="none")
    smoke.add_argument("--declared-cache-state", default=None)
    smoke.add_argument("--profile-interval-ms", type=int, default=250)
    smoke.add_argument("--stop-on-error", action="store_true")

    return parser.parse_args()
```

---

## 6. Schema utilities skeleton

```python
# nwb_benchmark/schemas.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        manifest = json.load(f)

    assert "datasets" in manifest
    assert isinstance(manifest["datasets"], list)

    return manifest


def load_questions(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        questions = json.load(f)

    assert "dataset_id" in questions
    assert "questions" in questions
    assert isinstance(questions["questions"], list)

    seen = set()
    for q in questions["questions"]:
        assert "id" in q
        assert "text" in q
        assert "answer" in q
        assert "order" not in q
        assert "depends_on" not in q
        assert "input_files" not in q
        assert q["id"] not in seen
        seen.add(q["id"])

    return questions


def select_dataset(manifest: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [
        dataset for dataset in manifest["datasets"]
        if dataset["dataset_id"] == dataset_id
    ]

    if len(matches) != 1:
        raise ValueError(f"Expected one dataset for {dataset_id}, found {len(matches)}")

    return matches[0]


def strip_answers(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped = []

    for q in questions:
        q2 = dict(q)
        q2.pop("answer", None)
        stripped.append(q2)

    return stripped
```

---

## 7. Validation skeleton

```python
# nwb_benchmark/validation.py

from __future__ import annotations

import math
from typing import Any


def validate_answers(
    questions_with_answers: list[dict[str, Any]],
    submitted_answers: list[dict[str, Any]],
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    abs_tol = validation_config.get("float_abs_tol", 1e-9)
    rel_tol = validation_config.get("float_rel_tol", 1e-9)

    results = []
    all_correct = True

    if len(questions_with_answers) != len(submitted_answers):
        all_correct = False

    for index, question in enumerate(questions_with_answers):
        expected_id = question["id"]
        expected_answer = question["answer"]

        if index >= len(submitted_answers):
            results.append({
                "index": index,
                "question_id": expected_id,
                "correct": False,
                "reason": "missing_answer"
            })
            all_correct = False
            continue

        submitted = submitted_answers[index]
        actual_id = submitted.get("question_id")
        actual_answer = submitted.get("answer")

        if actual_id != expected_id:
            results.append({
                "index": index,
                "question_id": expected_id,
                "correct": False,
                "reason": "question_id_mismatch",
                "actual_question_id": actual_id
            })
            all_correct = False
            continue

        correct, reason = compare_json_value(
            expected_answer,
            actual_answer,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

        results.append({
            "index": index,
            "question_id": expected_id,
            "correct": correct,
            "reason": reason
        })

        if not correct:
            all_correct = False

    return {
        "correct": all_correct,
        "num_questions": len(questions_with_answers),
        "num_submitted": len(submitted_answers),
        "num_correct": sum(1 for r in results if r["correct"]),
        "results": results
    }


def compare_json_value(expected: Any, actual: Any, abs_tol: float, rel_tol: float):
    if isinstance(expected, bool):
        return actual is expected, "exact_bool"

    if isinstance(expected, int) and not isinstance(expected, bool):
        return actual == expected and isinstance(actual, int), "exact_int"

    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False, "expected_float"
        return math.isclose(float(actual), expected, abs_tol=abs_tol, rel_tol=rel_tol), "float_close"

    if isinstance(expected, str):
        return actual == expected, "exact_str"

    if expected is None:
        return actual is None, "exact_null"

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False, "expected_list"
        if len(expected) != len(actual):
            return False, "list_length_mismatch"
        for e, a in zip(expected, actual):
            ok, reason = compare_json_value(e, a, abs_tol, rel_tol)
            if not ok:
                return False, f"list_item_{reason}"
        return True, "list_match"

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, "expected_object"
        if set(expected.keys()) != set(actual.keys()):
            return False, "object_keys_mismatch"
        for key in expected:
            ok, reason = compare_json_value(expected[key], actual[key], abs_tol, rel_tol)
            if not ok:
                return False, f"object_value_{key}_{reason}"
        return True, "object_match"

    return False, "unsupported_expected_type"
```

---

## 8. Inline metadata and environment capture

```python
# nwb_benchmark/metadata.py

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def collect_run_metadata(
    implementation: dict[str, Any],
    declared_cache_state: str,
    manifest_path: Path,
    questions_path: Path,
    submission_path: Path,
    implementation_import_duration_ns: int,
    warmup: str,
) -> dict[str, Any]:
    inline_metadata = parse_uv_inline_metadata(submission_path)
    key_packages = package_versions_from_inline_metadata(inline_metadata)

    return {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "benchmark_harness_version": get_harness_version(),
        "implementation": implementation,
        "declared_cache_state": declared_cache_state,
        "warmup": warmup,
        "implementation_import_duration_ns": implementation_import_duration_ns,
        "paths": {
            "manifest": str(manifest_path),
            "questions": str(questions_path),
            "submission": str(submission_path),
        },
        "sha256": {
            "manifest": sha256_file(manifest_path),
            "questions": sha256_file(questions_path),
            "submission": sha256_file(submission_path),
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


def parse_uv_inline_metadata(script_path: Path) -> dict[str, Any]:
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

    return tomllib.loads("\n".join(block_lines))


def package_versions_from_inline_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    deps = metadata.get("dependencies", [])
    versions = {}

    for dep in deps:
        name = normalize_requirement_name(dep)
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    return versions


def normalize_requirement_name(requirement: str) -> str:
    # Keep minimal. Replace with packaging.requirements.Requirement if available.
    raw = requirement.split(";")[0].strip()
    raw = raw.split("[")[0]
    for sep in ["==", ">=", "<=", "~=", ">", "<", " @ "]:
        if sep in raw:
            raw = raw.split(sep)[0]
    return raw.strip().lower().replace("_", "-")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_harness_version() -> str:
    try:
        return importlib.metadata.version("nwb-benchmark")
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

## 9. Profiler skeleton

```python
# nwb_benchmark/profiling.py

from __future__ import annotations

import os
import threading
import time
from typing import Any

import psutil


class Profiler:
    def __init__(self, interval_seconds: float = 0.25):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = None
        self._process = psutil.Process(os.getpid())
        self._net_start = None
        self._disk_start = None
        self._net_end = None
        self._disk_end = None

    def start(self):
        self._net_start = psutil.net_io_counters()
        self._disk_start = psutil.disk_io_counters()

        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2)

        self._net_end = psutil.net_io_counters()
        self._disk_end = psutil.disk_io_counters()

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self._sample())
            time.sleep(self.interval_seconds)

    def _sample(self) -> dict[str, Any]:
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
            }
        }

    def summary(self) -> dict[str, Any]:
        peak_rss = max(
            [s["process"]["rss_bytes"] for s in self.samples],
            default=None,
        )

        peak_total_rss = max(
            [
                s["process"]["rss_bytes"] + s["children"]["rss_bytes"]
                for s in self.samples
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
            "disk_scope": "system_delta"
        }


def diff_counters(start, end):
    if start is None or end is None:
        return None

    out = {}
    for key in start._fields:
        out[key] = getattr(end, key) - getattr(start, key)

    return out
```

---

## 10. Result bundle writer

```python
# nwb_benchmark/packaging.py

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


def write_result_bundle(
    out_dir: Path,
    manifest: dict[str, Any],
    questions: dict[str, Any],
    metadata: dict[str, Any],
    timings: dict[str, Any],
    answers: list[dict[str, Any]],
    validation: dict[str, Any],
    profile_samples: list[dict[str, Any]],
    profile_summary: dict[str, Any],
    status: str,
    error: dict[str, Any] | None,
) -> None:
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "questions.json", questions)
    write_json(out_dir / "run_metadata.json", metadata)
    write_json(out_dir / "timings.json", timings)
    write_json(out_dir / "validation.json", validation)
    write_json(out_dir / "profile_summary.json", profile_summary)
    write_json(out_dir / "status.json", {"status": status, "error": error})

    write_jsonl(out_dir / "answers.jsonl", answers)
    write_jsonl(out_dir / "profile_samples.jsonl", profile_samples)

    bundle_path = out_dir / "results_bundle.zip"

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in out_dir.iterdir():
            if path.name == bundle_path.name:
                continue
            if path.is_file():
                z.write(path, arcname=path.name)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
```

---

# Output files

```text
results/
  manifest.json
  questions.json
  run_metadata.json
  timings.json
  answers.jsonl
  validation.json
  profile_samples.jsonl
  profile_summary.json
  status.json
  results_bundle.zip
```

---

# `timings.json` shape

```json
{
  "implementation_import_duration_ns": 123456789,
  "setup_timed": false,
  "total_questions_duration_ns": 987654321,
  "per_question": [
    {
      "index": 0,
      "question_id": "q001_units_VISp_default_qc",
      "duration_ns": 123456,
      "status": "ok"
    }
  ]
}
```

---

# `answers.jsonl` shape

```json
{"index":0,"question_id":"q001_units_VISp_default_qc","answer":123,"status":"ok","error":null}
```

---

# `run_metadata.json` required fields

```text
datetime_utc
hostname
benchmark_harness_version
implementation
declared_cache_state
warmup
implementation_import_duration_ns
paths
sha256
python
platform
process
inline_script_metadata
key_package_versions
```

---

# Acceptance checklist

## Schema

* [ ] `questions.json` has top-level `dataset_id`.
* [ ] Each question has `id`, `text`, `answer`.
* [ ] No question has `order`.
* [ ] No question has `depends_on`.
* [ ] No question has `input_files`.
* [ ] Question list order is execution order.
* [ ] Manifest dataset ID matches questions dataset ID.
* [ ] All dataset files are passed to `setup()`.

## Timing

* [ ] Implementation import duration captured.
* [ ] Harness import happens only inside `if __name__ == "__main__":`.
* [ ] Setup duration is not captured.
* [ ] Warmup duration is not captured.
* [ ] Teardown duration is not captured.
* [ ] Per-question duration captured.
* [ ] Total ordered-question-loop duration captured directly.
* [ ] Validation starts only after total question-loop timer stops.
* [ ] Answers are written after timed loop.

## Implementation interface

* [ ] User script defines `IMPLEMENTATION`.
* [ ] User script defines `setup()`.
* [ ] User script defines `answer_question()`.
* [ ] User script optionally defines `warmup()`.
* [ ] User script optionally defines `teardown()`.
* [ ] `answer_question()` receives stripped question objects without `answer`.

## Metadata

* [ ] No `pip freeze`.
* [ ] Parse uv inline script metadata.
* [ ] Record declared dependencies.
* [ ] Record installed versions for declared dependencies.
* [ ] Record Python executable/version.
* [ ] Record platform info.
* [ ] Record datetime UTC.
* [ ] Record file hashes.
* [ ] Record declared cache state.

## Profiling

* [ ] Sample process CPU.
* [ ] Sample process RSS/VMS.
* [ ] Sample child process RSS/CPU.
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
* [ ] Float compare with top-level tolerance.
* [ ] Recursive compare lists/objects.
* [ ] Write `validation.json`.

## Packaging

* [ ] Write all result files.
* [ ] Create `results_bundle.zip`.
* [ ] Include manifest.
* [ ] Include questions with answers.
* [ ] Include metadata.
* [ ] Include timings.
* [ ] Include answers.
* [ ] Include validation.
* [ ] Include profile samples.
* [ ] Include profile summary.
