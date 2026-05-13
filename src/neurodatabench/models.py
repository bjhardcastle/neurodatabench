"""Typed data structures for NeuroDataBench benchmarks and runs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Literal, TypeAlias

import pydantic

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = pydantic.JsonValue
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclasses.dataclass(slots=True, frozen=True)
class Question:
    """Question info passed to implementation code: the answer should be derived from data and submitted to the benchmark runner."""

    id: str
    text: str
    answer: JsonValue
    """The expected answer"""


class Benchmark(pydantic.BaseModel):
    """A full benchmark specification, including all questions and expected answers, as well as any additional metadata needed to run the benchmark end-to-end."""

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    nwb_paths: list[str]
    nwb_format: Literal["hdf5", "zarr"]
    questions: list[Question]


@dataclasses.dataclass(slots=True, frozen=True)
class Implementation:
    """Metadata about a particular implementation of the benchmark that can't be obtained programatically."""

    id: str
    local_cache: Literal["cold", "warm", False] | None
    """The state of the implementation's local cache, if any"""
    remote_cache: bool
    """Whether the implementation depends on a pre-computed cache object ranges (e.g. kerchunk/lindi)"""


@dataclasses.dataclass(slots=True)
class RunContext:
    """Runner-owned context passed to benchmark implementation hooks."""

    benchmark: Benchmark
    submit_answer: Callable[[str, JsonValue], None]


@dataclasses.dataclass(slots=True, frozen=True)
class RunPhaseTiming:
    """Start, stop, and duration timing for one measured run phase."""

    phase: str
    start_seconds: float
    stop_seconds: float
    duration_seconds: float


@dataclasses.dataclass(slots=True, frozen=True)
class AnswerSubmissionTiming:
    """Wall-clock timing for one submitted answer."""

    question_id: str
    submitted_at: str
    submitted_elapsed_seconds: float | None


class RunTimings(pydantic.BaseModel):
    """Timing summary for setup and answer submission."""

    model_config = pydantic.ConfigDict(frozen=True)

    setup_duration_ns: int
    submit_answers_duration_ns: int
    total_duration_ns: int
    phase_timings: list[RunPhaseTiming]
    answer_submissions: list[AnswerSubmissionTiming]
