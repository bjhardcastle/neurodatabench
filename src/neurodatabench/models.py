"""Typed data structures."""

from __future__ import annotations

import dataclasses
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

@pydantic.dataclasses.dataclass(slots=True, frozen=True)
class Benchmark:
    """A full benchmark specification, including all questions and expected answers, as well as any additional metadata needed to run the benchmark end-to-end."""
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
