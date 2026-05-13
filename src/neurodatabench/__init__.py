"""NeuroDataBench benchmark runner package."""

from __future__ import annotations

import importlib.metadata
import logging

try:
    __version__ = importlib.metadata.version("neurodatabench")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

import neurodatabench.models as models
import neurodatabench.validation as validation
from neurodatabench.models import (
    AnswerSubmissionTiming,
    Benchmark,
    Implementation,
    JsonObject,
    JsonPrimitive,
    JsonValue,
    Question,
    RunContext,
    RunPhaseTiming,
    RunTimings,
)
from neurodatabench.runner import BenchmarkValidationError, main

__all__ = [
    "AnswerSubmissionTiming",
    "Benchmark",
    "BenchmarkValidationError",
    "Implementation",
    "JsonObject",
    "JsonPrimitive",
    "JsonValue",
    "Question",
    "RunContext",
    "RunPhaseTiming",
    "RunTimings",
    "__version__",
    "main",
    "models",
    "validation",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
