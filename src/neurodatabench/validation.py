"""Answer validation utilities for NeuroDataBench benchmark runs."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pydantic

import neurodatabench.models

_MISSING_ANSWER = object()
_UNKNOWN_ANSWER = object()


class BenchmarkValidationError(RuntimeError):
    """Raised when a benchmark run completes with incorrect answers."""

    def __init__(self, errors: str | Sequence[str]) -> None:
        """Create a validation error from one or more answer error details."""
        if isinstance(errors, str):
            self.errors = [errors]
        else:
            self.errors = list(errors)
        super().__init__(_validation_summary_message(self.errors))


def expected_answer(
    benchmark: neurodatabench.models.Benchmark,
    question_id: str,
) -> neurodatabench.models.JsonValue | None:
    """Return the expected answer for a question ID, if present."""
    question = _question_by_id(benchmark, question_id)
    if question is None:
        return None
    return question.answer


def validate_answer_submission(
    benchmark: neurodatabench.models.Benchmark,
    submitted_answers: Sequence[Mapping[str, Any]],
    question_id: str,
    answer: neurodatabench.models.JsonValue,
) -> None:
    """Validate one submitted answer against known questions and prior answers."""
    question = _question_by_id(benchmark, question_id)
    if question is None:
        raise BenchmarkValidationError(
            _validation_error_detail(
                question_id=question_id,
                reason="unknown_question_id",
                actual_answer=_UNKNOWN_ANSWER,
                submitted_answer=answer,
            )
        )

    if any(
        str(submitted["question_id"]) == question_id
        for submitted in submitted_answers
    ):
        raise BenchmarkValidationError(
            _validation_error_detail(
                question_id=question_id,
                reason="duplicate_answer",
                actual_answer=question.answer,
                submitted_answer=answer,
            )
        )

    mismatch_reason = _compare_json_value(question.answer, answer)
    if mismatch_reason is not None:
        raise BenchmarkValidationError(
            _validation_error_detail(
                question_id=question_id,
                reason=mismatch_reason,
                actual_answer=question.answer,
                submitted_answer=answer,
            )
        )


def validate_answers(
    benchmark: neurodatabench.models.Benchmark,
    submitted_answers: Sequence[Mapping[str, Any]],
) -> None:
    """Validate all submitted answers after timed execution finishes."""
    errors: list[str] = []
    by_question_id: dict[str, list[Mapping[str, Any]]] = {}
    for submitted_answer in submitted_answers:
        question_id = str(submitted_answer["question_id"])
        by_question_id.setdefault(question_id, []).append(submitted_answer)

    for question in benchmark.questions:
        matches = by_question_id.get(question.id, [])
        if not matches:
            errors.append(
                _validation_error_detail(
                    question_id=question.id,
                    reason="missing_answer",
                    actual_answer=question.answer,
                    submitted_answer=_MISSING_ANSWER,
                )
            )
            continue
        if len(matches) > 1:
            errors.append(
                _validation_error_detail(
                    question_id=question.id,
                    reason="duplicate_answer",
                    actual_answer=question.answer,
                    submitted_answer=[match["answer"] for match in matches],
                )
            )
            continue

        submitted_answer = matches[0]
        mismatch_reason = _compare_json_value(
            question.answer,
            submitted_answer["answer"],
        )
        if mismatch_reason is not None:
            errors.append(
                _validation_error_detail(
                    question_id=question.id,
                    reason=mismatch_reason,
                    actual_answer=question.answer,
                    submitted_answer=submitted_answer["answer"],
                )
            )

    known_question_ids = {question.id for question in benchmark.questions}
    for submitted_answer in submitted_answers:
        question_id = str(submitted_answer["question_id"])
        if question_id not in known_question_ids:
            errors.append(
                _validation_error_detail(
                    question_id=question_id,
                    reason="unknown_question_id",
                    actual_answer=_UNKNOWN_ANSWER,
                    submitted_answer=submitted_answer["answer"],
                )
            )

    if errors:
        raise BenchmarkValidationError(errors)


def _question_by_id(
    benchmark: neurodatabench.models.Benchmark,
    question_id: str,
) -> neurodatabench.models.Question | None:
    """Return a benchmark question by ID, if present."""
    for question in benchmark.questions:
        if question.id == question_id:
            return question
    return None


def _validation_summary_message(errors: Sequence[str]) -> str:
    """Return a validation message that lists all answer failures."""
    if len(errors) == 1:
        return f"Benchmark answers failed validation: {errors[0]}"
    error_lines = "\n".join(f"- {error}" for error in errors)
    return f"Benchmark answers failed validation:\n{error_lines}"


def _validation_error_detail(
    *,
    question_id: str,
    reason: str,
    actual_answer: Any,
    submitted_answer: Any,
) -> str:
    """Return a validation error message with answer values."""
    return (
        f"{question_id}: {reason}; "
        f"submitted_answer={_format_answer_for_error(submitted_answer)}; "
        f"actual_answer={_format_answer_for_error(actual_answer)}"
    )


def _format_answer_for_error(answer: Any) -> str:
    """Return a compact representation of an answer for validation errors."""
    if answer is _MISSING_ANSWER:
        return "<missing>"
    if answer is _UNKNOWN_ANSWER:
        return "<unknown_question_id>"
    try:
        return json.dumps(_jsonable(answer), sort_keys=True)
    except TypeError:
        return repr(answer)


def _compare_json_value(
    expected: neurodatabench.models.JsonValue,
    actual: neurodatabench.models.JsonValue,
) -> str | None:
    """Return a mismatch reason for JSON-compatible values, or None when equal."""
    if isinstance(expected, bool):
        return None if actual is expected else "exact_bool"
    if isinstance(expected, int) and not isinstance(expected, bool):
        if (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        ):
            return None
        return "exact_int"
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return "expected_float"
        return None if bool(np.isclose(float(actual), expected)) else "float_close"
    if isinstance(expected, str):
        return None if actual == expected else "exact_str"
    if expected is None:
        return None if actual is None else "exact_null"
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return "expected_list"
        if len(expected) != len(actual):
            return "list_length_mismatch"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            mismatch_reason = _compare_json_value(expected_item, actual_item)
            if mismatch_reason is not None:
                return f"list_item_{index}_{mismatch_reason}"
        return None
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return "expected_object"
        if set(expected) != set(actual):
            return "object_keys_mismatch"
        for key in expected:
            mismatch_reason = _compare_json_value(expected[key], actual[key])
            if mismatch_reason is not None:
                return f"object_value_{key}_{mismatch_reason}"
        return None
    return "unsupported_expected_type"


def _jsonable(value: object) -> object:
    """Convert dataclasses, Pydantic models, and paths into JSON-compatible values."""
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
