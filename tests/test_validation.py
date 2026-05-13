"""Tests for NeuroDataBench answer validation helpers."""

from __future__ import annotations

import unittest

import neurodatabench.models
import neurodatabench.validation


class ValidationTests(unittest.TestCase):
    """Exercise standalone answer validation behavior."""

    def test_validate_answer_submission_rejects_immediate_failures(self) -> None:
        """Single-answer validation should catch wrong, duplicate, and unknown answers."""
        benchmark = neurodatabench.models.Benchmark.model_validate(
            {
                "id": "validation",
                "nwb_paths": ["file:///tmp/test.nwb"],
                "nwb_format": "hdf5",
                "questions": [
                    {"id": "first", "text": "Q", "answer": 1},
                    {"id": "second", "text": "Q", "answer": 2},
                ],
            }
        )

        scenarios: list[
            tuple[str, list[dict[str, object]], str, object, str]
        ] = [
            ("wrong", [], "first", 99, "first: exact_int"),
            (
                "duplicate",
                [{"question_id": "first", "answer": 1}],
                "first",
                1,
                "first: duplicate_answer",
            ),
            ("unknown", [], "missing", 1, "missing: unknown_question_id"),
        ]

        for name, submitted_answers, question_id, answer, expected in scenarios:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    neurodatabench.validation.BenchmarkValidationError,
                    expected,
                ):
                    neurodatabench.validation.validate_answer_submission(
                        benchmark,
                        submitted_answers,
                        question_id,
                        answer,
                    )

    def test_validate_answers_reports_all_failures(self) -> None:
        """Final validation should aggregate every answer problem."""
        benchmark = neurodatabench.models.Benchmark.model_validate(
            {
                "id": "validation",
                "nwb_paths": ["file:///tmp/test.nwb"],
                "nwb_format": "hdf5",
                "questions": [
                    {"id": "missing", "text": "Q", "answer": 1},
                    {"id": "duplicate", "text": "Q", "answer": 2},
                    {"id": "wrong", "text": "Q", "answer": 3},
                ],
            }
        )

        with self.assertRaises(
            neurodatabench.validation.BenchmarkValidationError
        ) as error:
            neurodatabench.validation.validate_answers(
                benchmark,
                [
                    {"question_id": "duplicate", "answer": 2},
                    {"question_id": "duplicate", "answer": 4},
                    {"question_id": "wrong", "answer": 99},
                    {"question_id": "unknown", "answer": 5},
                ],
            )

        self.assertEqual(
            error.exception.errors,
            [
                "missing: missing_answer; submitted_answer=<missing>; actual_answer=1",
                "duplicate: duplicate_answer; submitted_answer=[2, 4]; actual_answer=2",
                "wrong: exact_int; submitted_answer=99; actual_answer=3",
                "unknown: unknown_question_id; submitted_answer=5; "
                "actual_answer=<unknown_question_id>",
            ],
        )
        self.assertIn("- missing: missing_answer", str(error.exception))
        self.assertIn("- unknown: unknown_question_id", str(error.exception))


if __name__ == "__main__":
    unittest.main()
