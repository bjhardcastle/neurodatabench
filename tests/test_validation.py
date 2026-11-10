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
                "data_sources": {"hdf5": ["file:///tmp/test.nwb"]},
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
                "data_sources": {"hdf5": ["file:///tmp/test.nwb"]},
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

    def test_question_key_appends_params_to_id(self) -> None:
        """A question's key is its ID, plus name=value params joined by underscores."""
        scenarios: list[tuple[str, dict[str, str] | None, str]] = [
            ("no params", None, "q"),
            ("empty params", {}, "q"),
            ("one param", {"n_tables": "10"}, "q_n_tables=10"),
            ("two params", {"a": "1", "b": "x"}, "q_a=1_b=x"),
        ]
        for name, params, expected in scenarios:
            with self.subTest(name=name):
                question = neurodatabench.models.Question(
                    id="q", text="Q", answer=1, params=params
                )
                self.assertEqual(question.key, expected)

    def test_validation_matches_answers_by_question_key(self) -> None:
        """Questions sharing an ID are validated independently by their param keys."""
        benchmark = neurodatabench.models.Benchmark.model_validate(
            {
                "id": "validation",
                "data_sources": {"hdf5": ["file:///tmp/test.nwb"]},
                "questions": [
                    {"id": "load", "text": "Q", "answer": 1, "params": {"col": "a"}},
                    {"id": "load", "text": "Q", "answer": 2, "params": {"col": "b"}},
                ],
            }
        )

        neurodatabench.validation.validate_answers(
            benchmark,
            [
                {"question_id": "load_col=a", "answer": 1},
                {"question_id": "load_col=b", "answer": 2},
            ],
        )
        neurodatabench.validation.validate_answer_submission(
            benchmark, [], "load_col=b", 2
        )
        with self.assertRaisesRegex(
            neurodatabench.validation.BenchmarkValidationError,
            "load_col=b: exact_int",
        ):
            neurodatabench.validation.validate_answer_submission(
                benchmark, [], "load_col=b", 1
            )

    def test_validation_rejects_bare_id_for_parameterized_question(self) -> None:
        """The bare ID is not a valid key for a parameterized question."""
        benchmark = neurodatabench.models.Benchmark.model_validate(
            {
                "id": "validation",
                "data_sources": {"hdf5": ["file:///tmp/test.nwb"]},
                "questions": [
                    {"id": "load", "text": "Q", "answer": 1, "params": {"col": "a"}},
                ],
            }
        )

        with self.assertRaises(
            neurodatabench.validation.BenchmarkValidationError
        ) as error:
            neurodatabench.validation.validate_answers(
                benchmark, [{"question_id": "load", "answer": 1}]
            )
        self.assertEqual(
            error.exception.errors,
            [
                "load_col=a: missing_answer; submitted_answer=<missing>; actual_answer=1",
                "load: unknown_question_id; submitted_answer=1; "
                "actual_answer=<unknown_question_id>",
            ],
        )


if __name__ == "__main__":
    unittest.main()
