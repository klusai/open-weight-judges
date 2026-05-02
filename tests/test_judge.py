"""Smoke tests for Judge — validates response parsing and validation logic.

These tests use mocked API responses rather than live Ollama calls.
For live integration tests, run with --live flag (requires Ollama running).
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from judges.judge import Judge, JudgeConfig, JudgeCallResult


@pytest.fixture
def local_config():
    return JudgeConfig(
        name="test-judge",
        model="test:latest",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.0,
        max_tokens=1024,
        use_strict_schema=True,
    )


class TestJudgeValidation:
    def test_validates_missing_fields(self):
        error = Judge._validate(
            {"grammar": 8},
            expected_fields=["grammar", "creativity"],
            score_range=(1, 10),
        )
        assert error is not None
        assert "Missing fields" in error

    def test_validates_out_of_range(self):
        error = Judge._validate(
            {"grammar": 15, "creativity": 5},
            expected_fields=["grammar", "creativity"],
            score_range=(1, 10),
        )
        assert error is not None
        assert "out of range" in error

    def test_passes_valid_scores(self):
        error = Judge._validate(
            {"grammar": 8, "creativity": 5, "moral_clarity": 7, "adherence_to_prompt": 9},
            expected_fields=["grammar", "creativity", "moral_clarity", "adherence_to_prompt"],
            score_range=(1, 10),
        )
        assert error is None

    def test_handles_nested_scores(self):
        error = Judge._validate(
            {"accuracy": {"score": 4}, "fluency": {"score": 5}},
            expected_fields=["accuracy", "fluency"],
            score_range=None,
        )
        assert error is None

    def test_no_validation_if_no_constraints(self):
        error = Judge._validate({"anything": "goes"}, None, None)
        assert error is None


class TestJudgeCallResult:
    def test_successful_result(self):
        result = JudgeCallResult(
            scores={"grammar": 8},
            input_tokens=100,
            output_tokens=50,
            latency_ms=1234.5,
            model="test:latest",
        )
        assert result.error is None
        assert result.scores["grammar"] == 8

    def test_error_result(self):
        result = JudgeCallResult(
            scores={},
            model="test:latest",
            error="API error: connection refused",
        )
        assert result.error is not None


class TestModelSafe:
    def test_sanitizes_slashes_and_colons(self, local_config):
        local_config.model = "openai/o3-mini:latest"
        judge = Judge(local_config)
        assert judge.model_safe == "openai-o3-mini-latest"
