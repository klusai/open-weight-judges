"""Single-judge abstraction for LLM-based evaluation.

Wraps the OpenAI client (Ollama-compatible) with:
- Config-driven model/endpoint selection
- JSON schema enforcement for Ollama, json_object fallback for APIs
- Response validation (dimension completeness + score range checks)
- Retry with exponential backoff on parse/validation failures
- Cost and token tracking per call
- Think-tag stripping for reasoning models
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class JudgeCallResult:
    """Result of a single judge evaluation call."""

    scores: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw_response: str = ""
    error: str | None = None


@dataclass
class JudgeConfig:
    """Configuration for a single judge model."""

    name: str
    model: str
    base_url: str | None = None
    api_key: str = "ollama"
    temperature: float = 0.0
    max_tokens: int = 4096
    use_strict_schema: bool = True
    disable_thinking: bool = False
    reasoning_effort: str | None = None


class Judge:
    """Single LLM judge that scores items according to a rubric."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        self._is_local = (
            config.base_url is not None
            and "openrouter" not in (config.base_url or "")
        )
        self._is_openrouter = (
            config.base_url is not None
            and "openrouter" in config.base_url
        )

        client_kwargs: dict[str, Any] = {}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        client_kwargs["api_key"] = config.api_key or os.environ.get("EVAL_API_KEY", "ollama")

        self._client = OpenAI(**client_kwargs)

    @property
    def model_safe(self) -> str:
        return self.config.model.replace("/", "-").replace(":", "-")

    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict | None = None,
        expected_fields: list[str] | None = None,
        score_range: tuple[int, int] | None = None,
        max_retries: int = 5,
        retry_delay: float = 2.0,
    ) -> JudgeCallResult:
        """Evaluate an item and return validated, parsed scores.

        Args:
            system_prompt: System message for the judge.
            user_prompt: User message containing the item to evaluate.
            json_schema: Full JSON schema dict for strict mode (Ollama).
            expected_fields: List of score field names that must be present.
            score_range: (min, max) inclusive range for integer scores.
            max_retries: Number of attempts before giving up.
            retry_delay: Base delay in seconds (doubles each retry).
        """
        for attempt in range(1, max_retries + 1):
            result = self._call_api(system_prompt, user_prompt, json_schema)

            if result.error:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt, max_retries, self.config.model, result.error,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** (attempt - 1)))
                continue

            validation_error = self._validate(
                result.scores, expected_fields, score_range
            )
            if validation_error:
                logger.warning(
                    "Attempt %d/%d validation failed for %s: %s",
                    attempt, max_retries, self.config.model, validation_error,
                )
                result.error = validation_error
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** (attempt - 1)))
                continue

            return result

        result.error = result.error or f"Failed after {max_retries} attempts"
        return result

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict | None = None,
    ) -> JudgeCallResult:
        """Make a single API call and parse the JSON response."""
        if json_schema and self.config.use_strict_schema and self._is_local:
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        else:
            response_format = {"type": "json_object"}

        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": response_format,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.reasoning_effort and self._is_openrouter:
            create_kwargs["reasoning_effort"] = self.config.reasoning_effort

        if self.config.disable_thinking:
            create_kwargs["extra_body"] = {"think": False}

        t0 = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            return JudgeCallResult(
                scores={},
                model=self.config.model,
                error=f"API error: {exc}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        raw = completion.choices[0].message.content or ""
        text = THINK_TAG_RE.sub("", raw).strip()

        try:
            scores = json.loads(text)
        except json.JSONDecodeError as exc:
            return JudgeCallResult(
                scores={},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                model=self.config.model,
                raw_response=raw,
                error=f"JSON parse error: {exc}",
            )

        return JudgeCallResult(
            scores=scores,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=self.config.model,
            raw_response=raw,
        )

    @staticmethod
    def _validate(
        scores: dict,
        expected_fields: list[str] | None,
        score_range: tuple[int, int] | None,
    ) -> str | None:
        """Validate parsed scores. Returns error string or None if valid."""
        if expected_fields:
            missing = [f for f in expected_fields if f not in scores]
            if missing:
                return f"Missing fields: {missing}"

        if score_range and expected_fields:
            lo, hi = score_range
            for f in expected_fields:
                val = scores.get(f)
                if isinstance(val, (int, float)) and not (lo <= val <= hi):
                    return f"Score out of range: {f}={val} (expected {lo}-{hi})"

        return None
