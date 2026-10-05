"""Reusable structured-output LLM clients for post-processing agents.

Nothing in this module knows about meetings.  A future action-plan agent can depend on
``StructuredLLM`` in exactly the same way as the minutes agent does.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError
from ..config import LLM_MAX_INPUT_TOKENS, LLM_MAX_OUTPUT_TOKENS, LLM_ESTIMATED_BYTES_PER_TOKEN, LLM_TEMPERATURE, LLM_MODEL, LLM_TOP_P, LLM_TIMEOUT_SECONDS, LLM_EXTRA_BODY
T = TypeVar("T", bound=BaseModel)
TokenCounter = Callable[[str], int]


class LLMConfigurationError(RuntimeError):
    pass


class TokenBudgetError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    """A provider response that could not satisfy the requested Pydantic model."""

    def __init__(
        self,
        message: str,
        raw_output: str,
        validation_error: Exception,
        finish_reason: str = "",
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.validation_error = validation_error
        self.finish_reason = finish_reason


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class CallMetric:
    task: str
    provider: str
    model: str
    elapsed_seconds: float
    input_tokens: int
    count_method: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str = ""


def strict_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Return the compact, reference-free schema expected by strict OpenAI endpoints."""
    schema = copy.deepcopy(model.model_json_schema())
    definitions = schema.pop("$defs", {})

    def clean(node: Any, properties_map: bool = False) -> Any:
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            resolved = copy.deepcopy(definitions[node["$ref"].rsplit("/", 1)[-1]])
            resolved.update({key: value for key, value in node.items() if key != "$ref"})
            node = resolved

        result = {
            key: clean(value, properties_map=(key == "properties"))
            for key, value in node.items()
            if key != "default" and not (key == "title" and not properties_map)
        }
        if result.get("type") == "object" or "properties" in result:
            result["required"] = list(result.get("properties", {}))
            result["additionalProperties"] = False
        return result

    return clean(schema)


def _parse_json_object(text: str, allow_local_repair: bool) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("The provider returned an empty response.")

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end and cleaned[start : end + 1] != cleaned:
        candidates.append(cleaned[start : end + 1])

    first_error: Exception | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if not isinstance(value, dict):
                raise TypeError("Expected a JSON object at the top level.")
            return value
        except (json.JSONDecodeError, TypeError) as error:
            first_error = first_error or error

    # json-repair is optional.  If it is absent, preserve the useful JSON error instead
    # of turning it into the ModuleNotFoundError seen in the notebook self-test.
    if allow_local_repair:
        try:
            import json_repair  # type: ignore[import-not-found]
        except ImportError:
            json_repair = None
        if json_repair is not None:
            for candidate in reversed(candidates):
                try:
                    value = json_repair.loads(candidate)
                    if isinstance(value, dict):
                        return value
                except Exception:
                    pass

    assert first_error is not None
    raise first_error


class StructuredLLM(ABC):
    """Generic Pydantic-output client with token checks and one-call validation.

    There is intentionally no automatic LLM repair call.  A bad response fails quickly,
    while optional local JSON repair costs no provider time.
    """

    provider = "custom"

    def __init__(
        self,
        model: str,
        *,
        token_counter: TokenCounter | None = None,
        allow_local_json_repair: bool = True,
    ) -> None:
        self.model = model
        self.max_input_tokens = LLM_MAX_INPUT_TOKENS
        self.max_output_tokens = LLM_MAX_OUTPUT_TOKENS
        self.allow_local_json_repair = allow_local_json_repair
        self._token_counter = token_counter
        self._bytes_per_token = LLM_ESTIMATED_BYTES_PER_TOKEN
        self._lock = threading.Lock()
        self._count_cache: dict[tuple[str, str, str], tuple[int, str]] = {}
        self.metrics: list[CallMetric] = []

    def estimate_text_tokens(self, text: str) -> int:
        if self._token_counter is not None:
            return max(0, int(self._token_counter(text)))
        return math.ceil(len(text.encode("utf-8")) / self._bytes_per_token)

    def count_input_tokens(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        *,
        exact: bool = True,
    ) -> tuple[int, str]:
        key = (schema.__name__, system, user)
        with self._lock:
            cached = self._count_cache.get(key)
        if exact and cached is not None:
            return cached

        schema_text = json.dumps(
            strict_schema_for(schema), ensure_ascii=False, separators=(",", ":")
        )
        envelope = f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nJSON_SCHEMA:\n{schema_text}"
        if exact:
            provider_count = self._count_tokens(envelope)
            if provider_count is not None:
                result = (provider_count, f"{self.provider}_count_tokens")
                with self._lock:
                    self._count_cache[key] = result
                return result

        method = "custom_tokenizer" if self._token_counter else "utf8_estimate"
        return self.estimate_text_tokens(envelope) + 64, method

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        task: str,
    ) -> T:
        input_tokens, method = self.count_input_tokens(system, user, schema)
        if input_tokens > self.max_input_tokens:
            raise TokenBudgetError(
                f"{task} needs {input_tokens:,} input tokens ({method}); "
                f"the limit is {self.max_input_tokens:,}."
            )

        started = time.perf_counter()
        completion = self._generate(system, user, schema)
        elapsed = time.perf_counter() - started
        metric = CallMetric(
            task=task,
            provider=self.provider,
            model=self.model,
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            count_method=method,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            finish_reason=completion.finish_reason,
        )
        with self._lock:
            self.metrics.append(metric)

        if "length" in completion.finish_reason.lower() or "max_token" in completion.finish_reason.lower():
            error = ValueError(f"Provider stopped at its output-token limit: {completion.finish_reason}")
            raise StructuredOutputError(
                "Structured output was truncated; increase max_output_tokens or use smaller chunks.",
                completion.text,
                error,
                completion.finish_reason,
            ) from error

        try:
            payload = _parse_json_object(completion.text, self.allow_local_json_repair)
            return schema.model_validate(payload)
        except (ValidationError, ValueError, TypeError) as error:
            raise StructuredOutputError(
                "Structured-output validation failed. No slow LLM repair call was made; "
                "inspect .raw_output and .validation_error.",
                completion.text,
                error,
                completion.finish_reason,
            ) from error

    def metrics_table(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(metric) for metric in self.metrics]

    def _count_tokens(self, envelope: str) -> int | None:
        return None

    @abstractmethod
    def _generate(
        self, system: str, user: str, schema: type[BaseModel]
    ) -> Completion:
        raise NotImplementedError


class GeminiLLM(StructuredLLM):
    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        *,

        thinking_budget: int = 0,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError("GOOGLE_API_KEY is missing.")
        super().__init__(LLM_MODEL, **kwargs)
        from google import genai
        from google.genai import types

        self._types = types
        self._client = client or genai.Client(api_key=api_key)
        self._temperature = LLM_TEMPERATURE
        self._thinking_budget = thinking_budget

    @classmethod
    def from_env(cls, **kwargs: Any) -> "GeminiLLM":
        return cls(os.getenv("GOOGLE_API_KEY", ""), **kwargs)

    def _count_tokens(self, envelope: str) -> int | None:
        response = self._client.models.count_tokens(model=self.model, contents=envelope)
        return int(response.total_tokens)

    def _generate(
        self, system: str, user: str, schema: type[BaseModel]
    ) -> Completion:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=self._temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=self._types.ThinkingConfig(thinking_budget=self._thinking_budget),
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self._client.models.generate_content(
            model=self.model, contents=user, config=config
        )
        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        return Completion(
            text=getattr(response, "text", "") or "",
            finish_reason=str(getattr(candidates[0], "finish_reason", "") if candidates else ""),
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
        )


class OpenAICompatibleLLM(StructuredLLM):
    """OpenAI-compatible endpoint. ``from_gemma_env`` adds OpenRouter tuning."""

    provider = "openai-compatible"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        client: Any | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError("An OpenAI-compatible API key is missing.")
        super().__init__(LLM_MODEL, **kwargs)
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=0,
            )
        self._client = client
        self._temperature = LLM_TEMPERATURE
        self._top_p = LLM_TOP_P
        self._extra_body = extra_body if extra_body is not None else (LLM_EXTRA_BODY or {})
    @classmethod
    def from_gemma_env(cls, **kwargs: Any) -> "OpenAICompatibleLLM":
        base_url = os.getenv("GEMMA_BASE_URL", "")
        if not base_url:
            raise LLMConfigurationError("GEMMA_BASE_URL is missing.")
        extra_body = kwargs.pop(
            "extra_body",
            {
                "provider": {"require_parameters": False},
                "reasoning": {"effort": "none", "exclude": True},
                "top_k": 64,
            },
        )
        return cls(
            os.getenv("GEMMA_API_KEY", ""),
            base_url=base_url,
            extra_body=extra_body,
            **kwargs,
        )

    def _generate(
        self, system: str, user: str, schema: type[BaseModel]
    ) -> Completion:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": strict_schema_for(schema),
                },
            },
        }
        if self._extra_body:
            request["extra_body"] = self._extra_body
        response = self._client.chat.completions.create(**request)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            text=choice.message.content or "",
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

