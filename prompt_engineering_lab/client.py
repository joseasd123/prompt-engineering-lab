"""Reusable Claude API client: error handling, retries, token counting, and per-call cost."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import anthropic

DEFAULT_MODEL = "claude-opus-4-8"

# USD per 1M tokens. Keep in sync with shared/models.md pricing table.
PRICING: dict[str, dict[str, float]] = {
    "claude-fable-5": {"input": 10.00, "output": 50.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

# Multipliers relative to the model's input price.
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL cache write
CACHE_READ_MULTIPLIER = 0.10

# Errors worth retrying: transient network/rate-limit/server issues.
RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


@dataclass
class CallResult:
    """Result of a single Claude API call, including usage and estimated cost."""

    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float
    raw: Any = field(repr=False)


class ClaudeClient:
    """Thin wrapper around the Anthropic SDK with retries, token counting, and cost tracking.

    Credentials are resolved by the SDK from the environment (ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile) unless api_key is given.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = DEFAULT_MODEL,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        # max_retries=0 on the SDK client: retry logic lives entirely in this wrapper
        # so callers get consistent logging/backoff regardless of SDK version defaults.
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        self.default_model = default_model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.total_cost_usd = 0.0
        self.total_calls = 0

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> float:
        """Estimate the USD cost of a call from its token usage."""
        prices = PRICING.get(model)
        if prices is None:
            raise ValueError(
                f"No pricing data for model '{model}'. Add it to PRICING in client.py."
            )
        input_price = prices["input"]
        output_price = prices["output"]
        cost = (
            input_tokens * input_price
            + cache_creation_input_tokens * input_price * CACHE_WRITE_MULTIPLIER
            + cache_read_input_tokens * input_price * CACHE_READ_MULTIPLIER
        ) / 1_000_000
        cost += output_tokens * output_price / 1_000_000
        return cost

    def count_tokens(
        self,
        messages: Iterable[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
    ) -> int:
        """Count input tokens for a prospective request without sending it."""
        kwargs: dict[str, Any] = {"model": model or self.default_model, "messages": list(messages)}
        if system is not None:
            kwargs["system"] = system
        result = self._call_with_retry(self._client.messages.count_tokens, **kwargs)
        return result.input_tokens

    def complete(
        self,
        messages: Iterable[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> CallResult:
        """Send a message request and return the text, usage, and estimated cost."""
        model = model or self.default_model
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": list(messages),
            **kwargs,
        }
        if system is not None:
            request["system"] = system

        response = self._call_with_retry(self._client.messages.create, **request)

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cost = self.estimate_cost(
            model,
            usage.input_tokens,
            usage.output_tokens,
            cache_creation,
            cache_read,
        )
        self.total_cost_usd += cost
        self.total_calls += 1

        return CallResult(
            text=text,
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            cost_usd=cost,
            raw=response,
        )

    def _call_with_retry(self, fn, **kwargs: Any):
        """Call `fn(**kwargs)`, retrying transient failures with exponential backoff + jitter."""
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return fn(**kwargs)
            except anthropic.BadRequestError:
                raise  # invalid request — retrying won't help
            except anthropic.AuthenticationError:
                raise  # bad/missing credentials — retrying won't help
            except anthropic.PermissionDeniedError:
                raise  # key lacks access — retrying won't help
            except anthropic.NotFoundError:
                raise  # bad model/endpoint — retrying won't help
            except RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exception = exc
                else:
                    raise  # other 4xx — not retryable

            if attempt < self.max_retries:
                delay = min(self.base_delay * (2**attempt) + random.uniform(0, 1), self.max_delay)
                time.sleep(delay)

        assert last_exception is not None
        raise last_exception
