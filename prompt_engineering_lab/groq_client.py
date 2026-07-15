"""Reusable Groq API client: error handling, retries, token counting, and per-call cost."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# USD per 1M tokens. Groq changes pricing/free-tier terms often — verify at
# https://groq.com/pricing before relying on these numbers for anything real.
# Unknown models default to $0.00 (raise via estimate_cost's `strict` flag if you
# want unknown models to fail loudly instead).
PRICING: dict[str, dict[str, float]] = {
    # Fill in / adjust from https://groq.com/pricing as needed.
    "llama-3.3-70b-versatile": {"input": 0.00, "output": 0.00},
    "llama-3.1-8b-instant": {"input": 0.00, "output": 0.00},
}

# Errors worth retrying: transient network/rate-limit/server issues.
RETRYABLE_EXCEPTIONS = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.InternalServerError,
)

# ~4 chars/token is a rough heuristic; Groq's API has no dedicated
# token-counting endpoint, so this is only used for a pre-call estimate.
_CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class CallResult:
    """Result of a single Groq API call, including usage and estimated cost."""

    text: str
    model: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    raw: Any = field(repr=False)


class GroqClient:
    """Thin wrapper around the Groq SDK with retries, token estimation, and cost tracking.

    Credentials are resolved by the SDK from GROQ_API_KEY in the environment
    unless api_key is given explicitly.
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
        self._client = groq.Groq(api_key=api_key, max_retries=0)
        self.default_model = default_model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.total_cost_usd = 0.0
        self.total_calls = 0

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        strict: bool = False,
    ) -> float:
        """Estimate the USD cost of a call from its token usage.

        If `model` isn't in PRICING: returns 0.0, unless `strict=True` (raises).
        """
        prices = PRICING.get(model)
        if prices is None:
            if strict:
                raise ValueError(
                    f"No pricing data for model '{model}'. Add it to PRICING in groq_client.py."
                )
            return 0.0
        cost = prompt_tokens * prices["input"] + completion_tokens * prices["output"]
        return cost / 1_000_000

    def count_tokens(self, text: str) -> int:
        """Rough token estimate for text you haven't sent yet.

        Groq's API has no pre-call token-counting endpoint (unlike Anthropic's
        `count_tokens`), so this is a `len(text) / 4` heuristic — use it for
        rough budgeting only. Real counts come from `CallResult` after a call.
        """
        return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)

    def complete(
        self,
        messages: Iterable[dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CallResult:
        """Send a chat completion request and return the text, usage, and estimated cost."""
        model = model or self.default_model
        request: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            **kwargs,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        response = self._call_with_retry(self._client.chat.completions.create, **request)

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        cost = self.estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        self.total_cost_usd += cost
        self.total_calls += 1

        return CallResult(
            text=text,
            model=response.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=cost,
            raw=response,
        )

    def _call_with_retry(self, fn, **kwargs: Any):
        """Call `fn(**kwargs)`, retrying transient failures with exponential backoff + jitter."""
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return fn(**kwargs)
            except groq.BadRequestError:
                raise  # invalid request — retrying won't help
            except groq.AuthenticationError:
                raise  # bad/missing credentials — retrying won't help
            except groq.PermissionDeniedError:
                raise  # key lacks access — retrying won't help
            except groq.NotFoundError:
                raise  # bad model/endpoint — retrying won't help
            except RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
            except groq.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exception = exc
                else:
                    raise  # other 4xx — not retryable

            if attempt < self.max_retries:
                delay = min(self.base_delay * (2**attempt) + random.uniform(0, 1), self.max_delay)
                time.sleep(delay)

        assert last_exception is not None
        raise last_exception
