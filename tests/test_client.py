"""Pure unit tests for cost estimation — no network calls / API key required."""

import pytest

from prompt_engineering_lab.client import ClaudeClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return ClaudeClient()


def test_estimate_cost_basic(client):
    # claude-opus-4-8: $5.00 / $25.00 per 1M tokens
    cost = client.estimate_cost("claude-opus-4-8", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(30.00)


def test_estimate_cost_with_cache(client):
    cost = client.estimate_cost(
        "claude-opus-4-8",
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    # 1M * $5 * 1.25 (write) + 1M * $5 * 0.10 (read)
    assert cost == pytest.approx(6.25 + 0.50)


def test_estimate_cost_unknown_model(client):
    with pytest.raises(ValueError):
        client.estimate_cost("not-a-real-model", input_tokens=100, output_tokens=100)
