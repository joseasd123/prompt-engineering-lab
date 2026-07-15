"""Pure unit tests for cost estimation — no network calls / API key required."""

import pytest

from prompt_engineering_lab.groq_client import GroqClient, PRICING


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return GroqClient()


def test_estimate_cost_known_model(client, monkeypatch):
    monkeypatch.setitem(PRICING, "test-model", {"input": 1.00, "output": 2.00})
    cost = client.estimate_cost("test-model", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(3.00)


def test_estimate_cost_unknown_model_defaults_to_zero(client):
    cost = client.estimate_cost("not-a-real-model", prompt_tokens=1000, completion_tokens=1000)
    assert cost == 0.0


def test_estimate_cost_unknown_model_strict_raises(client):
    with pytest.raises(ValueError):
        client.estimate_cost("not-a-real-model", prompt_tokens=1000, completion_tokens=1000, strict=True)


def test_count_tokens_rough_estimate(client):
    assert client.count_tokens("a" * 40) == 10
    assert client.count_tokens("") == 1  # minimum of 1
