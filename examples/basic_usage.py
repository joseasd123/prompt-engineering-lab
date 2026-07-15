"""Minimal usage example for ClaudeClient.

Requires ANTHROPIC_API_KEY in the environment (see .env.example).
"""

from prompt_engineering_lab import ClaudeClient

client = ClaudeClient()

result = client.complete(
    messages=[{"role": "user", "content": "In one sentence, what is prompt engineering?"}],
    max_tokens=200,
)

print(result.text)
print(f"model={result.model} stop_reason={result.stop_reason}")
print(f"tokens: in={result.input_tokens} out={result.output_tokens}")
print(f"cost: ${result.cost_usd:.6f}")
print(f"session total: {client.total_calls} calls, ${client.total_cost_usd:.6f}")
