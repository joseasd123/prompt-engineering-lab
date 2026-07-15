"""Minimal usage example for GroqClient.

Requires GROQ_API_KEY in the environment (see .env.example / .env).
"""

from dotenv import load_dotenv

from prompt_engineering_lab import GroqClient

load_dotenv()  # reads .env into the environment, if present

client = GroqClient()

result = client.complete(
    messages=[{"role": "user", "content": "In one sentence, what is prompt engineering?"}],
    max_tokens=200,
)

print(result.text)
print(f"model={result.model} finish_reason={result.finish_reason}")
print(f"tokens: prompt={result.prompt_tokens} completion={result.completion_tokens}")
print(f"cost: ${result.cost_usd:.6f}")
print(f"session total: {client.total_calls} calls, ${client.total_cost_usd:.6f}")
