"""Interactive chat loop against GroqClient — talk to the model from the terminal.

Requires GROQ_API_KEY in the environment (see .env.example / .env).
Run it with: python examples/chat.py
Type 'salir', 'exit', or 'quit' to end the conversation.
"""

from dotenv import load_dotenv

from prompt_engineering_lab import GroqClient

load_dotenv()

client = GroqClient()
messages: list[dict[str, str]] = []

print(f"Chateando con {client.default_model} (Groq). Escribe 'salir' para terminar.\n")

while True:
    try:
        user_input = input("Tú: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not user_input:
        continue
    if user_input.lower() in {"salir", "exit", "quit"}:
        break

    messages.append({"role": "user", "content": user_input})

    result = client.complete(messages=messages, max_tokens=1024)
    messages.append({"role": "assistant", "content": result.text})

    print(f"\nModelo: {result.text}\n")
    print(f"[tokens: prompt={result.prompt_tokens} completion={result.completion_tokens} | costo: ${result.cost_usd:.6f}]\n")

print(f"\nSesión terminada — {client.total_calls} llamadas, ${client.total_cost_usd:.6f} total.")
