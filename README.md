# prompt-engineering-lab

Laboratorio de prompt engineering con un cliente Python reutilizable para la API de Claude.

## `ClaudeClient`

`prompt_engineering_lab/client.py` envuelve el SDK oficial de Anthropic (`anthropic`) con:

- **Manejo de errores tipado**: distingue errores no reintentables (`BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`) de errores transitorios (`RateLimitError`, `APIConnectionError`, errores 5xx).
- **Reintentos con backoff exponencial + jitter** para errores transitorios (configurable: `max_retries`, `base_delay`, `max_delay`).
- **Conteo de tokens** vía `count_tokens()` antes de enviar una request.
- **Costo por llamada**: cada `complete()` devuelve un `CallResult` con tokens de entrada/salida, tokens de cache (creación/lectura) y el costo estimado en USD, además de acumular el total de la sesión (`client.total_cost_usd`, `client.total_calls`).

## Instalación

```bash
pip install -r requirements.txt
```

Configura tu credencial (no la hardcodees en el código):

```bash
cp .env.example .env
# edita .env y agrega tu ANTHROPIC_API_KEY
```

## Uso

```python
from prompt_engineering_lab import ClaudeClient

client = ClaudeClient()  # usa ANTHROPIC_API_KEY del entorno

result = client.complete(
    messages=[{"role": "user", "content": "¿Qué es prompt engineering?"}],
    max_tokens=200,
)

print(result.text)
print(f"costo: ${result.cost_usd:.6f}")
```

Ver [`examples/basic_usage.py`](examples/basic_usage.py) para un ejemplo completo.

## Tests

```bash
pip install pytest
pytest
```

Los tests cubren el cálculo de costos de forma pura (sin llamadas de red).

## Modelo por defecto

`claude-opus-4-8`. Los precios (USD por 1M tokens) están en `PRICING` dentro de `client.py` — actualízalos si cambian los precios de la API.
