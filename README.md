# prompt-engineering-lab

Laboratorio de prompt engineering con un cliente Python reutilizable para la API de Groq.

## `GroqClient`

`prompt_engineering_lab/groq_client.py` envuelve el SDK oficial de Groq (`groq`) con:

- **Manejo de errores tipado**: distingue errores no reintentables (`BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`) de errores transitorios (`RateLimitError`, `APIConnectionError`, errores 5xx).
- **Reintentos con backoff exponencial + jitter** para errores transitorios (configurable: `max_retries`, `base_delay`, `max_delay`).
- **Estimación de tokens**: `count_tokens()` da una aproximación (~4 caracteres/token) antes de enviar una request. Groq no expone un endpoint de conteo real como Anthropic; el conteo exacto viene en `CallResult` después de la llamada (`prompt_tokens`, `completion_tokens`, `total_tokens`).
- **Costo por llamada**: cada `complete()` devuelve un `CallResult` con el costo estimado en USD según la tabla `PRICING` (por defecto en $0.00 — actualízala con los precios reales de [groq.com/pricing](https://groq.com/pricing), que cambian con frecuencia), y acumula el total de la sesión (`client.total_cost_usd`, `client.total_calls`).

## Instalación

```bash
pip install -r requirements.txt
```

Configura tu credencial (nunca la hardcodees en el código ni la subas a git):

```bash
cp .env.example .env
# edita .env y agrega tu GROQ_API_KEY
```

`.env` está en `.gitignore` — nunca se sube al repo.

## Uso

```python
from dotenv import load_dotenv
from prompt_engineering_lab import GroqClient

load_dotenv()
client = GroqClient()  # usa GROQ_API_KEY del entorno

result = client.complete(
    messages=[{"role": "user", "content": "¿Qué es prompt engineering?"}],
    max_tokens=200,
)

print(result.text)
print(f"costo: ${result.cost_usd:.6f}")
```

Ver [`examples/basic_usage.py`](examples/basic_usage.py) para un ejemplo completo.

### Chat interactivo

Para hablar con el modelo desde la terminal (en vez de una sola llamada fija):

```bash
python examples/chat.py
```

Abre un loop de conversación (mantiene el historial de mensajes); escribe `salir` para terminar.

## Experimentos

- [`experiments/exp01_ingles_vs_espanol_colab.py`](experiments/exp01_ingles_vs_espanol_colab.py) — pensado para pegar en Google Colab. Compara el costo en tokens de preguntar directo en español vs preguntar en inglés y traducir la respuesta de vuelta con 3 métodos distintos (otra IA más chica, un tool de traducción, y un traductor clásico), sin gastar tokens del modelo principal en la traducción.

## Material de estudio

- [`docs/leccion-01-cliente-api.md`](docs/leccion-01-cliente-api.md) — lección que explica el diseño del cliente pieza por pieza (secretos/.env, wrapper, errores, backoff, tokens, costo, parámetros de generación, zero-shot vs few-shot, evaluación con métricas).
- [`docs/quiz.html`](docs/quiz.html) — quiz de 13 preguntas con feedback instantáneo. Ábrelo con doble clic en el navegador.

## Tests

```bash
pip install pytest
pytest
```

Los tests cubren el cálculo de costos y la estimación de tokens de forma pura (sin llamadas de red).

## Modelo por defecto

`llama-3.3-70b-versatile`. Los precios (USD por 1M tokens) están en `PRICING` dentro de `groq_client.py` — actualízalos según [groq.com/pricing](https://groq.com/pricing).
