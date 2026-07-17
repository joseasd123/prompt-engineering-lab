# prompt-engineering-lab

**[English](README.md) | Español**

Laboratorio de prompt engineering construido alrededor de un cliente Python reutilizable para la API de Groq, más dos casos de uso resueltos de punta a punta y un mini framework de evaluación de prompts con resultados reproducibles.

## Qué contiene

| Pieza | Qué es |
|---|---|
| `prompt_engineering_lab/` | `GroqClient` reutilizable: manejo de errores tipado, reintentos, conteo de tokens, costo por llamada. |
| `projects/proyecto_a_clasificador_tickets/` | **Proyecto A** — clasificador de tickets de soporte (3 versiones de prompt vs 50 casos etiquetados a mano). |
| `projects/proyecto_b_extractor_datos/` | **Proyecto B** — extractor de datos + framework de evaluación (casos en YAML) + un experimento de system prompt corto vs largo. |
| `docs/` | Material de estudio (lección + recorrido) y un quiz de autoevaluación. |
| `tests/`, `projects/**/test_*.py` | 20 tests puros (sin red / sin API key). |

## `GroqClient`

`prompt_engineering_lab/groq_client.py` envuelve el SDK oficial de Groq (`groq`) con:

- **Manejo de errores tipado**: distingue errores no reintentables (`BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`) de errores transitorios (`RateLimitError`, `APIConnectionError`, errores 5xx).
- **Reintentos con backoff exponencial + jitter** para errores transitorios (configurable: `max_retries`, `base_delay`, `max_delay`).
- **Estimación de tokens**: `count_tokens()` da una aproximación antes de enviar (~4 caracteres/token); el conteo exacto viene en `CallResult` después de la llamada (`prompt_tokens`, `completion_tokens`, `total_tokens`).
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

Ver [`examples/basic_usage.py`](examples/basic_usage.py) para un ejemplo completo, o [`examples/chat.py`](examples/chat.py) para un loop de chat interactivo en la terminal.

## Proyectos

### Proyecto A — Clasificador de tickets de soporte
[`projects/proyecto_a_clasificador_tickets/`](projects/proyecto_a_clasificador_tickets/)

Clasifica texto libre en una de 5 categorías (facturación, soporte técnico, cuenta/acceso, cancelación, otro) con salida JSON estructurada. Compara 3 versiones de prompt (zero-shot mínimo → zero-shot guiado con taxonomía → few-shot) contra 50 tickets etiquetados a mano, midiendo accuracy, costo y tokens de cada versión.

```bash
python projects/proyecto_a_clasificador_tickets/evaluate.py
```

### Proyecto B — Extractor de datos + framework de evaluación
[`projects/proyecto_b_extractor_datos/`](projects/proyecto_b_extractor_datos/)

Extrae `producto`, `cantidad` y `precio_unitario` de descripciones de compra en texto libre (facturas informales), incluyendo casos donde solo se da el precio total y hay que calcular el unitario. Compara zero-shot vs few-shot contra 20 casos etiquetados a mano definidos en [`casos.yaml`](projects/proyecto_b_extractor_datos/casos.yaml), midiendo accuracy por campo y por caso completo.

Este es también el **mini framework de evaluación**: los casos de prueba viven en un archivo YAML (datos separados del código), y `evaluate.py` corre todas las versiones de prompt de `PROMPT_VERSIONS` e imprime una tabla comparativa.

```bash
python projects/proyecto_b_extractor_datos/evaluate.py            # zero-shot vs few-shot
python projects/proyecto_b_extractor_datos/experimento_system.py  # system prompt corto vs largo (2x2)
```

## Resultados

Tablas comparativas (versión de prompt × accuracy × costo), generadas corriendo cada script contra la API real de Groq (modelo `llama-3.3-70b-versatile`, plan gratuito → costo real $0; las columnas de costo existen para cuando se use un modelo de pago).

**Proyecto A — clasificador de tickets** (50 casos, accuracy = ticket clasificado correctamente):

| Prompt | Accuracy | Costo total | Costo / 1.000 llamadas | Tokens prom. |
|---|---|---|---|---|
| v1 zero-shot mínimo | 2% | $0.000000 | $0.0000 | 106 |
| v2 zero-shot guiado | 96% | $0.000000 | $0.0000 | 258 |
| v3 few-shot | 96% | $0.000000 | $0.0000 | 416 |

**Proyecto B — extractor de datos** (20 casos; por campo = de los 3 campos extraídos, por caso = los 3 correctos a la vez):

| Prompt | Accuracy (campo) | Accuracy (caso) | Costo total | Costo / 1.000 llamadas | Tokens prom. |
|---|---|---|---|---|---|
| zero-shot | 100% | 100% | $0.000000 | $0.0000 | 226 |
| few-shot | 100% | 100% | $0.000000 | $0.0000 | 376 |

**Experimento — system prompt corto vs largo (sobre-especificado)** (Proyecto B, 2×2: estrategia × system prompt):

| Estrategia | System prompt | Accuracy (campo) | Accuracy (caso) | Tokens prom. |
|---|---|---|---|---|
| zero-shot | corto | 100% | 100% | 226 |
| zero-shot | largo | 100% | 100% | 252 |
| few-shot | corto | 100% | 100% | 376 |
| few-shot | largo | 98% | 95% | 402 |

**Lecturas rápidas:**
- **Proyecto A:** la instrucción clara (v2) es lo que más rinde — el few-shot (v3) no mejora sobre v2 pero cuesta 61% más tokens.
- **Proyecto B:** la combinación ganadora es un system prompt *corto*. El largo y sobre-especificado (que deletrea "divide el total entre la cantidad") nunca mejora la accuracy, cuesta ~10% más tokens, y en few-shot **rompe** un caso: `factura_monitor` ("2 monitores LG por $189.990 **c/u**") termina con su precio, que ya era unitario, dividido entre 2. Evidencia concreta de que agregar más instrucciones y más ejemplos no es gratis — puede introducir sesgos nuevos, no solo mejorar accuracy.

## Material de estudio

- [`docs/leccion-01-cliente-api.md`](docs/leccion-01-cliente-api.md) — lección que explica el diseño del cliente pieza por pieza (secretos/.env, wrapper, errores, backoff, tokens, costo, parámetros de generación, zero-shot vs few-shot, evaluación con métricas).
- [`docs/proyecto-a-clasificador-tickets.md`](docs/proyecto-a-clasificador-tickets.md) — recorrido paso a paso del Proyecto A y sus resultados reales.
- [`docs/quiz.html`](docs/quiz.html) — quiz de 13 preguntas con feedback instantáneo. Ábrelo en el navegador.

## Tests

```bash
pip install pytest
pytest
```

20 tests puros (sin llamadas de red ni API key): el cálculo de costos y la estimación de tokens del cliente, más el parseo/matching de respuestas de los proyectos A y B.

## Roadmap

- Replicar los ejercicios con un modelo open-source local vía [Ollama](https://ollama.com/) para comparar el comportamiento del modelo contra el Llama hosteado de Groq.

## Modelo por defecto

`llama-3.3-70b-versatile`. Los precios (USD por 1M tokens) están en `PRICING` dentro de `groq_client.py` — actualízalos según [groq.com/pricing](https://groq.com/pricing).
