# Lección 1 — Anatomía de un cliente de API para LLMs

Esta lección explica **qué construimos en este repo y por qué**, pieza por pieza. Al final deberías entender el código de `groq_client.py` lo suficiente como para modificarlo, y tener la base para el Proyecto B (extractor de datos) y el mini framework de evaluación.

---

## 1. El problema de la API key (secretos y git)

### Qué hicimos
- La key vive en un archivo `.env` (solo en tu PC).
- `.env` está listado en `.gitignore`, así que git **nunca** lo sube.
- El código la lee del entorno con `load_dotenv()` + el SDK.
- `.env.example` (sin la key, solo el nombre de la variable) sí se sube, para documentar qué variables necesita el proyecto.

### Por qué
Un repo público en GitHub lo puede leer cualquiera — incluyendo bots que escanean GitHub 24/7 buscando keys filtradas (encuentran una key en **minutos**). Si tu key se filtra, otro puede gastar tu cuota o dinero en tu nombre. La regla de oro:

> **El código dice "necesito una key llamada GROQ_API_KEY". El valor de esa key nunca aparece en el código.**

Esto se llama separar **configuración** de **código**. El mismo código funciona en tu PC, en el PC de un compañero, o en un servidor — cada uno con su propia `.env`.

### Cómo verificarlo
```bash
git status --porcelain --ignored | grep .env
# "!! .env"  → el !! significa "ignorado", nunca se subirá
```

---

## 2. Por qué un "wrapper" y no usar el SDK directo

Podrías llamar `groq.Groq().chat.completions.create(...)` directo en cada script. Funciona, pero cada script tendría que repetir: manejo de errores, reintentos, cálculo de costos... Un **wrapper** (envoltorio) centraliza eso una sola vez:

```
tu script  →  GroqClient (nuestro wrapper)  →  SDK de groq  →  API por internet
```

`GroqClient.complete()` hace una llamada y devuelve un `CallResult`: un objeto con el texto, los tokens usados, el costo estimado y la respuesta cruda. Todos tus experimentos futuros (Proyecto B, evaluaciones) usan esta misma pieza — por eso el enunciado pedía un "cliente **reutilizable**".

---

## 3. Manejo de errores: no todos los errores son iguales

Cuando una llamada a la API falla, la pregunta clave es: **¿reintentar ayudaría?**

| Error | HTTP | ¿Reintentar? | Por qué |
|---|---|---|---|
| `BadRequestError` | 400 | ❌ No | Tu request está mal formada. Reenviarla igual dará el mismo error. |
| `AuthenticationError` | 401 | ❌ No | Key inválida/ausente. Reintentar no arregla la key. |
| `PermissionDeniedError` | 403 | ❌ No | Tu key no tiene acceso a eso. |
| `NotFoundError` | 404 | ❌ No | Modelo o endpoint no existe (típico: typo en el nombre del modelo). |
| `RateLimitError` | 429 | ✅ Sí | Enviaste demasiadas requests por minuto. **Esperar y reintentar funciona.** |
| `APIConnectionError` | — | ✅ Sí | Falla de red (WiFi, DNS...). Suele ser transitoria. |
| Errores 5xx | 500+ | ✅ Sí | El servidor de Groq tuvo un problema. No es tu culpa; suele resolverse solo. |

En el código esto es la cadena de `except` en `_call_with_retry`: los no-reintentables hacen `raise` inmediato (fallan rápido y con un error claro), los transitorios caen al bloque de reintento.

**Antipatrón a evitar:** `except Exception: retry()`. Reintentar un 401 cincuenta veces solo te hace esperar 5 minutos para ver el mismo error que tenías al segundo cero.

---

## 4. Backoff exponencial + jitter

Cuando sí reintentamos, ¿cuánto esperar entre intentos?

```python
delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
```

Tres ideas en una línea:

1. **Exponencial** (`2 ** attempt`): espera 1s, luego 2s, 4s, 8s... Si el servidor está saturado, martillarlo cada 0.1s empeora el problema. Darle cada vez más aire aumenta la probabilidad de que se recupere.
2. **Jitter** (`random.uniform(0, 1)`): un extra aleatorio. Si 100 clientes fallan al mismo tiempo y todos reintentan exactamente a los 2.0s, vuelven a chocar todos juntos (el "efecto estampida"). El azar los desincroniza.
3. **Tope** (`min(..., max_delay)`): que la espera nunca pase de 60s, para no quedarse dormido para siempre.

Este patrón es estándar en la industria — lo vas a ver en AWS, Google Cloud, y en cualquier sistema distribuido.

---

## 5. Tokens: la unidad de medida de los LLMs

Los modelos no cobran ni miden por caracteres ni palabras, sino por **tokens** (fragmentos de ~3-4 caracteres en promedio; "hola" es 1 token, "extraordinariamente" son varios).

- **`prompt_tokens`** — lo que TÚ enviaste (instrucciones + historial + pregunta).
- **`completion_tokens`** — lo que el modelo generó.
- **`total_tokens`** — la suma.

Dos momentos donde importan:

1. **Antes de llamar** (estimación): nuestro `count_tokens()` usa la heurística `len(texto) / 4`. Es una aproximación burda porque Groq no ofrece un endpoint de conteo real (Anthropic sí lo tiene). Sirve para presupuestar, no para facturar.
2. **Después de llamar** (real): la respuesta de la API incluye el conteo exacto en `usage`. Eso es lo que reportamos en `CallResult`.

`max_tokens` limita cuántos tokens puede **generar** el modelo — es tu freno de mano contra respuestas infinitas (y costos infinitos).

---

## 6. Costo por llamada

La fórmula es simple:

```
costo = (prompt_tokens × precio_input + completion_tokens × precio_output) / 1_000_000
```

Los precios se expresan en **USD por millón de tokens**, y el output siempre cuesta más que el input (generar es más caro que leer). La tabla `PRICING` en `groq_client.py` mapea modelo → precios; como tu plan de Groq es gratuito, está en $0.00, pero la maquinaria ya está lista: cuando uses un modelo pagado, actualizas la tabla y todo tu historial de experimentos tendrá costo real.

¿Por qué acumular `total_cost_usd` y `total_calls` en el cliente? Porque en el mini framework de evaluación (Proyecto 5) vas a correr **decenas de llamadas por experimento** — y la pregunta "¿cuánto costó comparar el prompt A contra el B?" se responde sola.

---

## 7. Parámetros de generación (el "panel de control")

Estos van como argumentos extra a `complete(...)` (los pasa via `**kwargs`):

| Parámetro | Qué controla | Cuándo usarlo |
|---|---|---|
| `max_tokens` | Longitud máxima de la salida | Siempre — es tu límite de seguridad |
| `temperature` (0–2) | Aleatoriedad. 0 ≈ determinista, alto = creativo | **Extracción de datos: bajo (0–0.3).** Escritura creativa: alto |
| `top_p` | Alternativa a temperature (nucleus sampling) | Usa uno u otro, no ambos |
| `stop` | Secuencias que cortan la generación en seco | Forzar formatos ("para al ver `###`") |
| `seed` | Reproducibilidad entre llamadas idénticas | Experimentos comparables |
| `response_format` | `{"type": "json_object"}` fuerza JSON válido | **Clave para el Proyecto B** |
| `stream` | Respuesta token a token en vez de completa | Chats con sensación de "escritura en vivo" |

Para el Proyecto B la combinación ganadora es: `temperature` baja + `response_format` JSON + un buen prompt. Extraer datos es un problema de **precisión**, no de creatividad.

---

## 8. Roles de mensaje: system, user, assistant

Cada mensaje del historial tiene un rol:

- **`system`** — instrucciones de comportamiento para TODA la conversación ("Eres un extractor de datos. Respondes solo JSON."). El modelo le da más peso que a un mensaje normal.
- **`user`** — lo que escribe el humano.
- **`assistant`** — lo que respondió el modelo antes (así "recuerda" la conversación — como hicimos en `chat.py`, que acumula la lista `messages`).

La API es **sin estado** (stateless): el modelo no recuerda nada entre llamadas. La "memoria" es que tú le reenvías el historial completo cada vez. Por eso los `prompt_tokens` crecen turno a turno en un chat largo.

---

## 9. Zero-shot vs few-shot (el corazón del Proyecto B)

Dos estrategias de prompting para la misma tarea:

**Zero-shot** — solo la instrucción:
```
Extrae producto, cantidad y precio del siguiente texto. Responde en JSON.

Texto: "Compré 3 teclados mecánicos a $45.990 cada uno"
```

**Few-shot** — instrucción + ejemplos resueltos antes del caso real:
```
Extrae producto, cantidad y precio. Responde en JSON.

Texto: "2 monitores LG por $189.990 c/u"
JSON: {"producto": "monitor LG", "cantidad": 2, "precio_unitario": 189990}

Texto: "un mouse inalámbrico, salió $12.500"
JSON: {"producto": "mouse inalámbrico", "cantidad": 1, "precio_unitario": 12500}

Texto: "Compré 3 teclados mecánicos a $45.990 cada uno"
JSON:
```

Los ejemplos le muestran al modelo el **formato exacto** y cómo resolver ambigüedades (¿"un mouse" es cantidad 1? ¿el precio va con o sin puntos?). Típicamente few-shot mejora la consistencia del formato y la accuracy en casos borde — **pero cuesta más tokens por llamada** (los ejemplos viajan en cada request).

Ese trade-off (accuracy ↑ vs costo ↑) es exactamente lo que tu Proyecto B debe **medir, no suponer**.

---

## 10. Evaluación con métricas (el corazón del Proyecto 5)

"Este prompt se siente mejor" no es ingeniería. La estructura de una evaluación seria:

1. **Casos de prueba**: pares (entrada, salida esperada). Ej: 20 facturas de texto con su JSON correcto escrito a mano (el *gold standard* / *ground truth*).
2. **Correr cada versión del prompt** contra todos los casos.
3. **Comparar** salida del modelo vs salida esperada → cada caso es acierto o fallo.
4. **Métricas**:
   - **Accuracy** = aciertos / total (ej: 17/20 = 85%). Puede ser por campo extraído (más fino) o por caso completo (más estricto).
   - **Costo** = suma de `cost_usd` de todas las llamadas (¡nuestro cliente ya lo acumula!).
   - Opcional: latencia promedio.
5. **Tabla comparativa**:

   | Prompt | Accuracy | Costo total | Tokens prom. |
   |---|---|---|---|
   | zero-shot | 70% | $0.0012 | 180 |
   | few-shot (3 ej.) | 90% | $0.0031 | 470 |

¿Por qué **YAML** para los casos de prueba? Porque separa los **datos** del **código**: agregar un caso de prueba nuevo es editar un archivo de texto legible, sin tocar Python. Un caso se vería así:

```yaml
casos:
  - id: factura_simple
    entrada: "Compré 3 teclados mecánicos a $45.990 cada uno"
    esperado:
      producto: teclado mecánico
      cantidad: 3
      precio_unitario: 45990
```

---

## 11. Mapa: qué pieza del repo sirve para qué

| Pieza | Rol en los proyectos futuros |
|---|---|
| `GroqClient.complete()` | Motor de todas las llamadas |
| `CallResult.cost_usd` / tokens | Columnas de costo en la tabla comparativa |
| `client.total_cost_usd` | Costo total de un experimento completo |
| Reintentos automáticos | Que un 429 a mitad de 20 casos no mate el experimento |
| `temperature=0` + `response_format` JSON | Salidas parseables y reproducibles para medir accuracy |
| `.env` / `.gitignore` | Poder publicar el repo del proyecto sin filtrar la key |

---

## Autoevaluación

Abre `docs/quiz.html` en tu navegador (doble clic, o clic derecho → Open with Live Server en VS Code) y responde el quiz — cada pregunta te da feedback inmediato con la explicación.
