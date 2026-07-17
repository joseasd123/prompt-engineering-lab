# Proyecto A — Clasificador de tickets de soporte

Resumen de lo construido en `projects/proyecto_a_clasificador_tickets/`: dado un texto libre de un ticket de soporte, clasificarlo en una categoría con salida JSON estructurada, iterando 3 versiones del prompt y midiendo accuracy contra 50 casos etiquetados a mano.

## Qué se creó

```
projects/proyecto_a_clasificador_tickets/
├── dataset.json      # 50 tickets etiquetados a mano (5 categorías × 10)
├── prompts.py        # las 3 versiones del prompt
├── evaluate.py        # corre las 150 llamadas y mide accuracy/costo/tokens
├── test_evaluate.py   # tests puros del parser JSON (sin red)
└── resultados.json    # detalle caso por caso, generado al correrlo
```

## Paso 1 — El dataset (`dataset.json`)

50 tickets de soporte en español, cada uno con su categoría correcta (`categoria_esperada`) puesta a mano: eso es el *gold standard* contra el que se mide accuracy. Cinco categorías, 10 tickets c/u:

- `facturacion` — cobros, facturas, medios de pago
- `soporte_tecnico` — bugs, errores, fallas
- `cuenta_acceso` — login, contraseñas, 2FA
- `cancelacion` — cancelar/pausar/dar de baja
- `otro` — todo lo demás (sugerencias, ventas, preguntas generales)

## Paso 2 — Tres versiones del prompt (`prompts.py`)

Todas piden `temperature=0` + `response_format={"type": "json_object"}` (salida determinista y parseable), pero varían cuánto guían al modelo:

- **v1 — zero-shot mínimo**: pide JSON con `"categoria"` pero **no da la lista de categorías**. El modelo tiene que inventarse sus propias etiquetas.
- **v2 — zero-shot guiado**: da la lista cerrada de 5 categorías con una definición corta de cada una, y exige usar exactamente esos nombres.
- **v3 — few-shot**: el mismo prompt guiado de v2, pero antes del ticket real muestra 5 ejemplos ya resueltos (input→JSON), como turnos de conversación `user`/`assistant`. Esos ejemplos **no están en el dataset de evaluación** — si lo estuvieran, el modelo ya habría "visto la respuesta" y la accuracy estaría inflada.

## Paso 3 — El evaluador (`evaluate.py`)

Por cada versión: corre las 50 llamadas vía `GroqClient` (reutilizando el cliente del repo, con reintentos incluidos), parsea el JSON de respuesta con `parsear_categoria()` (si no es JSON válido o no trae `categoria`, cuenta como fallo — no revienta el experimento), compara contra `categoria_esperada`, y acumula accuracy, costo (`cost_usd`) y tokens. Al final imprime la tabla comparativa y guarda el detalle caso por caso en `resultados.json`.

## Paso 4 — Resultado real (150 llamadas contra Groq)

| Prompt | Accuracy | Costo total | Tokens prom. |
|---|---|---|---|
| v1 zero-shot mínimo | **2%** (1/50) | $0.00 | 106 |
| v2 zero-shot guiado | **96%** (48/50) | $0.00 | 258 |
| v3 few-shot | **96%** (48/50) | $0.00 | 416 |

(Costo en $0 porque el plan de Groq usado es gratis — la lógica de costo ya está lista para cuando se use un modelo de pago.)

## Qué enseña esto

1. **v1 casi no sirve (2%)**: sin decirle las categorías, el modelo inventa etiquetas como `"reclamo_facturacion"` o `"problema_tecnico"` en vez de `facturacion`/`soporte_tecnico` — nunca calzan con las nuestras. Lección: en clasificación, restringir el espacio de salida (darle la lista exacta) importa más que casi cualquier otra cosa.
2. **v2 da el salto grande (2%→96%)** solo con una lista de categorías bien definida, sin ejemplos. La mayoría del valor está en las *instrucciones claras*, no en el few-shot.
3. **v3 no mejora sobre v2** (96%→96%) pero **cuesta 61% más tokens** (258→416 en promedio) porque cada llamada arrastra los 5 ejemplos. Este es exactamente el trade-off que menciona `docs/leccion-01-cliente-api.md`: few-shot no es gratis, y hay que *medirlo*, no asumir que "más contexto = mejor".
4. **Los 2 fallos que sobreviven en v2/v3 son casos genuinamente ambiguos** (ver `resultados.json` para el detalle):
   - `t034` ("Cancelé hace dos semanas y me siguen cobrando") → el modelo dice `facturacion`, la etiqueta esperada era `cancelacion`. Defendible en ambos sentidos.
   - `t043` ("¿Sacan factura para clientes fuera del país?") → el modelo dice `facturacion`, la etiqueta esperada era `otro`. Esa etiqueta es discutible — el texto menciona "factura" explícitamente. Buen recordatorio de que el *gold standard* también puede tener errores de criterio.

Además hay tests puros (`test_evaluate.py`, 6 tests, corren sin red ni API key) para el parser de JSON, siguiendo el mismo patrón que `tests/test_client.py`.

## Cómo volver a correrlo

```bash
python projects/proyecto_a_clasificador_tickets/evaluate.py
# o, parado en esa carpeta:
py evaluate.py
```

Requiere `GROQ_API_KEY` en `.env` (ver `.env.example`).
