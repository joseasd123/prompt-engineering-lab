"""Experimento 1 — ¿Español directo o inglés + traducción?

Pensado para pegar en Google Colab. Cada bloque separado por '# %%' es una
celda sugerida — puedes pegarlo todo en una sola celda (funciona igual) o
dividirlo en varias para leer los resultados paso a paso.

Pregunta que responde: para el mismo prompt, ¿es más barato (en tokens)
preguntarle al modelo en inglés y traducir la respuesta de vuelta al
español con un método barato/gratis, o preguntarle directo en español?

Comparamos 3 formas de traducir la respuesta en inglés, NINGUNA usa el
modelo principal (así no contaminamos la medición de costo del experimento):

  1. Otra IA         -> un modelo Groq más chico/barato (llama-3.1-8b-instant)
  2. Un "tool"        -> deep-translator con motor Google Translate (gratis, sin LLM)
  3. Un traductor clásico -> deep-translator con motor MyMemory (otro motor,
     gratis, sin LLM, sin conflicto de dependencias con el SDK de Groq)

Nota técnica: NO usamos la librería `googletrans` a propósito — fija una
versión vieja de `httpx` que choca con el SDK de `groq` (que necesita una
versión moderna). `deep-translator` no tiene ese problema y trae varios
motores de traducción, así que la usamos para los métodos 2 y 3.
"""

# %% CELDA 1 — Instalar dependencias
# !pip install groq deep-translator pandas -q

# %% CELDA 2 — API key (nunca la escribas directo en el código)
import os
from getpass import getpass

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass("Pega tu GROQ_API_KEY (no se mostrará en pantalla): ")

# %% CELDA 3 — Cliente Groq mínimo (mismo patrón de prompt-engineering-lab:
# manejo de errores + retry + costo por llamada, pero compacto para Colab)
import time
import random
from dataclasses import dataclass

from groq import Groq

MODEL_PRINCIPAL = "llama-3.3-70b-versatile"  # el modelo que estamos evaluando (español vs inglés)
MODEL_TRADUCTOR_IA = "llama-3.1-8b-instant"  # modelo chico/barato, SOLO para el método 1 de traducción

# USD por 1M tokens — ajusta si tu plan de Groq deja de ser gratis.
PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.00, "output": 0.00},
    "llama-3.1-8b-instant": {"input": 0.00, "output": 0.00},
}

client = Groq()


@dataclass
class Resultado:
    texto: str
    modelo: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    costo_usd: float


def llamar(messages, model, max_tokens=400, max_reintentos=3):
    """Llama al modelo con reintentos simples ante errores transitorios."""
    prices = PRICING.get(model, {"input": 0.0, "output": 0.0})
    for intento in range(max_reintentos):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
            u = resp.usage
            costo = (u.prompt_tokens * prices["input"] + u.completion_tokens * prices["output"]) / 1_000_000
            return Resultado(
                texto=resp.choices[0].message.content,
                modelo=resp.model,
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                total_tokens=u.total_tokens,
                costo_usd=costo,
            )
        except Exception:
            if intento == max_reintentos - 1:
                raise
            time.sleep(2**intento + random.uniform(0, 1))


# %% CELDA 4 — Los 3 prompts de prueba
# Traducidos a mano (NO por IA) para que la traducción del prompt de entrada
# no contamine la medición — el experimento mide la RESPUESTA, no el input.
PROMPTS = [
    {
        "id": "resumen",
        "es": "Resume en 3 frases los beneficios de hacer ejercicio regularmente.",
        "en": "Summarize in 3 sentences the benefits of exercising regularly.",
    },
    {
        "id": "explicacion",
        "es": "Explica qué es la fotosíntesis como si tuviera 10 años.",
        "en": "Explain what photosynthesis is as if I were 10 years old.",
    },
    {
        "id": "clasificacion",
        "es": (
            "Clasifica este comentario de cliente como positivo, negativo o neutral, "
            "y justifica en una frase: 'El producto llegó tarde pero la calidad es excelente.'"
        ),
        "en": (
            "Classify this customer comment as positive, negative, or neutral, "
            "and justify in one sentence: 'The product arrived late but the quality is excellent.'"
        ),
    },
]

# %% CELDA 5 — Los 3 métodos de traducción (ninguno usa el modelo principal)
from deep_translator import GoogleTranslator, MyMemoryTranslator


def traducir_con_ia(texto_en: str):
    """Método 1: otra IA (modelo chico de Groq). SÍ consume tokens, pero de otro modelo."""
    r = llamar(
        [
            {
                "role": "system",
                "content": "Traduce el siguiente texto del inglés al español. Responde solo con la traducción, sin comentarios.",
            },
            {"role": "user", "content": texto_en},
        ],
        model=MODEL_TRADUCTOR_IA,
        max_tokens=500,
    )
    return r.texto, r.total_tokens, r.costo_usd


def traducir_con_tool(texto_en: str):
    """Método 2: 'tool' de traducción — motor Google Translate vía deep-translator. Gratis, sin LLM."""
    traduccion = GoogleTranslator(source="en", target="es").translate(texto_en)
    return traduccion, 0, 0.0


def traducir_con_traductor_clasico(texto_en: str):
    """Método 3: traductor clásico — motor MyMemory vía deep-translator. Otro motor, gratis, sin LLM."""
    try:
        traduccion = MyMemoryTranslator(source="en-GB", target="es-ES").translate(texto_en)
    except Exception as e:
        traduccion = f"[MyMemory falló: {e}]"
    return traduccion, 0, 0.0


METODOS_TRADUCCION = {
    "Inglés + traducción IA": traducir_con_ia,
    "Inglés + tool (Google Translate)": traducir_con_tool,
    "Inglés + traductor clásico (MyMemory)": traducir_con_traductor_clasico,
}

# %% CELDA 6 — Correr el experimento completo
filas = []

for p in PROMPTS:
    # A) Español directo — nuestra línea base
    r_es = llamar([{"role": "user", "content": p["es"]}], model=MODEL_PRINCIPAL)
    filas.append(
        {
            "prompt_id": p["id"],
            "variante": "Español directo",
            "prompt_tokens_llm": r_es.prompt_tokens,
            "completion_tokens_llm": r_es.completion_tokens,
            "tokens_traduccion": 0,
            "tokens_totales": r_es.total_tokens,
            "costo_llm_usd": r_es.costo_usd,
            "costo_traduccion_usd": 0.0,
            "costo_total_usd": r_es.costo_usd,
            "respuesta_final": r_es.texto,
        }
    )

    # B) Inglés (el modelo responde en inglés) + 3 formas de traducir la respuesta de vuelta
    r_en = llamar([{"role": "user", "content": p["en"]}], model=MODEL_PRINCIPAL)

    for nombre_metodo, fn in METODOS_TRADUCCION.items():
        respuesta_es, tokens_traduccion, costo_traduccion = fn(r_en.texto)
        filas.append(
            {
                "prompt_id": p["id"],
                "variante": nombre_metodo,
                "prompt_tokens_llm": r_en.prompt_tokens,
                "completion_tokens_llm": r_en.completion_tokens,
                "tokens_traduccion": tokens_traduccion,
                "tokens_totales": r_en.total_tokens + tokens_traduccion,
                "costo_llm_usd": r_en.costo_usd,
                "costo_traduccion_usd": costo_traduccion,
                "costo_total_usd": r_en.costo_usd + costo_traduccion,
                "respuesta_final": respuesta_es,
            }
        )

# %% CELDA 7 — Tabla comparativa completa
import pandas as pd

df = pd.DataFrame(filas)
pd.set_option("display.max_colwidth", 60)
print(df.sort_values(["prompt_id", "tokens_totales"]).to_string(index=False))

# %% CELDA 8 — Resumen: ¿qué variante gasta menos, en promedio de los 3 prompts?
resumen = (
    df.groupby("variante")[["tokens_totales", "costo_total_usd"]]
    .mean()
    .sort_values("tokens_totales")
)
print("\n--- Promedio por variante (menor tokens_totales = más barato) ---")
print(resumen.to_string())

ahorro_pct = (
    (resumen.loc["Español directo", "tokens_totales"] - resumen["tokens_totales"].min())
    / resumen.loc["Español directo", "tokens_totales"]
    * 100
)
print(f"\nLa variante más barata ahorra ~{ahorro_pct:.1f}% de tokens vs preguntar directo en español.")
print("(Con costo real >$0 por token, ese % se traduce directo a dólares ahorrados.)")
