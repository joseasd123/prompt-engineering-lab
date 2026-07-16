"""Experimento 2 — Chat interactivo: Español directo vs Inglés + traducción.

Extiende exp01 (que probaba 3 métodos de traducción con prompts fijos) a un
chat en vivo. Escribes en español y en cada turno se comparan 2 caminos:

  A) Español directo      -> tu mensaje tal cual al modelo, responde en español.
  B) Inglés + traducción  -> tu mensaje se traduce a inglés (gratis, sin
     tokens de LLM), se envía al modelo, responde en inglés, y esa
     respuesta se traduce de vuelta a español para que la puedas leer.

Usamos solo el método de traducción que salió mejor en exp01 (deep-translator
con motor Google Translate): es gratis -no gasta tokens del modelo- y más
confiable que MyMemory (que tiene límite de caracteres en el tier gratis).
El tercer método de exp01 ("otra IA") lo dejamos fuera a propósito: gastaba
tokens de otro modelo, lo que ensuciaría la comparación de costo.

Cada turno muestra las 3 respuestas visibles (español directo, inglés
original, e inglés traducido a español) y los tokens que gastó cada camino,
más un acumulado de la sesión, para ver en vivo cuál sale más barato.

Requiere el secret GROQ_API_KEY configurado en Colab (🔑).
"""

# %% CELDA 1 — Instalar dependencias
# !pip install groq deep-translator -q

# %% CELDA 2 — API key segura (Colab Secrets, con fallback a getpass)
import os
from getpass import getpass

try:
    from google.colab import userdata

    valor = userdata.get("GROQ_API_KEY")
    if valor:
        os.environ["GROQ_API_KEY"] = valor
except Exception:
    pass

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Pega tu GROQ_API_KEY (no se mostrará en pantalla): ")

# %% CELDA 3 — Cliente Groq + traductor
from groq import Groq
from deep_translator import GoogleTranslator

MODEL_PRINCIPAL = "llama-3.3-70b-versatile"
client = Groq()

traducir_es_en = GoogleTranslator(source="es", target="en").translate
traducir_en_es = GoogleTranslator(source="en", target="es").translate


def preguntar(messages, max_tokens=400):
    resp = client.chat.completions.create(model=MODEL_PRINCIPAL, messages=messages, max_tokens=max_tokens)
    return resp.choices[0].message.content, resp.usage


# %% CELDA 4 — Loop de chat comparativo
historial_es = []  # conversación en español (camino A) — cada camino mantiene su propio contexto
historial_en = []  # conversación en inglés (camino B)

tokens_es_sesion = 0
tokens_en_sesion = 0
turnos = 0

print(f"Chat comparativo con {MODEL_PRINCIPAL}: español directo vs inglés + traducción.")
print("Escribe 'salir' para terminar.\n")

while True:
    try:
        entrada_es = input("Tú (español): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not entrada_es:
        continue
    if entrada_es.lower() in {"salir", "exit", "quit"}:
        break

    turnos += 1

    # --- Camino A: español directo ---
    historial_es.append({"role": "user", "content": entrada_es})
    respuesta_es_directa, usage_es = preguntar(historial_es)
    historial_es.append({"role": "assistant", "content": respuesta_es_directa})
    tokens_es_sesion += usage_es.total_tokens

    # --- Camino B: traducir entrada -> preguntar en inglés -> traducir respuesta ---
    entrada_en = traducir_es_en(entrada_es)
    historial_en.append({"role": "user", "content": entrada_en})
    respuesta_en, usage_en = preguntar(historial_en)
    historial_en.append({"role": "assistant", "content": respuesta_en})
    respuesta_en_traducida = traducir_en_es(respuesta_en)
    tokens_en_sesion += usage_en.total_tokens

    # --- Reporte del turno ---
    print("\n[A] Español directo:")
    print(f"    {respuesta_es_directa}")
    print(f"    tokens: {usage_es.total_tokens} (prompt={usage_es.prompt_tokens} completion={usage_es.completion_tokens})")

    print("\n[B] Inglés (respuesta original del modelo):")
    print(f"    {respuesta_en}")
    print("[B] Traducido a español:")
    print(f"    {respuesta_en_traducida}")
    print(f"    tokens: {usage_en.total_tokens} (prompt={usage_en.prompt_tokens} completion={usage_en.completion_tokens})")

    diferencia = usage_es.total_tokens - usage_en.total_tokens
    if diferencia > 0:
        ganador = "B (inglés+traducción)"
    elif diferencia < 0:
        ganador = "A (español)"
    else:
        ganador = "empate"
    print(f"\n[turno {turnos}] diferencia: {abs(diferencia)} tokens a favor de {ganador}")
    print(f"[acumulado] español: {tokens_es_sesion} tokens | inglés+traducción: {tokens_en_sesion} tokens\n")
    print("-" * 70 + "\n")

print(f"\n=== Resumen final ({turnos} turnos) ===")
print(f"Español directo:      {tokens_es_sesion} tokens")
print(f"Inglés + traducción:  {tokens_en_sesion} tokens")
if turnos > 0 and tokens_es_sesion > 0:
    ahorro = (tokens_es_sesion - tokens_en_sesion) / tokens_es_sesion * 100
    print(f"Ahorro de inglés+traducción vs español directo: {ahorro:.1f}%")
