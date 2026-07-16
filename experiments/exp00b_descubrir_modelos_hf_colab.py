"""Descubrir qué modelos responden con tu token de Hugging Face.

Pensado para pegar en Google Colab. La disponibilidad de modelos en la
Inference API gratuita de Hugging Face cambia seguido — algunos requieren
plan PRO o créditos de "Inference Providers", otros se retiran del tier
gratis sin aviso. En vez de darte una lista fija (que se desactualiza),
este script prueba una tanda de modelos candidatos EN VIVO contra tu
propio token y te dice cuáles funcionan hoy.

Requiere el secret HF_TOKEN configurado en Colab (🔑) — mismo patrón
seguro que los otros experimentos: nunca hardcodees el valor.
"""

# %% CELDA 1 — Instalar dependencias
# !pip install huggingface_hub -q

# %% CELDA 2 — Cargar el token de forma segura
import os
from getpass import getpass


def cargar_key(nombre_secret: str):
    """Colab Secrets primero; si no existe (o no estamos en Colab), la pide oculta."""
    valor = os.environ.get(nombre_secret)
    if valor:
        return valor
    try:
        from google.colab import userdata

        valor = userdata.get(nombre_secret)
        if valor:
            os.environ[nombre_secret] = valor
            return valor
    except Exception:
        pass
    valor = getpass(f"Pega tu {nombre_secret} (no se mostrará en pantalla): ")
    if valor:
        os.environ[nombre_secret] = valor
        return valor
    return None


hf_key = cargar_key("HF_TOKEN")
if not hf_key:
    raise SystemExit("Necesitas HF_TOKEN para este script.")

# %% CELDA 3 — Modelos candidatos a probar
# Mezcla deliberada: modelos chicos clásicos (históricamente disponibles
# gratis), modelos instruct populares (a veces requieren PRO/créditos), y
# un modelo de embeddings. Agrega/quita los que quieras probar.
MODELOS_A_PROBAR = [
    "gpt2",
    "distilgpt2",
    "google/flan-t5-small",
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "sentence-transformers/all-MiniLM-L6-v2",
]

# %% CELDA 4 — Probar cada modelo (chat primero, texto simple como respaldo)
from huggingface_hub import InferenceClient

client = InferenceClient(token=hf_key)


def probar_modelo(model_id: str):
    """Intenta chat_completion (modelos instruct); si falla, cae a text_generation."""
    try:
        resp = client.chat_completion(
            messages=[{"role": "user", "content": "Responde solo con: ok"}],
            model=model_id,
            max_tokens=10,
        )
        return (model_id, "chat_completion", True, resp.choices[0].message.content.strip())
    except Exception as e_chat:
        try:
            texto = client.text_generation(prompt="Di ok:", model=model_id, max_new_tokens=10)
            return (model_id, "text_generation", True, str(texto).strip())
        except Exception as e_gen:
            detalle = f"chat_completion: {e_chat} | text_generation: {e_gen}"
            return (model_id, "-", False, detalle)


resultados = [probar_modelo(m) for m in MODELOS_A_PROBAR]

# %% CELDA 5 — Reporte
print(f"{'Modelo':<42} {'Método':<16} {'Estado':<7} Detalle")
print("-" * 110)
for model_id, metodo, ok, detalle in resultados:
    estado = "OK" if ok else "FALLO"
    detalle_corto = (detalle[:70] + "...") if len(detalle) > 70 else detalle
    print(f"{model_id:<42} {metodo:<16} {estado:<7} {detalle_corto}")

print("\nLos que salgan OK son los que puedes usar hoy con tu token.")
print("Si TODOS fallan con 'not supported' o 'requires a Pro subscription',")
print("tu token es válido (ya lo confirmamos con whoami) pero tu cuenta no")
print("tiene acceso a Inference Providers en el tier gratis para esos modelos.")
