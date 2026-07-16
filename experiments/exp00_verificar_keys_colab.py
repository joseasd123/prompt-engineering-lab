"""Verificación rápida de API keys — Groq, OpenAI, Hugging Face.

Pensado para pegar en Google Colab. Prueba cada key con una llamada mínima
(pocos tokens, casi sin costo) y te dice si funciona.

No usa NINGUNA key hardcodeada: las lee desde Colab Secrets (icono 🔑 en la
barra lateral izquierda) o, si no estás en Colab, las pide de forma oculta
con getpass. Configura estos secrets antes de correr:

  GROQ_API_KEY
  OPENAI_API_KEY
  HF_TOKEN

Puedes dejar cualquiera vacío (Enter) si no quieres probar esa — el reporte
te va a marcar "sin key" para esa fila en vez de fallar todo el script.
"""

# %% CELDA 1 — Instalar dependencias
# !pip install groq openai huggingface_hub -q

# %% CELDA 2 — Cargar las 3 keys de forma segura
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
    valor = getpass(f"Pega tu {nombre_secret} (no se mostrará en pantalla, Enter para saltar): ")
    if valor:
        os.environ[nombre_secret] = valor
        return valor
    return None


groq_key = cargar_key("GROQ_API_KEY")
openai_key = cargar_key("OPENAI_API_KEY")
hf_key = cargar_key("HF_TOKEN")

# %% CELDA 3 — Una función de verificación por proveedor
# Nota: un "FALLÓ" puede significar key inválida, pero también puede ser
# "key válida pero sin crédito/acceso a ese modelo" — lee el detalle.


def verificar_groq(key):
    if not key:
        return ("Groq", False, "sin key")
    try:
        from groq import Groq

        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Responde solo con: ok"}],
            max_tokens=5,
        )
        return ("Groq", True, resp.choices[0].message.content.strip())
    except Exception as e:
        return ("Groq", False, str(e))


def verificar_openai(key):
    if not key:
        return ("OpenAI", False, "sin key")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Responde solo con: ok"}],
            max_tokens=5,
        )
        return ("OpenAI", True, resp.choices[0].message.content.strip())
    except Exception as e:
        return ("OpenAI", False, str(e))


def verificar_huggingface(key):
    if not key:
        return ("Hugging Face", False, "sin key")
    try:
        from huggingface_hub import HfApi

        info = HfApi().whoami(token=key)
        return ("Hugging Face", True, f"token válido, usuario: {info.get('name', '?')}")
    except Exception as e:
        return ("Hugging Face", False, str(e))


# %% CELDA 4 — Correr las 3 verificaciones y mostrar el reporte
resultados = [
    verificar_groq(groq_key),
    verificar_openai(openai_key),
    verificar_huggingface(hf_key),
]

print(f"{'Proveedor':<14} {'Estado':<8} Detalle")
print("-" * 60)
for nombre, ok, detalle in resultados:
    estado = "OK" if ok else "FALLO"
    print(f"{nombre:<14} {estado:<8} {detalle}")
