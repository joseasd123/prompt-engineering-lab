"""Versiones del prompt de extracción para el Proyecto B.

Dos ejes que se pueden combinar:
- Estrategia: zero-shot (solo instrucción) vs few-shot (con ejemplos resueltos).
- System prompt: corto (instrucción mínima) vs largo (sobre-especificado).

Tarea: extraer producto, cantidad y precio_unitario de una descripción de
compra en texto libre. Si el texto solo da el precio total de varias unidades,
el modelo debe calcular precio_unitario = total / cantidad.

- PROMPT_VERSIONS         → las 2 versiones "de producción" (system corto, el
  ganador del experimento), que usa evaluate.py.
- PROMPT_VERSIONS_SYSTEM  → el 2x2 completo (corto vs largo × zero/few), que usa
  experimento_system.py para la ablación de system prompt.
"""

from __future__ import annotations

import json
from functools import partial
from typing import Callable

# System corto: instrucción mínima. Ganó el experimento (misma accuracy que el
# largo, con ~10% menos tokens). Ver experimento_system.py.
_SYSTEM_CORTO = (
    "Eres un extractor de datos de facturas y descripciones de compra. "
    "Del texto, extrae exactamente 3 campos: producto, cantidad y precio_unitario "
    "(calcula el precio unitario si hace falta).\n\n"
    'Responde solo JSON con la forma exacta {"producto": "<texto>", "cantidad": <entero>, '
    '"precio_unitario": <entero>}. El producto va tal como aparece en el texto (no lo cambies '
    "a singular ni le agregues o quites palabras), en minúsculas. cantidad y precio_unitario "
    "son números enteros, sin puntos ni símbolos de moneda."
)

# System largo: sobre-especifica la regla de dividir el total entre la cantidad.
# Más tokens, sin mejorar la accuracy (a veces la empeora en few-shot).
_SYSTEM_LARGO = (
    "Eres un extractor de datos de facturas y descripciones de compra. "
    "Del texto, extrae exactamente 3 campos: producto, cantidad y precio_unitario "
    "(el precio POR UNIDAD, no el total). Si el texto da el precio total de varias "
    "unidades, calcula el precio_unitario dividiendo el total entre la cantidad.\n\n"
    'Responde solo JSON con la forma exacta {"producto": "<texto>", "cantidad": <entero>, '
    '"precio_unitario": <entero>}. El producto va tal como aparece en el texto (no lo cambies '
    "a singular ni le agregues o quites palabras), en minúsculas. cantidad y precio_unitario "
    "son números enteros, sin puntos ni símbolos de moneda."
)

# Ejemplos para el few-shot: NO pertenecen a casos.yaml (evita medir accuracy
# con casos que el modelo ya vio resueltos en el propio prompt). El tercero
# muestra explícitamente el caso "solo dan el total" resuelto con división.
_EJEMPLOS_FEW_SHOT = [
    ("Compré 2 mouses ópticos a $9.990 cada uno", {"producto": "mouses ópticos", "cantidad": 2, "precio_unitario": 9990}),
    ("Una silla de oficina, pagué $89.990", {"producto": "silla de oficina", "cantidad": 1, "precio_unitario": 89990}),
    ("3 audífonos con cable, el total fue $44.970", {"producto": "audífonos con cable", "cantidad": 3, "precio_unitario": 14990}),
]


def _zero_shot(system: str, texto: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": texto},
    ]


def _few_shot(system: str, texto: str) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for ejemplo_texto, ejemplo_json in _EJEMPLOS_FEW_SHOT:
        messages.append({"role": "user", "content": ejemplo_texto})
        messages.append({"role": "assistant", "content": json.dumps(ejemplo_json, ensure_ascii=False)})
    messages.append({"role": "user", "content": texto})
    return messages


# Producción: system corto (ganador). Estas son las que compara evaluate.py.
PROMPT_VERSIONS: dict[str, Callable[[str], list[dict]]] = {
    "zero_shot": partial(_zero_shot, _SYSTEM_CORTO),
    "few_shot": partial(_few_shot, _SYSTEM_CORTO),
}

# Experimento de ablación: system corto vs largo, cruzado con zero/few.
PROMPT_VERSIONS_SYSTEM: dict[str, Callable[[str], list[dict]]] = {
    "zero_shot__corto": partial(_zero_shot, _SYSTEM_CORTO),
    "zero_shot__largo": partial(_zero_shot, _SYSTEM_LARGO),
    "few_shot__corto": partial(_few_shot, _SYSTEM_CORTO),
    "few_shot__largo": partial(_few_shot, _SYSTEM_LARGO),
}
