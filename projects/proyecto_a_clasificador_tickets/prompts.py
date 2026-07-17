"""Tres versiones del prompt de clasificación, de más simple a más guiado.

v1 — zero-shot mínimo: no le damos la lista de categorías al modelo.
v2 — zero-shot guiado: lista cerrada de categorías con una definición breve.
v3 — few-shot: mismo prompt guiado de v2 + ejemplos resueltos antes del caso real.
"""

from __future__ import annotations

from typing import Callable

CATEGORIAS = {
    "facturacion": "cobros, facturas, métodos de pago, precios",
    "soporte_tecnico": "errores, fallas o bugs de la app o el sitio",
    "cuenta_acceso": "login, contraseñas, verificación en dos pasos, bloqueos de cuenta",
    "cancelacion": "cancelar, dar de baja, pausar o cerrar la suscripción/cuenta",
    "otro": "preguntas generales, sugerencias, feedback, ventas",
}

_LISTA_CATEGORIAS = "\n".join(f"- {nombre}: {desc}" for nombre, desc in CATEGORIAS.items())

_SYSTEM_GUIADO = (
    "Eres un clasificador de tickets de soporte al cliente. "
    "Clasifica el ticket en EXACTAMENTE una de estas categorías:\n"
    f"{_LISTA_CATEGORIAS}\n\n"
    'Responde solo JSON con la forma exacta {"categoria": "<una_de_las_categorias_de_arriba>"}. '
    "Usa el nombre de la categoría tal cual está escrito arriba, en minúsculas y sin acentos."
)

# Ejemplos para el few-shot: NO pertenecen al dataset de evaluación (evita medir
# accuracy con casos que el modelo ya vio resueltos en el propio prompt).
_EJEMPLOS_FEW_SHOT = [
    ("Quiero cambiar mi tarjeta de crédito vencida por una nueva.", "facturacion"),
    ("La app se congela al abrir la sección de reportes.", "soporte_tecnico"),
    ("No me llega el código de verificación al celular nuevo.", "cuenta_acceso"),
    ("Ya cancelé hace un mes y me siguen cobrando cada mes.", "cancelacion"),
    ("¿Tienen plan con descuento para estudiantes?", "otro"),
]


def prompt_v1(texto: str) -> list[dict]:
    system = (
        "Eres un clasificador de tickets de soporte. "
        'Responde solo JSON con la forma {"categoria": "..."}.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": texto},
    ]


def prompt_v2(texto: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_GUIADO},
        {"role": "user", "content": texto},
    ]


def prompt_v3(texto: str) -> list[dict]:
    messages = [{"role": "system", "content": _SYSTEM_GUIADO}]
    for ejemplo_texto, ejemplo_categoria in _EJEMPLOS_FEW_SHOT:
        messages.append({"role": "user", "content": ejemplo_texto})
        messages.append({"role": "assistant", "content": f'{{"categoria": "{ejemplo_categoria}"}}'})
    messages.append({"role": "user", "content": texto})
    return messages


PROMPT_VERSIONS: dict[str, Callable[[str], list[dict]]] = {
    "v1_zero_shot_minimo": prompt_v1,
    "v2_zero_shot_guiado": prompt_v2,
    "v3_few_shot": prompt_v3,
}
