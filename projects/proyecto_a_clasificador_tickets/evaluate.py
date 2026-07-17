"""Proyecto A — Clasificador de tickets de soporte.

Corre las 3 versiones del prompt (ver prompts.py) contra los 50 casos
etiquetados a mano de dataset.json y compara accuracy, costo y tokens.

Uso:
    python projects/proyecto_a_clasificador_tickets/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

try:
    from .prompts import PROMPT_VERSIONS  # cuando pytest lo importa como parte del paquete projects.*
except ImportError:
    from prompts import PROMPT_VERSIONS  # cuando se corre directo: python evaluate.py

from prompt_engineering_lab import GroqClient

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "resultados.json"


def cargar_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def parsear_categoria(texto_json: str) -> str | None:
    """Extrae la categoría predicha de la respuesta del modelo.

    Devuelve None si la respuesta no es JSON válido o no trae "categoria"
    (eso también cuenta como fallo al medir accuracy, no como crash).
    """
    try:
        data = json.loads(texto_json)
    except json.JSONDecodeError:
        return None
    categoria = data.get("categoria") if isinstance(data, dict) else None
    if isinstance(categoria, str):
        return categoria.strip().lower()
    return None


def evaluar_version(client: GroqClient, build_messages, casos: list[dict]) -> dict:
    aciertos = 0
    costo_total = 0.0
    tokens_total = 0
    detalle = []

    for caso in casos:
        result = client.complete(
            messages=build_messages(caso["texto"]),
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=60,
        )
        predicha = parsear_categoria(result.text)
        correcto = predicha == caso["categoria_esperada"]
        aciertos += int(correcto)
        costo_total += result.cost_usd
        tokens_total += result.total_tokens
        detalle.append(
            {
                "id": caso["id"],
                "esperado": caso["categoria_esperada"],
                "predicho": predicha,
                "correcto": correcto,
            }
        )

    total = len(casos)
    return {
        "accuracy": aciertos / total,
        "aciertos": aciertos,
        "total": total,
        "costo_usd": costo_total,
        "tokens_prom": tokens_total / total,
        "detalle": detalle,
    }


def main() -> None:
    load_dotenv()
    client = GroqClient()
    casos = cargar_dataset()

    resultados = {}
    for nombre, build_messages in PROMPT_VERSIONS.items():
        print(f"Evaluando {nombre} ({len(casos)} casos)...")
        resultado = evaluar_version(client, build_messages, casos)
        resultados[nombre] = resultado
        errores = [d for d in resultado["detalle"] if not d["correcto"]]
        print(
            f"  accuracy={resultado['accuracy']:.0%} "
            f"({resultado['aciertos']}/{resultado['total']})  "
            f"costo=${resultado['costo_usd']:.6f}  "
            f"tokens_prom={resultado['tokens_prom']:.0f}"
        )
        if errores:
            ids = ", ".join(e["id"] for e in errores[:10])
            print(f"  fallos: {ids}{' ...' if len(errores) > 10 else ''}")

    print("\n| Prompt | Accuracy | Costo total | Tokens prom. |")
    print("|---|---|---|---|")
    for nombre, resultado in resultados.items():
        print(
            f"| {nombre} | {resultado['accuracy']:.0%} | "
            f"${resultado['costo_usd']:.6f} | {resultado['tokens_prom']:.0f} |"
        )

    print(f"\nSesión completa: {client.total_calls} llamadas, ${client.total_cost_usd:.6f}")

    RESULTS_PATH.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detalle guardado en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
