"""Experimento: system prompt corto vs largo (sobre-especificado).

Cruza las 2 estrategias (zero-shot / few-shot) con los 2 system prompts
(corto / largo) — un 2x2 — para medir si el system largo, que sobre-especifica
la regla de "divide el total entre la cantidad", mejora la accuracy o solo
gasta más tokens sin ganancia.

Reutiliza la maquinaria de evaluate.py (cargar_casos, evaluar_version); lo único
que cambia es qué diccionario de prompts se le pasa.

Uso:
    python projects/proyecto_b_extractor_datos/experimento_system.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

try:
    from .evaluate import cargar_casos, evaluar_version  # como paquete (pytest / import)
    from .prompts import PROMPT_VERSIONS_SYSTEM
except ImportError:
    from evaluate import cargar_casos, evaluar_version  # ejecución directa: python experimento_system.py
    from prompts import PROMPT_VERSIONS_SYSTEM

from prompt_engineering_lab import GroqClient

RESULTS_PATH = Path(__file__).parent / "resultados_experimento_system.json"


def main() -> None:
    load_dotenv()
    client = GroqClient()
    casos = cargar_casos()

    resultados = {}
    for nombre, build_messages in PROMPT_VERSIONS_SYSTEM.items():
        print(f"Evaluando {nombre} ({len(casos)} casos)...")
        r = evaluar_version(client, build_messages, casos)
        resultados[nombre] = r
        print(
            f"  accuracy_campo={r['accuracy_campo']:.0%}  "
            f"accuracy_caso={r['accuracy_caso']:.0%}  "
            f"tokens_prom={r['tokens_prom']:.0f}"
        )
        errores = [d for d in r["detalle"] if not d["correcto"]]
        if errores:
            print(f"  fallos: {', '.join(e['id'] for e in errores)}")

    print("\n| Estrategia | System | Accuracy (campo) | Accuracy (caso) | Tokens prom. |")
    print("|---|---|---|---|---|")
    for nombre, r in resultados.items():
        estrategia, system = nombre.split("__")
        print(
            f"| {estrategia} | {system} | {r['accuracy_campo']:.0%} | "
            f"{r['accuracy_caso']:.0%} | {r['tokens_prom']:.0f} |"
        )

    print(f"\nSesión completa: {client.total_calls} llamadas, ${client.total_cost_usd:.6f}")

    RESULTS_PATH.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detalle guardado en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
