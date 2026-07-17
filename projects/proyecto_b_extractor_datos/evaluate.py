"""Proyecto B — Extractor de datos (facturas / descripciones de productos).

El "mini framework de evaluación": carga casos de prueba desde un YAML,
corre cada versión de prompt (zero-shot vs few-shot, ver prompts.py) contra
todos los casos, y arma una tabla comparativa de accuracy y costo — incluido
el costo por 1.000 llamadas, para poder cotizar un proyecto real.

Uso:
    python projects/proyecto_b_extractor_datos/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

try:
    from .prompts import PROMPT_VERSIONS  # cuando pytest lo importa como parte del paquete projects.*
except ImportError:
    from prompts import PROMPT_VERSIONS  # cuando se corre directo: python evaluate.py

from prompt_engineering_lab import GroqClient

CASOS_PATH = Path(__file__).parent / "casos.yaml"
RESULTS_PATH = Path(__file__).parent / "resultados.json"

CAMPOS = ("producto", "cantidad", "precio_unitario")


def cargar_casos() -> list[dict]:
    data = yaml.safe_load(CASOS_PATH.read_text(encoding="utf-8"))
    return data["casos"]


def parsear_extraccion(texto_json: str) -> dict | None:
    """Extrae el dict de campos de la respuesta del modelo.

    Devuelve None si la respuesta no es JSON válido o no es un objeto
    (eso cuenta como fallo en todos los campos, no como crash).
    """
    try:
        data = json.loads(texto_json)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalizar_texto(valor: object) -> str:
    return " ".join(str(valor).strip().lower().split())


def campo_coincide(esperado: object, predicho: object) -> bool:
    """Compara un campo esperado vs predicho.

    Números: igualdad exacta (tolerando que el modelo los devuelva como
    string). Texto: coincidencia flexible — igual, o uno contiene al otro,
    normalizando mayúsculas/espacios — para tolerar variaciones menores de
    redacción sin exigir una técnica de similitud más pesada.
    """
    if predicho is None:
        return False
    if isinstance(esperado, (int, float)):
        try:
            return float(predicho) == float(esperado)
        except (TypeError, ValueError):
            return False
    esperado_norm = _normalizar_texto(esperado)
    predicho_norm = _normalizar_texto(predicho)
    return esperado_norm == predicho_norm or esperado_norm in predicho_norm or predicho_norm in esperado_norm


def evaluar_version(client: GroqClient, build_messages, casos: list[dict]) -> dict:
    aciertos_campo = 0
    total_campos = 0
    aciertos_caso = 0
    costo_total = 0.0
    tokens_total = 0
    detalle = []

    for caso in casos:
        result = client.complete(
            messages=build_messages(caso["entrada"]),
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=100,
        )
        predicho = parsear_extraccion(result.text) or {}
        coincidencias = {
            campo: campo_coincide(caso["esperado"].get(campo), predicho.get(campo)) for campo in CAMPOS
        }
        aciertos_campo += sum(coincidencias.values())
        total_campos += len(CAMPOS)
        caso_correcto = all(coincidencias.values())
        aciertos_caso += int(caso_correcto)
        costo_total += result.cost_usd
        tokens_total += result.total_tokens
        detalle.append(
            {
                "id": caso["id"],
                "esperado": caso["esperado"],
                "predicho": predicho,
                "coincidencias": coincidencias,
                "correcto": caso_correcto,
            }
        )

    total = len(casos)
    return {
        "accuracy_campo": aciertos_campo / total_campos,
        "accuracy_caso": aciertos_caso / total,
        "total": total,
        "costo_usd": costo_total,
        "costo_por_1000": (costo_total / total) * 1000,
        "tokens_prom": tokens_total / total,
        "detalle": detalle,
    }


def main() -> None:
    load_dotenv()
    client = GroqClient()
    casos = cargar_casos()

    resultados = {}
    for nombre, build_messages in PROMPT_VERSIONS.items():
        print(f"Evaluando {nombre} ({len(casos)} casos)...")
        resultado = evaluar_version(client, build_messages, casos)
        resultados[nombre] = resultado
        print(
            f"  accuracy_campo={resultado['accuracy_campo']:.0%}  "
            f"accuracy_caso={resultado['accuracy_caso']:.0%}  "
            f"costo=${resultado['costo_usd']:.6f}  "
            f"costo/1000_llamadas=${resultado['costo_por_1000']:.4f}  "
            f"tokens_prom={resultado['tokens_prom']:.0f}"
        )
        errores = [d for d in resultado["detalle"] if not d["correcto"]]
        if errores:
            ids = ", ".join(e["id"] for e in errores[:10])
            print(f"  fallos: {ids}{' ...' if len(errores) > 10 else ''}")

    print("\n| Prompt | Accuracy (campo) | Accuracy (caso) | Costo total | Costo/1.000 llamadas | Tokens prom. |")
    print("|---|---|---|---|---|---|")
    for nombre, r in resultados.items():
        print(
            f"| {nombre} | {r['accuracy_campo']:.0%} | {r['accuracy_caso']:.0%} | "
            f"${r['costo_usd']:.6f} | ${r['costo_por_1000']:.4f} | {r['tokens_prom']:.0f} |"
        )

    print(f"\nSesión completa: {client.total_calls} llamadas, ${client.total_cost_usd:.6f}")

    RESULTS_PATH.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detalle guardado en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
