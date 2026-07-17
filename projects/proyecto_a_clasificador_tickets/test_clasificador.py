"""Pure unit tests for JSON-response parsing — no network calls / API key required."""

from .evaluate import parsear_categoria


def test_parsear_categoria_json_valido():
    assert parsear_categoria('{"categoria": "facturacion"}') == "facturacion"


def test_parsear_categoria_normaliza_mayusculas_y_espacios():
    assert parsear_categoria('{"categoria": "  Facturacion  "}') == "facturacion"


def test_parsear_categoria_json_invalido_devuelve_none():
    assert parsear_categoria("esto no es json") is None


def test_parsear_categoria_sin_campo_categoria_devuelve_none():
    assert parsear_categoria('{"otro_campo": "x"}') is None


def test_parsear_categoria_valor_no_string_devuelve_none():
    assert parsear_categoria('{"categoria": 123}') is None


def test_parsear_categoria_json_no_es_objeto_devuelve_none():
    assert parsear_categoria('["facturacion"]') is None
