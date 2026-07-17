"""Pure unit tests for extraction parsing and field matching — no network calls / API key required."""

from .evaluate import campo_coincide, parsear_extraccion


def test_parsear_extraccion_json_valido():
    assert parsear_extraccion('{"producto": "mouse", "cantidad": 2}') == {"producto": "mouse", "cantidad": 2}


def test_parsear_extraccion_json_invalido_devuelve_none():
    assert parsear_extraccion("esto no es json") is None


def test_parsear_extraccion_lista_devuelve_none():
    assert parsear_extraccion("[1, 2, 3]") is None


def test_campo_coincide_numeros_exactos():
    assert campo_coincide(45990, 45990) is True
    assert campo_coincide(45990, 45991) is False


def test_campo_coincide_numero_predicho_como_string():
    assert campo_coincide(3, "3") is True


def test_campo_coincide_numero_predicho_none():
    assert campo_coincide(3, None) is False


def test_campo_coincide_texto_exacto():
    assert campo_coincide("mouse óptico", "mouse óptico") is True


def test_campo_coincide_texto_normaliza_mayusculas_y_espacios():
    assert campo_coincide("Monitores LG", "  monitores   lg  ") is True


def test_campo_coincide_texto_contencion_flexible():
    assert campo_coincide("silla de oficina", "silla de oficina ergonómica") is True


def test_campo_coincide_texto_distinto_no_coincide():
    assert campo_coincide("teclado mecánico", "mouse inalámbrico") is False
