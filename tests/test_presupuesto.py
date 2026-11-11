"""Confirmación del ADR de los tres topes de VRAM (T2.6).

- Cada registro de coste lleva el nivel de su ejecución, y el fichero de la ejecución,
  el nivel y su tope.
- La agregación falla si el pico de una ejecución con nivel supera el tope, o si no hay
  pico medido; las ejecuciones sin nivel se agregan como antes.
"""

import json

import pytest

from arnes.agregar import PresupuestoSuperado, agregar, leer, picos_vram
from arnes.coste import NIVELES, con_coste


def test_topes_del_adr():
    assert NIVELES == {"bajo": 8000, "medio": 12000, "alto": 16000}


def ejecutar(tmp_path, nivel, n=5):
    ruta = tmp_path / "coste.jsonl"
    scorer = con_coste(ruta, sistema="prueba", modo="recuperador", escenario="frio", nivel=nivel)(
        lambda consulta, candidatos, vista: {c: 0.0 for c in candidatos})
    for q in range(n):
        scorer({"idx": q, "escenario": "frio"}, [1, 2], None)
    return ruta, scorer.id_ejecucion


def test_cada_registro_lleva_el_nivel(tmp_path):
    ruta, id_ejecucion = ejecutar(tmp_path, "medio")
    filas = leer([ruta])
    assert len(filas) == 5 and all(f["nivel"] == "medio" for f in filas)
    ejecucion = json.loads((tmp_path / "ejecuciones" / f"{id_ejecucion}.json").read_text(encoding="utf-8"))
    assert ejecucion["nivel"] == "medio" and ejecucion["tope_mib"] == 12000


def test_nivel_desconocido():
    with pytest.raises(ValueError):
        con_coste("x.jsonl", sistema="prueba", modo="recuperador", nivel="enorme")


def test_agregacion_falla_si_el_pico_supera_el_tope(tmp_path):
    ruta, id_ejecucion = ejecutar(tmp_path, "bajo")
    filas = leer([ruta])
    tabla = agregar(filas, picos={id_ejecucion: 8000})  # el tope se puede alcanzar
    assert tabla[0]["nivel"] == "bajo" and tabla[0]["n_consultas"] == 2
    with pytest.raises(PresupuestoSuperado):
        agregar(filas, picos={id_ejecucion: 8001})
    with pytest.raises(PresupuestoSuperado):
        agregar(filas)  # sin pico medido no se puede comprobar


def test_picos_desde_medir_vram(tmp_path):
    ruta, id_ejecucion = ejecutar(tmp_path, "alto")
    vram = tmp_path / "vram.json"
    vram.write_text(json.dumps({"pico_mib": 16001, "ejecuciones": [id_ejecucion]}), encoding="utf-8")
    assert picos_vram([vram]) == {id_ejecucion: 16001}
    with pytest.raises(PresupuestoSuperado):
        agregar(leer([ruta]), picos=picos_vram([vram]))


def test_sin_nivel_se_agrega_como_antes(tmp_path):
    ruta, _ = ejecutar(tmp_path, None)
    tabla = agregar(leer([ruta]))
    assert "nivel" not in tabla[0] and tabla[0]["n_consultas"] == 2
