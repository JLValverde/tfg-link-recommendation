"""Variantes del contexto estructural de C2 que se prueban en desarrollo (T2.4; como mucho 3).

Se construyen insertando líneas en el prompt congelado de C1 (pipeline/prompts/c1.txt),
que no se toca:
- V1: "Connected articles: {lista}" tras el texto de A (solo en caliente) y tras el de B.
- V2: V1 más, tras el bloque de B y separada por una línea en blanco (solo en caliente),
  "Articles connected to both A and B: {comunes}".
- V3: V2 sin la línea "Connected articles" de A.
En frío la consulta no tiene vecinos: las tres variantes dan la misma plantilla (solo
la línea de B), así que frío se ejecuta una vez (con V1) y vale para las tres.

Cada variante es una especificación completa de C2 (plantillas por escenario y
parámetros de las listas); la ganadora se congela tal cual en pipeline/prompts/c2.json
(experimentos/seleccionar_c2.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import contexto  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402

LINEA_A = "Connected articles: {vecinos_consulta}\n"
LINEA_B = "Connected articles: {vecinos_candidato}\n"
LINEA_COMUNES = "Articles connected to both A and B: {comunes}\n"
FIN_A = "{texto_consulta}\n"
FIN_B = "{texto_candidato}\n"


def plantilla(c1, con_a, con_comunes):
    assert c1.count(FIN_A) == 1 and c1.count(FIN_B) == 1
    t = c1.replace(FIN_B, FIN_B + LINEA_B)
    if con_comunes:
        t = t.replace(LINEA_B, LINEA_B + "\n" + LINEA_COMUNES)
    if con_a:
        t = t.replace(FIN_A, FIN_A + LINEA_A)
    return t


def especificacion(variante, c1=None):
    c1 = ro.PLANTILLA if c1 is None else c1
    con_a, con_comunes = {1: (True, False), 2: (True, True), 3: (False, True)}[variante]
    return {"variante": variante, "grafo": contexto.GRAFO, "max_vecinos": contexto.MAX_VECINOS,
            "separador": contexto.SEPARADOR, "vacio": contexto.VACIO,
            "orden_vecinos": "grado descendente; empate: título",
            "orden_comunes": "grado ascendente; empate: título",
            "plantillas": {"caliente": plantilla(c1, con_a, con_comunes),
                           "frio": plantilla(c1, False, False)}}


VARIANTES = {v: especificacion(v) for v in (1, 2, 3)}


def sistema(variante):
    return f"c2_v{variante}"
