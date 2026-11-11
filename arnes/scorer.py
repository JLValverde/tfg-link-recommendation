"""Capa 3 del arnés: interfaz del scorer y registro de scorers.

Un scorer es una función
    puntuar(consulta, candidatos, vista) -> dict[int, float | None]
- consulta: solo {"idx", "escenario"}; el arnés no le pasa la referencia ni los ocultos.
- candidatos: lista de idx a puntuar.
- vista: arnes.vista.Vista; es la única vía de acceso al grafo y al texto.
Mayor puntuación = más recomendable. None = no se pudo puntuar. Un candidato que
falte en el resultado cuenta como None.

Añadir un scorer = escribir una función y registrarla con @registrar("nombre").

Scorers de control:
- aleatorio: puntuación uniforme con semilla fija por consulta.
- siempre_none: no puntúa nada.
- oraculo: 1 a los candidatos de la referencia y 0 al resto. Es un control que
  hace trampa a propósito: es el único que lee la referencia, desde el fichero
  de consultas.
"""

import json
import random
from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROCESADOS = RAIZ / "datos" / "procesados"
SEMILLA_ALEATORIO = 0

SCORERS = {}


def registrar(nombre):
    """Decorador: registra un scorer con ese nombre."""
    def decorador(funcion):
        if nombre in SCORERS:
            raise ValueError(f"Ya hay un scorer registrado como {nombre!r}")
        SCORERS[nombre] = funcion
        return funcion
    return decorador


@registrar("aleatorio")
def aleatorio(consulta, candidatos, vista):
    rng = random.Random(SEMILLA_ALEATORIO * 1_000_003 + consulta["idx"])
    return {c: rng.random() for c in sorted(candidatos)}


@registrar("siempre_none")
def siempre_none(consulta, candidatos, vista):
    return {c: None for c in candidatos}


@lru_cache(maxsize=None)
def _referencias(escenario):
    ruta = DIR_PROCESADOS / f"consultas_{escenario}.jsonl"
    return {c["idx"]: frozenset(c["referencia"])
            for c in (json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines())}


@registrar("oraculo")
def oraculo(consulta, candidatos, vista):
    ref = _referencias(consulta["escenario"])[consulta["idx"]]
    return {c: 1.0 if c in ref else 0.0 for c in candidatos}
