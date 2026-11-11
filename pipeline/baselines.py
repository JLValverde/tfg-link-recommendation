"""Baselines de texto (T1.9), registrados como scorers del arnés.

- c0: similitud coseno del recuperador de embeddings (modelo elegido en T1.6).
- bm25: BM25 (bm25s) con consulta = título + introducción de la consulta y
  documentos = título + texto visible.

Los dos solo leen el grafo y el texto a través de la vista: las representaciones
del texto original se precalculan una vez y, por consulta, se recalculan las de
los documentos alterados (con un ancla tapada a la consulta).
"""

import json
from pathlib import Path

import numpy as np

from arnes.scorer import registrar
from pipeline import bm25 as bm25_mod
from pipeline import recuperacion

RAIZ = Path(__file__).resolve().parent.parent
RUTA_ELECCION = RAIZ / "experimentos" / "resultados" / "eleccion_embeddings.json"


def modelo_elegido():
    return json.loads(RUTA_ELECCION.read_text(encoding="utf-8"))["ganador"]


@registrar("c0")
def c0(consulta, candidatos, vista):
    return recuperacion.indice(modelo_elegido()).puntuar(vista, consulta["idx"], candidatos)


@registrar("bm25")
def bm25(consulta, candidatos, vista):
    ind = bm25_mod.indice()
    q = consulta["idx"]
    tokens = [t for t in bm25_mod.tokens_consulta(vista, q) if t in ind.df]
    if not tokens:
        return {c: 0.0 for c in candidatos}
    todos = np.asarray(ind.modelo.get_scores(tokens), dtype=np.float64)
    alterados = set(candidatos) & set(vista.documentos_alterados())
    punt = {c: float(todos[ind.pos[c]]) for c in candidatos}
    for d in alterados:
        punt[d] = ind.puntuar_tokens(tokens, bm25_mod.tokenizar(
            bm25_mod.texto_documento(vista.titulo(d), vista.texto(d))))
    return punt
