"""Fuga inversa en las representaciones de los candidatos (T1.6-T1.9).

Las representaciones (embeddings, índice BM25) se precalculan con el texto
original; por consulta solo se recalculan las de los documentos cuyo texto_visible
difiere (los que tienen un ancla tapada a la consulta). Estos tests comprueban, en
50 consultas por escenario, que la puntuación de esos documentos coincide con la
de recalcularla desde cero con su texto_visible, y que ningún otro documento
cambia de texto.
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from arnes.candidatos import cargar_consultas
from arnes.vista import Vista, cargar_textos, cargar_titulos, texto_visible

RAIZ = Path(__file__).resolve().parent.parent
RUTA_ELECCION = RAIZ / "experimentos" / "resultados" / "eleccion_embeddings.json"
ESCENARIOS = ("caliente", "frio")
N_CONSULTAS = 50
SEMILLA = 48
TOL_EMBEDDINGS = 1e-4  # mismas entradas en el mismo lote: solo redondeo
RUIDO_BF16 = 0.01  # diferencia máxima observada por cambiar de lote en bfloat16: ~0,005


def muestra(escenario):
    """50 consultas con al menos un documento alterado (si no, no hay nada que comprobar)."""
    cs = [c for c in cargar_consultas(escenario) if Vista(c).documentos_alterados()]
    return random.Random(SEMILLA).sample(cs, N_CONSULTAS)


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_documentos_alterados_exactos(escenario):
    """Cambian de texto exactamente los documentos alterados (comprobado en todos los documentos con anclas)."""
    textos = cargar_textos()
    con_anclas = {d for d, (_, anclas) in textos.items() if anclas}
    for c in random.Random(SEMILLA).sample(cargar_consultas(escenario), N_CONSULTAS):
        alterados = Vista(c).documentos_alterados()
        cambian = {d for d in con_anclas if d != c["idx"] and texto_visible(d, c) != textos[d][0]}
        assert cambian == set(alterados), c["idx"]


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_bm25_alterados_desde_cero(escenario):
    from pipeline import bm25 as b
    from pipeline.baselines import bm25 as scorer_bm25

    ind = b.indice()
    titulos = cargar_titulos()

    def desde_cero(q_tokens, texto):
        """BM25 lucene escrito aparte, con las estadísticas del índice original."""
        toks = b.tokenizar(texto)
        if not toks:
            return 0.0
        norm = b.K1 * (1 - b.B + b.B * len(toks) / ind.longitud_media)
        total = 0.0
        for t in q_tokens:
            tf = toks.count(t)
            if tf and t in ind.df:
                idf = math.log(1 + (ind.n_docs - ind.df[t] + 0.5) / (ind.df[t] + 0.5))
                total += idf * tf / (tf + norm)
        return total

    comprobados = 0
    for c in muestra(escenario):
        v, q = Vista(c), c["idx"]
        qt = [t for t in b.tokens_consulta(v, q) if t in ind.df]
        todos = ind.puntuar_todos(qt, v)
        alterados = sorted(v.documentos_alterados())
        otros = random.Random(q).sample(sorted(set(ind.ids) - set(alterados) - {q}), 20)
        punt_scorer = scorer_bm25({"idx": q, "escenario": escenario}, alterados + otros, v)
        for d in alterados:
            esperado = desde_cero(qt, f"{titulos[d]}\n{texto_visible(d, c)}")
            assert todos[ind.pos[d]] == pytest.approx(esperado, rel=1e-9, abs=1e-12)
            assert punt_scorer[d] == pytest.approx(esperado, rel=1e-9, abs=1e-12)
            comprobados += 1
        # Los no alterados conservan la puntuación del índice original
        originales = ind.modelo.get_scores(qt) if qt else np.zeros(ind.n_docs)
        for d in otros:
            assert punt_scorer[d] == pytest.approx(float(originales[ind.pos[d]]), abs=1e-6)
    assert comprobados >= N_CONSULTAS


@pytest.mark.skipif(not RUTA_ELECCION.exists(), reason="aún no se ha elegido el modelo de embeddings")
@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_embeddings_alterados_desde_cero(escenario):
    from pipeline import recuperacion as r
    from pipeline.baselines import c0, modelo_elegido

    modelo = modelo_elegido()
    ind = r.indice(modelo)
    titulos = cargar_titulos()
    comprobados = 0
    for c in muestra(escenario):
        v, q = Vista(c), c["idx"]
        qv = ind.base[ind.pos[q]]
        todos = ind.puntuar_todos(v, q)
        alterados = sorted(v.documentos_alterados())
        punt_c0 = c0({"idx": q, "escenario": escenario}, alterados, v)
        # Desde cero: los textos visibles construidos aquí con texto_visible (sin la vista),
        # codificados en el mismo lote que usa el parche. Con bfloat16, codificar un texto
        # solo o en otro lote cambia el coseno hasta ~0,005 por el relleno, así que hay que
        # comparar en las mismas condiciones.
        visibles = r.codificar(modelo, [f"{titulos[d]}\n{texto_visible(d, c)}" for d in alterados])
        originales = r.codificar(modelo, [f"{titulos[d]}\n{cargar_textos()[d][0]}" for d in alterados])
        for d, e, o in zip(alterados, visibles, originales):
            esperado = float(e @ qv)
            assert abs(todos[ind.pos[d]] - esperado) < TOL_EMBEDDINGS, (q, d)
            assert abs(punt_c0[d] - esperado) < TOL_EMBEDDINGS, (q, d)
            # Si tapar el ancla cambia el coseno más que el ruido, el parche queda del lado visible
            if abs(float(o @ qv) - esperado) > RUIDO_BF16:
                assert abs(todos[ind.pos[d]] - esperado) < abs(todos[ind.pos[d]] - float(o @ qv)), (q, d)
            comprobados += 1
    assert comprobados >= N_CONSULTAS
