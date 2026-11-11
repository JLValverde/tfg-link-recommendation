"""Baselines estructurales (T1.10), registrados como scorers del arnés.

Vecinos comunes y Adamic-Adar sobre la vista NO DIRIGIDA del grafo visible de la
consulta: N(u) = sucesores visibles de u ∪ predecesores visibles de u
(GrafoVisible.vecinos_no_dirigidos).
- vecinos comunes: |N(q) ∩ N(c)|.
- Adamic-Adar: suma, sobre z ∈ N(q) ∩ N(c), de 1 / log |N(z)|, con el grado
  también en la vista no dirigida (|N(z)| >= 2, porque z es vecino de q y de c).
Un candidato sin vecinos comunes visibles puntúa 0.

Dos versiones de cada uno: sobre B (la del experimento) y sobre A (comparación),
con la vista de A de la consulta (ocultos_A). Solo tienen sentido en caliente: en
frío la consulta no tiene ningún enlace visible y se reportan como "no aplica"
(Protocolo de evaluación v2.0, §2); no se ejecutan.
"""

import math

from arnes.scorer import registrar


def _puntuar(vista, q, candidatos, grafo, medida):
    g = vista.grafo(grafo)
    nq = g.vecinos_no_dirigidos(q)
    grado = {}

    def deg(z):
        if z not in grado:
            grado[z] = len(g.vecinos_no_dirigidos(z))
        return grado[z]

    puntuaciones = {}
    for c in candidatos:
        comunes = nq & g.vecinos_no_dirigidos(c)
        if medida == "vecinos_comunes":
            puntuaciones[c] = float(len(comunes))
        else:
            puntuaciones[c] = float(sum(1.0 / math.log(deg(z)) for z in comunes))
    return puntuaciones


def _registrar(medida, grafo):
    @registrar(f"{medida}_{grafo}")
    def scorer(consulta, candidatos, vista):
        return _puntuar(vista, consulta["idx"], candidatos, grafo, medida)
    scorer.__name__ = f"{medida}_{grafo}"
    return scorer


vecinos_comunes_B = _registrar("vecinos_comunes", "B")
adamic_adar_B = _registrar("adamic_adar", "B")
vecinos_comunes_A = _registrar("vecinos_comunes", "A")
adamic_adar_A = _registrar("adamic_adar", "A")
