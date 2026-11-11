"""Baseline de fusión (T2.1): reciprocal rank fusion de BM25 y Adamic-Adar (B).

Dentro de la lista que recibe:
1. Puntúa con los scorers registrados `bm25` y `adamic_adar_B`, con la misma Vista.
2. Convierte cada puntuación en rangos 1..n sobre la lista entera con
   arnes.rangos.ordenar: mismo orden descendente, mismo desempate fijo por
   (consulta, candidato) y None (o candidato ausente, o NaN) al final.
3. Puntuación = Σ 1 / (K_RRF + rango), con K_RRF = 60 fijo.

No aplica los filtros de las tareas (neutros ya quitados por el arnés; explícitos):
rankea la lista que recibe y el arnés filtra después, como con cualquier scorer.

Solo tiene sentido en caliente, como Adamic-Adar; en frío, "no aplica".

NONE_POR_COMPONENTE cuenta, por componente, los candidatos que puntuó None; el
script lo pone a cero antes de cada ejecución y lo reporta.

rrf_c1_aa_B(...) es la fusión exploratoria de T2.8b (análisis a posteriori, fuera del
protocolo): la misma RRF con C1 en lugar de BM25, a partir de puntuaciones de C1 ya
guardadas. No se registra como scorer.
"""

from arnes.rangos import ordenar, valor
from arnes.scorer import SCORERS, registrar
from pipeline import baselines, estructurales  # noqa: F401  (registran bm25 y adamic_adar_B)

K_RRF = 60
COMPONENTES = ("bm25", "adamic_adar_B")
NONE_POR_COMPONENTE = {c: 0 for c in COMPONENTES}


def rangos(puntuaciones, candidatos, idx_consulta):
    """{candidato: rango 1..n} con el orden y el desempate del arnés. Ausente = None."""
    completas = {c: puntuaciones.get(c) for c in candidatos}
    return {c: r for r, c in enumerate(ordenar(completas, idx_consulta), start=1)}


def rrf(rangos_por_componente, candidatos):
    return {c: sum(1.0 / (K_RRF + r[c]) for r in rangos_por_componente) for c in candidatos}


@registrar("rrf_bm25_aa_B")
def rrf_bm25_aa_B(consulta, candidatos, vista):
    q = consulta["idx"]
    por_componente = []
    for nombre in COMPONENTES:
        p = SCORERS[nombre](consulta, list(candidatos), vista)
        NONE_POR_COMPONENTE[nombre] += sum(1 for c in candidatos if valor(p.get(c)) is None)
        por_componente.append(rangos(p, candidatos, q))
    return rrf(por_componente, candidatos)


def rrf_c1_aa_B(puntuaciones_c1):
    """Scorer de la fusión exploratoria (T2.8b): RRF de C1 y Adamic-Adar (B), sin llamar al modelo.

    puntuaciones_c1: {idx_consulta: {candidato: puntuación o None}} de una ejecución de C1 ya
    guardada. Mismos rangos, desempate, lista entera y K_RRF que rrf_bm25_aa_B. Falla si las
    puntuaciones de C1 de una consulta no son exactamente de la lista que da el arnés.
    El scorer lleva none_por_componente, con los None de cada componente.
    """
    none_por_componente = {"c1": 0, "adamic_adar_B": 0}

    def puntuar(consulta, candidatos, vista):
        q = consulta["idx"]
        c1 = puntuaciones_c1[q]
        if set(c1) != set(candidatos):
            raise ValueError(f"Las puntuaciones de C1 de la consulta {q} no son de la lista del arnés")
        aa = SCORERS["adamic_adar_B"](consulta, list(candidatos), vista)
        none_por_componente["c1"] += sum(1 for c in candidatos if valor(c1.get(c)) is None)
        none_por_componente["adamic_adar_B"] += sum(1 for c in candidatos if valor(aa.get(c)) is None)
        return rrf([rangos(c1, candidatos, q), rangos(aa, candidatos, q)], candidatos)

    puntuar.none_por_componente = none_por_componente
    return puntuar
