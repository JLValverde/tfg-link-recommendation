"""Capa 5 del arnés: métricas por consulta y tarea, y curva precisión-recall.

Cada métrica es un objeto llamable m(ranking, relevantes) -> float | None con un
atributo `nombre`. Se pasan como lista al orquestador (inyectadas, no fijas).
`relevantes` es R, el conjunto completo de relevantes de la tarea, estén o no en
el ranking.

- precision@k: aciertos en el top-k / k.
- precision_norm@k: aciertos en el top-k / min(k, |R|). Se normaliza por |R| y no
  por los relevantes que hay en la lista (decisión de T1.8): en modo recuperador
  muchas listas no traen ninguno, y así el fallo del recuperador cuenta y nunca
  hay 0/0. En modo aleatorio, con más de 10 referencias, el oráculo puede quedar
  por debajo de 1.
- mrr: 1 / rango del primer relevante; 0 si no hay ninguno.
- hits@k: 1 si hay algún relevante en el top-k.

La curva precisión-recall por umbral se calcula sobre los pares (puntuación,
acierto) de todas las consultas: un candidato se predice como enlace si su
puntuación >= umbral; los None nunca se predicen.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Metrica:
    nombre: str
    funcion: Callable

    def __call__(self, ranking, relevantes):
        return self.funcion(ranking, set(relevantes))


def precision_en(k):
    return Metrica(f"precision@{k}", lambda r, rel: sum(c in rel for c in r[:k]) / k)


def precision_normalizada_en(k):
    return Metrica(f"precision_norm@{k}",
                   lambda r, rel: sum(c in rel for c in r[:k]) / min(k, len(rel)) if rel else None)


def _mrr(r, rel):
    for pos, c in enumerate(r, 1):
        if c in rel:
            return 1.0 / pos
    return 0.0


MRR = Metrica("mrr", _mrr)


def hits_en(k):
    return Metrica(f"hits@{k}", lambda r, rel: float(any(c in rel for c in r[:k])))


METRICAS_POR_DEFECTO = (precision_en(1), precision_en(5), precision_normalizada_en(1),
                        precision_normalizada_en(5), MRR, hits_en(1), hits_en(5))


def pares_puntuacion_acierto(ranking, relevantes, puntuaciones):
    """[(puntuación o None, acierto)] de cada candidato del ranking."""
    return [(puntuaciones.get(c), c in relevantes) for c in ranking]


def curva_pr(pares, max_puntos=200):
    """Curva precisión-recall por umbral: lista de {umbral, precision, recall}.

    Se evalúa en cada umbral distinto (como mucho max_puntos, repartidos por cuantiles).
    """
    total_relevantes = sum(a for _, a in pares)
    puntuados = sorted(((p, a) for p, a in pares if p is not None), key=lambda x: -x[0])
    if not puntuados or not total_relevantes:
        return []
    p = np.asarray([x[0] for x in puntuados], dtype=float)
    a = np.asarray([x[1] for x in puntuados], dtype=float)
    aciertos = np.cumsum(a)
    # último índice de cada umbral distinto (incluye todos los empatados con él)
    fin_umbral = np.flatnonzero(np.r_[p[1:] != p[:-1], True])
    if len(fin_umbral) > max_puntos:
        fin_umbral = fin_umbral[np.unique(np.linspace(0, len(fin_umbral) - 1, max_puntos).round().astype(int))]
    return [{"umbral": float(p[i]), "precision": float(aciertos[i] / (i + 1)),
             "recall": float(aciertos[i] / total_relevantes)} for i in fin_umbral]
