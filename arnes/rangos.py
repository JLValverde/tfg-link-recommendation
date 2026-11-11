"""Capa 4 del arnés: de puntuaciones a rankings, por tarea.

- Orden por puntuación descendente.
- Desempate aleatorio con semilla fija por consulta, siempre activo: cada
  candidato recibe una clave aleatoria que solo depende de (consulta, candidato),
  así que el orden relativo de dos candidatos es el mismo en todas las tareas.
- None (o NaN) cuenta como la peor puntuación; los None empatan entre sí y van al
  final, desempatados igual.

Tres tareas sobre el mismo ranking:
- completa: relevantes = referencia.
- implicita (la principal): relevantes = referencia_implicita, y antes se quitan
  de la lista los candidatos que estén en explicitos.
- implicita_solo_ancla (sensibilidad al detector): como la implícita, pero con la
  definición de explícito "solo ancla": se quitan los explicitos_solo_ancla y los
  relevantes son la referencia menos los explicitos_solo_ancla.
En todas se quitan antes los neutros_plantilla.
"""

import math
import random

SEMILLA_DESEMPATE = 20261020
TAREAS = ("completa", "implicita", "implicita_solo_ancla")


def _desempate(idx_consulta, candidato):
    return random.Random(SEMILLA_DESEMPATE * 1_000_003 + idx_consulta * 100_003 + candidato).random()


def valor(p):
    """Puntuación normalizada: float o None (None y NaN no se pudieron puntuar)."""
    if p is None:
        return None
    p = float(p)
    return None if math.isnan(p) else p


def ordenar(puntuaciones, idx_consulta):
    """Lista de candidatos de mejor a peor."""
    def clave(c):
        p = valor(puntuaciones.get(c))
        return (1, 0.0, _desempate(idx_consulta, c)) if p is None else (0, -p, _desempate(idx_consulta, c))
    return sorted(puntuaciones, key=clave)


def rankings_por_tarea(consulta, puntuaciones):
    """{tarea: (ranking, relevantes)} con los filtros de cada tarea aplicados antes de ordenar."""
    neutros = set(consulta["neutros_plantilla"])
    explicitos = set(consulta["explicitos"])
    solo_ancla = set(consulta["explicitos_solo_ancla"])
    base = {c: p for c, p in puntuaciones.items() if c not in neutros}
    impl = {c: p for c, p in base.items() if c not in explicitos}
    impl_ancla = {c: p for c, p in base.items() if c not in solo_ancla}
    q = consulta["idx"]
    return {
        "completa": (ordenar(base, q), set(consulta["referencia"])),
        "implicita": (ordenar(impl, q), set(consulta["referencia_implicita"])),
        "implicita_solo_ancla": (ordenar(impl_ancla, q), set(consulta["referencia"]) - solo_ancla),
    }
