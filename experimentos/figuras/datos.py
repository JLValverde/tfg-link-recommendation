"""Lectura de resultados para las figuras y tablas de T2.8. No calcula métricas: solo las lee.

- Métricas con IC y diferencias pareadas: experimentos/resultados/tabla_principal.json
  (1800 consultas, modo recuperador o aleatorio con la semilla 1).
- Latencia por consulta (p50): de {sistema}_analisis.json en los sistemas con modelo de
  lenguaje (1800 consultas, sin las 3 de calentamiento) y de coste_{grupo}_agregado.json en
  los demás (las 2000 consultas menos las 3 de calentamiento: es lo que se midió).
- Curvas precisión-recall por umbral: metricas_{sistema}_recuperador_{escenario}.json
  (curva_pr_sin_desarrollo; no versionados: se rehacen con --solo-resumen).
- Techo del recuperador: recall_recuperador.json y recall_recuperador_por_consulta.jsonl.
- Análisis exploratorio (a posteriori) de T2.8b: la clave "exploratorio_a_posteriori" de
  tabla_principal.json, que solo lee metrica_exploratoria(); metrica() y diferencia() no la ven.
"""

import json
from functools import cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
DIR_RESULTADOS = RAIZ / "experimentos" / "resultados"
ESCENARIOS = ("frio", "caliente")
NOMBRE_ESC = {"caliente": "caliente", "frio": "frío"}
TAREAS = ("completa", "implicita", "implicita_solo_ancla")
NOMBRE_TAREA = {"completa": "completa", "implicita": "implícita", "implicita_solo_ancla": "implícita solo ancla"}
METRICAS = ("precision@1", "precision_norm@5", "mrr", "hits@5")
NOMBRE_METRICA = {"precision@1": "precision@1", "precision_norm@5": "precision_norm@5", "mrr": "MRR",
                  "hits@5": "hits@5"}
# Sistemas con modelo de lenguaje: su análisis y su modelo
ANALISIS = {"c1": "c1_analisis.json", "c1_7b": "c1_7b_analisis.json", "c1_14b": "c1_14b_analisis.json",
            "c2": "c2_analisis.json", "c2_7b": "c2_7b_analisis.json", "c2_14b": "c2_14b_analisis.json"}
MODELO = {"c1": "3b", "c1_7b": "7b", "c1_14b": "14b", "c2": "3b", "c2_7b": "7b", "c2_14b": "14b"}
GRUPO_COSTE = {"c0": "baselines", "bm25": "baselines", "vecinos_comunes_B": "estructurales",
               "adamic_adar_B": "estructurales", "vecinos_comunes_A": "estructurales",
               "adamic_adar_A": "estructurales", "rrf_bm25_aa_B": "fusion",
               "aleatorio": "controles", "oraculo": "controles"}


def leer(nombre):
    return json.loads((DIR_RESULTADOS / nombre).read_text(encoding="utf-8"))


@cache
def _tabla():
    t = leer("tabla_principal.json")
    filas = {(r["sistema"], r["modo"], r["escenario"], r["tarea"]): r for r in t["tabla"]}
    difs = {(d["a"], d["b"], d["escenario"], d["tarea"]): d for d in t["diferencias"]}
    return filas, difs, t["n_consultas"]


def n_consultas():
    return _tabla()[2]


def metrica(sistema, escenario, metrica="mrr", tarea="implicita", modo="recuperador"):
    """{"media", "ic95"}, o None si el sistema no se ejecutó, no aplica o no está en la tabla."""
    r = _tabla()[0].get((sistema, modo, escenario, tarea))
    if r is None or r.get("no_aplica") or r.get("no_ejecutado"):
        return None
    assert r["n_consultas"] == 1800, (sistema, escenario, tarea)
    return r[metrica]


def estado(sistema, escenario, modo="recuperador"):
    """"ok", "no aplica", "no ejecutado" o None si no está en la tabla."""
    r = _tabla()[0].get((sistema, modo, escenario, "implicita"))
    if r is None:
        return None
    return "no aplica" if r.get("no_aplica") else "no ejecutado" if r.get("no_ejecutado") else "ok"


def disponible(sistema, escenario, modo="recuperador"):
    return estado(sistema, escenario, modo) == "ok"


def diferencia(a, b, escenario, metrica="mrr", tarea="implicita"):
    """{"diferencia", "ic95", "n"} de A - B (bootstrap pareado), o None si no está en la tabla."""
    d = _tabla()[1].get((a, b, escenario, tarea))
    if d is None:
        return None
    assert d[metrica]["n"] == 1800, (a, b, escenario, tarea)
    return d[metrica]


def exploratorio():
    """(sistemas, filas) del análisis exploratorio (a posteriori) de T2.8b; ([], {}) si no lo hay."""
    ex = leer("tabla_principal.json").get("exploratorio_a_posteriori")
    if ex is None:
        return [], {}
    filas = {(r["sistema"], r["escenario"], r["tarea"]): r for r in ex["tabla"]}
    return list(dict.fromkeys(r["sistema"] for r in ex["tabla"])), filas


def metrica_exploratoria(sistema, escenario, metrica="mrr", tarea="implicita"):
    r = exploratorio()[1][(sistema, escenario, tarea)]
    assert r["n_consultas"] == 1800, (sistema, escenario, tarea)
    return r[metrica]


@cache
def analisis(sistema):
    ruta = DIR_RESULTADOS / ANALISIS[sistema]
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def latencia_p50_ms(sistema, escenario):
    if sistema in ANALISIS:
        a = analisis(sistema)
        return a["por_escenario"][escenario]["coste"]["latencia_ms_por_consulta"]["p50"] if a else None
    for f in leer(f"coste_{GRUPO_COSTE[sistema]}_agregado.json"):
        if (f["sistema"], f["modo"], f["escenario"]) == (sistema, "recuperador", escenario):
            return f["latencia_ms"]["p50"]
    return None


def curva_pr(sistema, escenario, tarea="implicita"):
    m = leer(f"metricas_{sistema}_recuperador_{escenario}.json")
    assert m["n_consultas_sin_desarrollo"] == 1800
    return m["curva_pr_sin_desarrollo"][tarea]


def niveles():
    """[(nivel, tope_mib, modelo)] en orden de tope, de presupuesto_niveles.json (T2.6)."""
    p = leer("presupuesto_niveles.json")["niveles"]
    return sorted(((n, v["tope_mib"], v["modelo"]) for n, v in p.items()), key=lambda x: x[1])


def nivel_de(sistema):
    """Nivel de presupuesto del modelo del sistema (los niveles de T2.6 eligen un modelo cada uno)."""
    if sistema not in MODELO:
        return None
    return next(n for n, _, m in niveles() if m == MODELO[sistema])
