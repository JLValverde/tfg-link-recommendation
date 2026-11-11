"""Orquestador del arnés: consultas -> candidatos -> scorer -> rangos -> métricas.

evaluar(sistema, modo, escenario, semilla) hace dos pasos:
1. Puntúa cada consulta y guarda las puntuaciones crudas en
   experimentos/resultados/puntuaciones_{sistema}_{modo}_{escenario}[_s{s}][_dev].jsonl.
   Antes de puntuar quita los neutros_plantilla de la lista de candidatos.
2. Lee ese fichero y calcula las métricas por consulta y tarea, el agregado (media
   con IC 95 % por bootstrap) y la curva precisión-recall. Las guarda en
   metricas_{...}.json.

Por defecto puntúa la muestra principal (2000 consultas); con solo_desarrollo,
solo las 200 de desarrollo. Las tablas se reportan con `agregado_sin_desarrollo`:
las 1800 consultas de la principal que no son de desarrollo (decisión del 21 oct).
`agregado` sigue cubriendo todas las consultas puntuadas. Nunca descarta una
consulta: si un scorer no puede puntuar, cuenta como None (peor puntuación) y se
reporta la tasa de None y el recuento de malformadas por tipo.

recalcular_metricas(...) rehace metricas_*.json desde las puntuaciones guardadas,
sin volver a puntuar.

metricas_*.json lleva el `id_ejecucion` del scorer si está envuelto con con_coste
(None si no); recalcular_metricas conserva el que ya tenía el fichero, salvo que
se le pase otro.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe -m arnes.evaluar --sistema aleatorio --modo aleatorio --escenario frio --semilla 1 [--desarrollo]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from arnes.candidatos import cargar_candidatos, cargar_consultas
from arnes.metricas import METRICAS_POR_DEFECTO, curva_pr, pares_puntuacion_acierto
from arnes.rangos import TAREAS, rankings_por_tarea, valor
from arnes.scorer import SCORERS
from arnes.vista import Vista

RAIZ = Path(__file__).resolve().parent.parent
DIR_RESULTADOS = RAIZ / "experimentos" / "resultados"
DIR_PROCESADOS = RAIZ / "datos" / "procesados"
N_BOOTSTRAP = 1000
SEMILLA_BOOTSTRAP = 0


def nombre_ejecucion(sistema, modo, escenario, semilla=None, solo_desarrollo=False):
    return (f"{sistema}_{modo}_{escenario}" + (f"_s{semilla}" if semilla is not None else "")
            + ("_dev" if solo_desarrollo else ""))


# ---------------------------------------------------------------- estadística

def ic_bootstrap(valores, n=N_BOOTSTRAP, semilla=SEMILLA_BOOTSTRAP):
    """Media e IC 95 % por bootstrap percentil. Ignora los None y lo reporta."""
    v = np.asarray([x for x in valores if x is not None], dtype=float)
    if not len(v):
        return {"media": None, "ic95": [None, None], "n": 0, "n_none": len(valores)}
    rng = np.random.default_rng(semilla)
    medias = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return {"media": float(v.mean()),
            "ic95": [float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))],
            "n": int(len(v)), "n_none": len(valores) - int(len(v))}


def diferencia_pareada(res_a, res_b, tarea, metrica, n=N_BOOTSTRAP, semilla=SEMILLA_BOOTSTRAP,
                       indices=None):
    """Diferencia media A - B por consulta, con IC 95 % por bootstrap pareado.

    indices: si se da, solo esas consultas (p. ej., las 1800 que no son de desarrollo).
    """
    a = dict(zip(res_a["idx"], res_a["por_consulta"][tarea][metrica]))
    b = dict(zip(res_b["idx"], res_b["por_consulta"][tarea][metrica]))
    comunes = sorted(set(a) & set(b) & (set(indices) if indices is not None else set(a)))
    d = np.asarray([a[i] - b[i] for i in comunes if a[i] is not None and b[i] is not None], dtype=float)
    rng = np.random.default_rng(semilla)
    medias = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return {"diferencia": float(d.mean()),
            "ic95": [float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))],
            "n": int(len(d))}


# ---------------------------------------------------------------- pasos

TIPOS_MALFORMADAS = ("none", "ausentes", "nan")


def contar_malformadas(p, lista):
    """Por tipo: None devueltos, candidatos ausentes de la salida y NaN.

    Los tres acaban como None (peor puntuación) en las puntuaciones crudas; aquí
    se cuentan antes de convertirlos para no perder el tipo.
    """
    return {
        "none": sum(1 for x in lista if x in p and p[x] is None),
        "ausentes": sum(1 for x in lista if x not in p),
        "nan": sum(1 for x in lista if x in p and p[x] is not None and math.isnan(float(p[x]))),
    }


def puntuar(scorer, consultas, candidatos, ruta):
    """Paso 1: puntuaciones crudas, una línea por consulta, con el recuento de malformadas."""
    with Path(ruta).open("w", encoding="utf-8", newline="\n") as f:
        for c in consultas:
            q = c["idx"]
            neutros = set(c["neutros_plantilla"])
            lista = [x for x in candidatos[q] if x not in neutros]
            publica = {"idx": q, "escenario": c["escenario"]}
            p = scorer(publica, list(lista), Vista(c))
            extra = set(p) - set(lista)
            if extra:
                raise ValueError(f"El scorer puntúa candidatos que no se le dieron (consulta {q}): {sorted(extra)[:5]}")
            puntuaciones = [[x, valor(p.get(x))] for x in sorted(lista)]
            f.write(json.dumps({"idx": q, "puntuaciones": puntuaciones,
                                "malformadas": contar_malformadas(p, lista)}) + "\n")


def agregar_malformadas(por_consulta, n_puntuaciones):
    """Totales y tasas por tipo. None en todo si algún fichero es anterior al recuento."""
    if not por_consulta or any(m is None for m in por_consulta):
        return {**{t: None for t in TIPOS_MALFORMADAS},
                **{f"tasa_{t}": None for t in TIPOS_MALFORMADAS}, "consultas_con_alguna": None}
    tot = {t: sum(m[t] for m in por_consulta) for t in TIPOS_MALFORMADAS}
    return {**tot, **{f"tasa_{t}": tot[t] / n_puntuaciones if n_puntuaciones else None
                      for t in TIPOS_MALFORMADAS},
            "consultas_con_alguna": sum(1 for m in por_consulta if any(m.values()))}


def calcular_metricas(ruta_puntuaciones, consultas, metricas):
    """Paso 2: métricas desde el fichero de puntuaciones crudas."""
    por_idx = {c["idx"]: c for c in consultas}
    res = {"idx": [], "por_consulta": {t: {m.nombre: [] for m in metricas} for t in TAREAS}}
    pares = {t: [] for t in TAREAS}
    pares_sin_dev = {t: [] for t in TAREAS}
    n_punt = n_none = 0
    malformadas = []
    for linea in Path(ruta_puntuaciones).read_text(encoding="utf-8").splitlines():
        d = json.loads(linea)
        c = por_idx[d["idx"]]
        punt = {x: p for x, p in d["puntuaciones"]}
        n_punt += len(punt)
        n_none += sum(p is None for p in punt.values())
        malformadas.append(d.get("malformadas"))  # None en ficheros anteriores al recuento
        res["idx"].append(d["idx"])
        for tarea, (ranking, relevantes) in rankings_por_tarea(c, punt).items():
            for m in metricas:
                res["por_consulta"][tarea][m.nombre].append(m(ranking, relevantes))
            p = pares_puntuacion_acierto(ranking, relevantes, punt)
            pares[tarea] += p
            if not c["es_desarrollo"]:
                pares_sin_dev[tarea] += p
    res["n_consultas"] = len(res["idx"])
    res["tasa_none"] = n_none / n_punt if n_punt else None
    res["n_puntuaciones"] = n_punt
    res["malformadas"] = agregar_malformadas(malformadas, n_punt)
    # Agregado sobre todas las consultas puntuadas (con --desarrollo, las 200)
    res["agregado"] = {t: {nombre: ic_bootstrap(v) for nombre, v in ms.items()}
                       for t, ms in res["por_consulta"].items()}
    res["curva_pr"] = {t: curva_pr(pares[t]) for t in TAREAS}
    # Agregado de reporte: solo las consultas que no son de desarrollo (las 1800 de la
    # muestra principal; decisión del 21 oct). Vacío si solo se puntuó desarrollo.
    sin_dev = [k for k, q in enumerate(res["idx"]) if not por_idx[q]["es_desarrollo"]]
    res["n_consultas_sin_desarrollo"] = len(sin_dev)
    res["agregado_sin_desarrollo"] = (
        {t: {nombre: ic_bootstrap([v[k] for k in sin_dev]) for nombre, v in ms.items()}
         for t, ms in res["por_consulta"].items()} if sin_dev else None)
    res["curva_pr_sin_desarrollo"] = {t: curva_pr(pares_sin_dev[t]) for t in TAREAS} if sin_dev else None
    return res


def evaluar(sistema, modo, escenario, semilla=None, solo_desarrollo=False, scorer=None,
            metricas=METRICAS_POR_DEFECTO, dir_resultados=DIR_RESULTADOS, candidatos=None):
    """Evalúa un sistema y devuelve el diccionario de resultados (también lo guarda en disco).

    scorer: por defecto, el registrado con el nombre `sistema`.
    candidatos: {idx: lista}; por defecto se leen del fichero del modo.
    """
    scorer = scorer if scorer is not None else SCORERS[sistema]
    consultas = cargar_consultas(escenario)
    if solo_desarrollo:
        consultas = [c for c in consultas if c["es_desarrollo"]]
    if candidatos is None:
        if modo == "aleatorio":
            if semilla is None:
                raise ValueError("El modo aleatorio necesita una semilla")
            candidatos = cargar_candidatos("aleatorio", escenario, semilla)
        elif modo == "recuperador":
            candidatos = cargar_candidatos("recuperador", escenario)
        else:
            raise ValueError(f"Modo desconocido: {modo!r}")
    dir_resultados = Path(dir_resultados)
    dir_resultados.mkdir(parents=True, exist_ok=True)
    nombre = nombre_ejecucion(sistema, modo, escenario, semilla, solo_desarrollo)
    ruta_p = dir_resultados / f"puntuaciones_{nombre}.jsonl"

    puntuar(scorer, consultas, candidatos, ruta_p)
    return _guardar_metricas(sistema, modo, escenario, semilla, solo_desarrollo, metricas,
                             dir_resultados, ruta_p, consultas,
                             getattr(scorer, "id_ejecucion", None))


def _guardar_metricas(sistema, modo, escenario, semilla, solo_desarrollo, metricas,
                      dir_resultados, ruta_p, consultas, id_ejecucion):
    res = {"sistema": sistema, "modo": modo, "escenario": escenario, "semilla": semilla,
           "solo_desarrollo": solo_desarrollo, "id_ejecucion": id_ejecucion,
           "metricas": [m.nombre for m in metricas],
           **calcular_metricas(ruta_p, consultas, metricas)}
    nombre = nombre_ejecucion(sistema, modo, escenario, semilla, solo_desarrollo)
    (Path(dir_resultados) / f"metricas_{nombre}.json").write_text(
        json.dumps(res, ensure_ascii=False) + "\n", encoding="utf-8")
    return res


def recalcular_metricas(sistema, modo, escenario, semilla=None, solo_desarrollo=False,
                        metricas=METRICAS_POR_DEFECTO, dir_resultados=DIR_RESULTADOS, id_ejecucion=None):
    """Recalcula y reescribe metricas_*.json desde las puntuaciones crudas ya guardadas, sin puntuar.

    id_ejecucion: si no se da, se conserva el del fichero de métricas anterior.
    """
    consultas = cargar_consultas(escenario)
    if solo_desarrollo:
        consultas = [c for c in consultas if c["es_desarrollo"]]
    nombre = nombre_ejecucion(sistema, modo, escenario, semilla, solo_desarrollo)
    ruta_p = Path(dir_resultados) / f"puntuaciones_{nombre}.jsonl"
    if not ruta_p.exists():
        raise FileNotFoundError(f"No hay puntuaciones guardadas en {ruta_p}")
    ruta_m = Path(dir_resultados) / f"metricas_{nombre}.json"
    if id_ejecucion is None and ruta_m.exists():
        id_ejecucion = json.loads(ruta_m.read_text(encoding="utf-8")).get("id_ejecucion")
    return _guardar_metricas(sistema, modo, escenario, semilla, solo_desarrollo, metricas,
                             dir_resultados, ruta_p, consultas, id_ejecucion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sistema", required=True)
    ap.add_argument("--modo", default="aleatorio")
    ap.add_argument("--escenario", required=True, choices=["caliente", "frio"])
    ap.add_argument("--semilla", type=int, default=None)
    ap.add_argument("--desarrollo", action="store_true")
    args = ap.parse_args()
    res = evaluar(args.sistema, args.modo, args.escenario, args.semilla, args.desarrollo)
    for tarea, ms in res["agregado"].items():
        print(tarea, {k: round(v["media"], 4) if v["media"] is not None else None for k, v in ms.items()})
    print(f"Tasa de None: {res['tasa_none']:.4f}")


if __name__ == "__main__":
    main()
