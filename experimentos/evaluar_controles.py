"""Evalúa los tres scorers de control (T1.1-T1.5; modo recuperador desde T1.8).

Para cada scorer (aleatorio, siempre_none, oraculo), escenario (caliente, frío) y
modo (aleatorio con las semillas de candidatos 1, 2, 3, y recuperador), sobre las
2000 consultas de la muestra principal. Las tablas del resumen se calculan sobre
las 1800 que no son de desarrollo (decisión del 21 oct).
Cada scorer se envuelve con el decorador de coste (el arnés no lo sabe).

Escribe en experimentos/resultados/:
- puntuaciones_*.jsonl y metricas_*.json de cada ejecución;
- coste_controles.jsonl (se reescribe en cada ejecución del script) y
  coste_controles_agregado.json;
- controles_resumen.json: media e IC 95 % de cada métrica sobre las 1800, más el
  valor esperado del azar |R_lista|/N_lista sobre esas mismas consultas.

Con --solo-resumen no se puntúa ni se registra coste: se recalculan las métricas y
el resumen desde las puntuaciones ya guardadas.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/evaluar_controles.py [--solo-resumen]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from arnes.agregar import agregar, leer  # noqa: E402
from arnes.candidatos import K_RECUPERADOR, SEMILLAS, cargar_candidatos, cargar_consultas  # noqa: E402
from arnes.coste import con_coste, config_ejecucion  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, evaluar, recalcular_metricas  # noqa: E402
from arnes.scorer import SCORERS  # noqa: E402

CONTROLES = ("aleatorio", "siempre_none", "oraculo")
ESCENARIOS = ("caliente", "frio")
EJECUCIONES = [("aleatorio", s) for s in SEMILLAS] + [("recuperador", None)]
RUTA_COSTE = DIR_RESULTADOS / "coste_controles.jsonl"


def esperado_azar(escenario, modo, semilla):
    """Media de |R_lista| / N_lista en cada tarea sobre las consultas que no son de desarrollo."""
    consultas = [c for c in cargar_consultas(escenario) if not c["es_desarrollo"]]
    cands = cargar_candidatos(modo, escenario, semilla)
    v = {"completa": [], "implicita": [], "implicita_solo_ancla": []}
    for c in consultas:
        lista = set(cands[c["idx"]]) - set(c["neutros_plantilla"])
        v["completa"].append(len(lista & set(c["referencia"])) / len(lista))
        impl = lista - set(c["explicitos"])
        v["implicita"].append(len(impl & set(c["referencia_implicita"])) / len(impl))
        ancla = lista - set(c["explicitos_solo_ancla"])
        v["implicita_solo_ancla"].append(
            len(ancla & (set(c["referencia"]) - set(c["explicitos_solo_ancla"]))) / len(ancla))
    return {t: float(np.mean(x)) for t, x in v.items()}


def filas_resumen(res, extra=None):
    """Una fila por tarea, con el agregado de reporte (consultas que no son de desarrollo)."""
    filas = []
    for tarea, ms in res["agregado_sin_desarrollo"].items():
        filas.append({"sistema": res["sistema"], "modo": res["modo"], "escenario": res["escenario"],
                      "semilla": res["semilla"], "tarea": tarea,
                      "muestra": "principal_sin_desarrollo",
                      "n_consultas": res["n_consultas_sin_desarrollo"],
                      "tasa_none": res["tasa_none"], "malformadas": res["malformadas"],
                      **(extra(tarea) if extra else {}),
                      **{m: {"media": v["media"], "ic95": v["ic95"]} for m, v in ms.items()}})
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-resumen", action="store_true",
                    help="recalcular métricas y resumen desde las puntuaciones guardadas, sin puntuar")
    args = ap.parse_args()
    if not args.solo_resumen:
        RUTA_COSTE.unlink(missing_ok=True)
    resumen = []
    for escenario in ESCENARIOS:
        for modo, s in EJECUCIONES:
            azar = esperado_azar(escenario, modo, s)
            for sistema in CONTROLES:
                if args.solo_resumen:
                    res = recalcular_metricas(sistema, modo, escenario, s)
                else:
                    config = config_ejecucion(K=K_RECUPERADOR if modo == "recuperador" else None)
                    scorer = con_coste(RUTA_COSTE, sistema=sistema, modo=modo, semilla=s, escenario=escenario,
                                       config=config)(SCORERS[sistema])
                    res = evaluar(sistema, modo, escenario, s, scorer=scorer)
                resumen += filas_resumen(res, lambda t: {"esperado_azar": azar[t]})
                print(f"{escenario} {modo} s{s} {sistema}: hecho", flush=True)
    (DIR_RESULTADOS / "controles_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not args.solo_resumen:
        tabla = agregar(leer([RUTA_COSTE]))
        (DIR_RESULTADOS / "coste_controles_agregado.json").write_text(
            json.dumps(tabla, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Escrito controles_resumen.json" + ("" if args.solo_resumen else " y coste_controles_agregado.json"))


if __name__ == "__main__":
    main()
