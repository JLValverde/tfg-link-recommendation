"""Baselines estructurales: vecinos comunes y Adamic-Adar (T1.10), con coste registrado.

Sobre la vista no dirigida del grafo visible de cada consulta, en versión B (la del
experimento) y A (comparación). Solo en CALIENTE: en frío no aplican y no se
ejecutan (se reportan como "no aplica"). Los dos modos de candidatos: aleatorio
(semillas 1, 2, 3) y recuperador (las mismas 50 candidatas que C0 y BM25), sobre
las 2000 consultas de la muestra principal; el resumen se calcula sobre las 1800
que no son de desarrollo.

Escribe en experimentos/resultados/:
- puntuaciones_*.jsonl y metricas_*.json de cada ejecución (no versionados);
- coste_estructurales.jsonl (se reescribe) y coste_estructurales_agregado.json;
- estructurales_resumen.json.

Con --solo-resumen se recalculan métricas y resumen desde las puntuaciones guardadas.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/evaluar_estructurales.py [--solo-resumen]
"""

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.agregar import agregar, leer  # noqa: E402
from arnes.candidatos import K_RECUPERADOR, SEMILLAS  # noqa: E402
from arnes.coste import con_coste, config_ejecucion  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, evaluar, recalcular_metricas  # noqa: E402
from arnes.scorer import SCORERS  # noqa: E402
from evaluar_controles import filas_resumen  # noqa: E402
from pipeline import estructurales  # noqa: E402,F401  (registra los cuatro scorers)

SISTEMAS = ("vecinos_comunes_B", "adamic_adar_B", "vecinos_comunes_A", "adamic_adar_A")
ESCENARIOS = ("caliente",)  # en frío no aplican
EJECUCIONES = [("aleatorio", s) for s in SEMILLAS] + [("recuperador", None)]
RUTA_COSTE = DIR_RESULTADOS / "coste_estructurales.jsonl"


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
            for sistema in SISTEMAS:
                if args.solo_resumen:
                    res = recalcular_metricas(sistema, modo, escenario, s)
                else:
                    config = config_ejecucion(K=K_RECUPERADOR if modo == "recuperador" else None)
                    scorer = con_coste(RUTA_COSTE, sistema=sistema, modo=modo, semilla=s, escenario=escenario,
                                       config=config)(SCORERS[sistema])
                    res = evaluar(sistema, modo, escenario, s, scorer=scorer)
                resumen += filas_resumen(res)
                print(f"{escenario} {modo} s{s} {sistema}: hecho", flush=True)
    (DIR_RESULTADOS / "estructurales_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not args.solo_resumen:
        (DIR_RESULTADOS / "coste_estructurales_agregado.json").write_text(
            json.dumps(agregar(leer([RUTA_COSTE])), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Escrito estructurales_resumen.json")


if __name__ == "__main__":
    main()
