"""Baseline de fusión RRF de BM25 y Adamic-Adar (B) (T2.1), con coste registrado.

Solo en CALIENTE: en frío Adamic-Adar no aplica y la fusión tampoco (se reporta
como "no aplica"). Los dos modos de candidatos: aleatorio (semillas 1, 2, 3) y
recuperador (las mismas 50 candidatas que C0, BM25 y las heurísticas), sobre las
2000 consultas de la muestra principal; el resumen se calcula sobre las 1800 que
no son de desarrollo. k = 60 fijo: no hay nada que ajustar, así que no se usa la
muestra de desarrollo.

Escribe en experimentos/resultados/:
- puntuaciones_rrf_bm25_aa_B_*.jsonl y metricas_rrf_bm25_aa_B_*.json (no versionados);
- coste_fusion.jsonl (se reescribe) y coste_fusion_agregado.json;
- fusion_resumen.json, con el recuento de None de cada componente por ejecución
  (none_bm25, none_adamic_adar_B; null con --solo-resumen, que no puntúa).

Con --solo-resumen se recalculan métricas y resumen desde las puntuaciones guardadas.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/evaluar_fusion.py [--solo-resumen]
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
from pipeline import fusion  # noqa: E402  (registra rrf_bm25_aa_B)

SISTEMA = "rrf_bm25_aa_B"
ESCENARIOS = ("caliente",)  # en frío no aplica
EJECUCIONES = [("aleatorio", s) for s in SEMILLAS] + [("recuperador", None)]
RUTA_COSTE = DIR_RESULTADOS / "coste_fusion.jsonl"


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
            nones = {c: None for c in fusion.COMPONENTES}
            if args.solo_resumen:
                res = recalcular_metricas(SISTEMA, modo, escenario, s)
            else:
                for c in fusion.COMPONENTES:
                    fusion.NONE_POR_COMPONENTE[c] = 0
                config = config_ejecucion(K=K_RECUPERADOR if modo == "recuperador" else None,
                                          k_rrf=fusion.K_RRF, componentes=list(fusion.COMPONENTES))
                scorer = con_coste(RUTA_COSTE, sistema=SISTEMA, modo=modo, semilla=s,
                                   escenario=escenario, config=config)(SCORERS[SISTEMA])
                res = evaluar(SISTEMA, modo, escenario, s, scorer=scorer)
                nones = dict(fusion.NONE_POR_COMPONENTE)
            resumen += filas_resumen(res, lambda t: {f"none_{c}": n for c, n in nones.items()})
            print(f"{escenario} {modo} s{s} {SISTEMA}: hecho; None por componente: {nones}", flush=True)
    (DIR_RESULTADOS / "fusion_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not args.solo_resumen:
        (DIR_RESULTADOS / "coste_fusion_agregado.json").write_text(
            json.dumps(agregar(leer([RUTA_COSTE])), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Escrito fusion_resumen.json")


if __name__ == "__main__":
    main()
