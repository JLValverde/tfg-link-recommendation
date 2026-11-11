"""Baselines de texto C0 y BM25 (T1.9), con coste registrado.

Para cada baseline, escenario y modo (aleatorio con semillas 1, 2, 3, y
recuperador), sobre las 2000 consultas de la muestra principal. En modo
recuperador reordenan las 50 candidatas del recuperador. Cada scorer se envuelve
con el decorador de coste. Antes se mide el coste de construir cada índice. Las
tablas del resumen se calculan sobre las 1800 consultas que no son de desarrollo
(decisión del 21 oct).

Escribe en experimentos/resultados/:
- puntuaciones_{c0,bm25}_*.jsonl y metricas_{c0,bm25}_*.json;
- coste_baselines.jsonl (se reescribe), coste_baselines_agregado.json;
- indexado_baselines.json (tiempo y VRAM de construir cada índice);
- baselines_resumen.json (media e IC 95 % de cada métrica sobre las 1800).

Con --solo-resumen no se puntúa, no se registra coste y no se mide el indexado:
se recalculan las métricas y el resumen desde las puntuaciones ya guardadas.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/evaluar_baselines.py [--solo-resumen]
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
from arnes.medir_vram import medir_en_proceso  # noqa: E402
from arnes.scorer import SCORERS  # noqa: E402
from evaluar_controles import filas_resumen  # noqa: E402
from pipeline import baselines, bm25, recuperacion  # noqa: E402,F401  (registra c0 y bm25)

SISTEMAS = ("c0", "bm25")
ESCENARIOS = ("caliente", "frio")
EJECUCIONES = [("aleatorio", s) for s in SEMILLAS] + [("recuperador", None)]
RUTA_COSTE = DIR_RESULTADOS / "coste_baselines.jsonl"


def medir_indexado():
    """Coste de construir cada índice.

    C0: se toma la medida de la elección del modelo (T1.6), mismo modelo y mismo
    procedimiento; no se vuelve a indexar para no reescribir los embeddings de los
    que salen las listas del modo recuperador. BM25: se construye aquí, midiendo.
    """
    modelo = baselines.modelo_elegido()
    eleccion = json.loads(baselines.RUTA_ELECCION.read_text(encoding="utf-8"))
    m_c0 = next(m for m in eleccion["modelos"] if m["modelo"] == modelo)["indexado"]
    _, m_bm25 = medir_en_proceso(bm25.indice)
    return {"c0": {"modelo": modelo, "fuente": "eleccion_embeddings.json", **m_c0},
            "bm25": {"fuente": "medido aquí", **m_bm25}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-resumen", action="store_true",
                    help="recalcular métricas y resumen desde las puntuaciones guardadas, sin puntuar")
    args = ap.parse_args()
    if not args.solo_resumen:
        indexado = medir_indexado()
        (DIR_RESULTADOS / "indexado_baselines.json").write_text(
            json.dumps(indexado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        RUTA_COSTE.unlink(missing_ok=True)
    resumen = []
    for escenario in ESCENARIOS:
        for modo, s in EJECUCIONES:
            for sistema in SISTEMAS:
                if args.solo_resumen:
                    res = recalcular_metricas(sistema, modo, escenario, s)
                else:
                    config = config_ejecucion(
                        K=K_RECUPERADOR if modo == "recuperador" else None,
                        **({"modelo_embeddings": baselines.modelo_elegido()} if sistema == "c0" else {}))
                    scorer = con_coste(RUTA_COSTE, sistema=sistema, modo=modo, semilla=s, escenario=escenario,
                                       config=config)(SCORERS[sistema])
                    res = evaluar(sistema, modo, escenario, s, scorer=scorer)
                resumen += filas_resumen(res)
                print(f"{escenario} {modo} s{s} {sistema}: hecho", flush=True)
    (DIR_RESULTADOS / "baselines_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not args.solo_resumen:
        (DIR_RESULTADOS / "coste_baselines_agregado.json").write_text(
            json.dumps(agregar(leer([RUTA_COSTE])), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Escrito baselines_resumen.json")


if __name__ == "__main__":
    main()
