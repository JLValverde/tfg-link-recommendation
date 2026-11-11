"""Fusión exploratoria de C1 y Adamic-Adar (B) (T2.8b; ANÁLISIS A POSTERIORI, decisión de Jose).

No es un sistema del protocolo: se añade después de ver los resultados de C1 y de los
baselines. En las tablas va en una sección aparte, "Análisis exploratorio (a posteriori)",
nunca con los sistemas del protocolo, y no entra en las figuras.

Sistema rrf_c1_<modelo>_aa_B (modelo 3b, 7b o 14b): RRF con k = 60 de C1 con ese modelo y
Adamic-Adar (B), como la fusión de BM25 y Adamic-Adar (pipeline/fusion.py: rangos del
arnés, desempate fijo, lista entera). No llama al modelo: lee las puntuaciones crudas
versionadas de C1, crudas/<id_ejecucion>/puntuaciones_<c1>_recuperador_caliente.jsonl.gz,
de la ejecución que da c1_resumen.json. Solo caliente (en frío Adamic-Adar no aplica) y
modo recuperador; se puntúan las 2000 consultas y se reportan las 1800.

Sin con_coste: no llama a ningún modelo ni mide nada, así que no hay coste propio ni
fichero en ejecuciones/. Su coste es el de C1 más el de Adamic-Adar. En su lugar, el
resumen guarda de dónde salen las puntuaciones de C1 (id_ejecucion) y el estado de git.

Escribe en experimentos/resultados/:
- puntuaciones_rrf_c1_<m>_aa_B_recuperador_caliente.jsonl y metricas_... (no versionados);
- fusion_exploratoria_resumen.json: una fila por (sistema, tarea), con la fuente de C1,
  los None de cada componente y el estado de git; se actualiza por modelo.

Uso (desde la raíz del repo; no llama a Ollama ni usa la GPU):
    .venv/Scripts/python.exe experimentos/evaluar_fusion_exploratoria.py --modelo 3b 14b
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.coste import estado_git  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, evaluar, nombre_ejecucion  # noqa: E402
from evaluar_controles import filas_resumen  # noqa: E402
from pipeline import fusion  # noqa: E402

MODO, ESCENARIO = "recuperador", "caliente"
C1_DE = {"3b": "c1", "7b": "c1_7b", "14b": "c1_14b"}
RUTA_RESUMEN = DIR_RESULTADOS / "fusion_exploratoria_resumen.json"


def sistema(modelo):
    return f"rrf_c1_{modelo}_aa_B"


def fuente_c1(c1):
    """id_ejecucion de C1 en caliente, modo recuperador, según c1_resumen.json."""
    filas = json.loads((DIR_RESULTADOS / "c1_resumen.json").read_text(encoding="utf-8"))
    ids = {r["id_ejecucion"] for r in filas if (r["sistema"], r["modo"], r["escenario"]) == (c1, MODO, ESCENARIO)}
    if len(ids) != 1:
        sys.exit(f"c1_resumen.json no da una sola ejecución de {c1} en {ESCENARIO}: {ids}")
    return ids.pop()


def leer_crudas(c1, id_ejecucion):
    ruta = DIR_RESULTADOS / "crudas" / id_ejecucion / f"puntuaciones_{nombre_ejecucion(c1, MODO, ESCENARIO)}.jsonl.gz"
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        lineas = [json.loads(l) for l in f]
    return {d["idx"]: {x: p for x, p in d["puntuaciones"]} for d in lineas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", nargs="+", choices=sorted(C1_DE), required=True)
    args = ap.parse_args()
    resumen = json.loads(RUTA_RESUMEN.read_text(encoding="utf-8")) if RUTA_RESUMEN.exists() else []
    git = estado_git()
    for modelo in args.modelo:
        c1, s = C1_DE[modelo], sistema(modelo)
        id_c1 = fuente_c1(c1)
        puntuaciones_c1 = leer_crudas(c1, id_c1)
        scorer = fusion.rrf_c1_aa_B(puntuaciones_c1)
        res = evaluar(s, MODO, ESCENARIO, scorer=scorer)
        assert res["n_consultas"] == len(puntuaciones_c1) == 2000 and res["n_consultas_sin_desarrollo"] == 1800
        nones = dict(scorer.none_por_componente)
        extra = {"modelo_c1": modelo, "fuente_c1": id_c1, "k_rrf": fusion.K_RRF,
                 **{f"none_{c}": n for c, n in nones.items()}, "git": git, "a_posteriori": True}
        resumen = [r for r in resumen if r["sistema"] != s] + filas_resumen(res, lambda t: extra)
        print(f"{s}: hecho (C1 de {id_c1}); None por componente: {nones}", flush=True)
    RUTA_RESUMEN.write_text(json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Escrito {RUTA_RESUMEN.name}")


if __name__ == "__main__":
    main()
