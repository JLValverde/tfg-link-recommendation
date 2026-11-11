"""Recuperador y modo recuperador sobre la muestra principal (T1.7, T1.8).

Con el modelo elegido en T1.6 (experimentos/resultados/eleccion_embeddings.json),
cada consulta se recupera una vez (top-100) y de ahí salen:
- las listas del modo recuperador (K = 50), que lee el arnés:
  datos/procesados/candidatos_recuperador_{escenario}.jsonl;
- el recall@K del recuperador (K = 10, 20, 50, 100) por consulta, en las tres tareas:
  experimentos/resultados/recall_recuperador_por_consulta.jsonl, y su agregado
  en experimentos/resultados/recall_recuperador.json (sobre las 1800 consultas que
  no son de desarrollo; con --solo-agregado se rehace desde el recall por consulta).

La generación de las listas vive aquí y no en arnes/ para que arnes/ no importe
nada de pipeline/.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/recuperador.py [--dir-listas D] [--dir-resultados D]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from arnes.candidatos import (DIR_PROCESADOS, K_RECUPERADOR, cargar_consultas,  # noqa: E402
                              escribir_candidatos_recuperador)
from arnes.evaluar import ic_bootstrap  # noqa: E402
from arnes.vista import Vista  # noqa: E402
from pipeline import recuperacion  # noqa: E402
from pipeline.baselines import modelo_elegido  # noqa: E402

DIR_RESULTADOS = RAIZ / "experimentos" / "resultados"
ESCENARIOS = ("caliente", "frio")
KS = (10, 20, 50, 100)
TAREAS = ("completa", "implicita", "implicita_solo_ancla")


def relevantes_por_tarea(c):
    return {"completa": set(c["referencia"]), "implicita": set(c["referencia_implicita"]),
            "implicita_solo_ancla": set(c["referencia"]) - set(c["explicitos_solo_ancla"])}


def recuperar(ind, consultas, k=max(KS)):
    """{idx: [(doc, puntuación)]} con los k primeros y la latencia por consulta (ms)."""
    tops, latencias = {}, []
    for c in consultas:
        t0 = time.perf_counter()
        tops[c["idx"]] = ind.recuperar(Vista(c), c["idx"], k)
        latencias.append((time.perf_counter() - t0) * 1000)
    return tops, latencias


def recall_por_consulta(consultas, tops):
    filas = []
    for c in consultas:
        top = [d for d, _ in tops[c["idx"]]]
        filas.append({"idx": c["idx"], "es_desarrollo": c["es_desarrollo"],
                      "recall": {t: {str(k): len(rel & set(top[:k])) / len(rel) for k in KS}
                                 for t, rel in relevantes_por_tarea(c).items()}})
    return filas


def agregar_recall(filas):
    """Media e IC del recall sobre las consultas que no son de desarrollo (decisión del 21 oct)."""
    filas = [f for f in filas if not f["es_desarrollo"]]
    return {t: {str(k): ic_bootstrap([f["recall"][t][str(k)] for f in filas]) for k in KS} for t in TAREAS}


def solo_agregado(dir_res):
    """Rehace recall_recuperador.json desde recall_recuperador_por_consulta.jsonl, sin recuperar."""
    filas = [json.loads(l) for l in (dir_res / "recall_recuperador_por_consulta.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    ruta = dir_res / "recall_recuperador.json"
    salida = json.loads(ruta.read_text(encoding="utf-8"))  # conserva modelo y latencias
    salida["muestra"] = "principal_sin_desarrollo"
    salida["n_consultas"] = {}
    for esc in ESCENARIOS:
        de_esc = [f for f in filas if f["escenario"] == esc]
        salida["recall"][esc] = agregar_recall(de_esc)
        salida["n_consultas"][esc] = sum(not f["es_desarrollo"] for f in de_esc)
    ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-listas", default=str(DIR_PROCESADOS))
    ap.add_argument("--dir-resultados", default=str(DIR_RESULTADOS))
    ap.add_argument("--solo-agregado", action="store_true",
                    help="rehacer recall_recuperador.json desde el recall por consulta guardado, sin recuperar")
    args = ap.parse_args()
    dir_listas, dir_res = Path(args.dir_listas), Path(args.dir_resultados)
    if args.solo_agregado:
        solo_agregado(dir_res)
        return
    dir_listas.mkdir(parents=True, exist_ok=True)
    dir_res.mkdir(parents=True, exist_ok=True)

    modelo = modelo_elegido()
    ind = recuperacion.indice(modelo)
    salida = {"modelo": modelo, "muestra": "principal_sin_desarrollo", "recall": {},
              "latencia_recuperar_ms": {}, "n_consultas": {}}
    por_consulta = []
    for esc in ESCENARIOS:
        consultas = cargar_consultas(esc)
        t0 = time.perf_counter()
        tops, lat = recuperar(ind, consultas)
        listas = {q: top[:K_RECUPERADOR] for q, top in tops.items()}
        print(f"Escrito {escribir_candidatos_recuperador(listas, esc, dir_listas)} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        filas = recall_por_consulta(consultas, tops)
        por_consulta += [{"escenario": esc, **f} for f in filas]
        salida["recall"][esc] = agregar_recall(filas)
        salida["n_consultas"][esc] = sum(not f["es_desarrollo"] for f in filas)
        salida["latencia_recuperar_ms"][esc] = {"p50": float(np.percentile(lat, 50)),
                                                 "p95": float(np.percentile(lat, 95))}
    with (dir_res / "recall_recuperador_por_consulta.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for fila in por_consulta:
            f.write(json.dumps(fila) + "\n")
    (dir_res / "recall_recuperador.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
