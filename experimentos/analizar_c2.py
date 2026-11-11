"""Análisis de C2: saturación, casos, empates y coste; en desarrollo, también métricas (T2.4, T2.5, T2.7).

Como experimentos/analizar_c1.py, con dos diferencias:
- En desarrollo (--variante N), frío es el de V1 para las tres variantes (el prompt de
  frío es el mismo), y se compara también con C1 congelado (V2 de C1, c1_v2 en
  desarrollo) y con la fusión RRF (caliente).
- En la principal (--sistema c2 | c2_7b | c2_14b), solo sobre las 1800 que no son de
  desarrollo, y el coste se agrega con el pico de VRAM: arnes.agregar falla si supera
  el tope del nivel de presupuesto de los registros (confirmación del ADR de los tres
  topes). La ejecución principal no carga el modelo de embeddings, y el ADR define el
  pico con los dos cargados, así que el pico que se comprueba es el mayor entre el de
  la propia ejecución (vram_{sistema}.json, o vram_{sistema}_{escenario}.json si se midió
  cada escenario aparte) y el de T2.6 para el mismo modelo con el de embeddings a la vez
  (vram_presupuesto_{tag}.json).
Los tokens con caché se estiman restando el prefijo común de cada consulta (todo lo que
va antes de "Article B:\\nTitle:", con su lista de vecinos), medido aparte con el modelo
de la ejecución.

Escribe {sistema}_analisis_dev.json (desarrollo) o {sistema}_analisis.json (principal).

Uso (desde la raíz del repo, con Ollama en marcha):
    .venv/Scripts/python.exe experimentos/analizar_c2.py --variante N
    .venv/Scripts/python.exe experimentos/analizar_c2.py --sistema c2 [--escenarios frio]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from analizar_c1 import CALENTAMIENTO, CIERRE_PLANTILLA, leer_jsonl, saturacion_y_empates  # noqa: E402
from arnes.agregar import agregar, comprobar_presupuesto, picos_vram  # noqa: E402
from arnes.candidatos import cargar_candidatos, cargar_consultas  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, ic_bootstrap, nombre_ejecucion  # noqa: E402
from arnes.vista import Vista  # noqa: E402
from evaluar_c2 import SISTEMA_MODELO  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402
from variantes_c2 import VARIANTES, sistema as sistema_variante  # noqa: E402

MODO = "recuperador"
ESCENARIOS = ("caliente", "frio")
METRICAS = ("mrr", "precision@1")
TAREA = "implicita"
COMPARADOS = {"caliente": ("aleatorio", "oraculo", "c0", "bm25", "adamic_adar_B", "rrf_bm25_aa_B"),
              "frio": ("aleatorio", "oraculo", "c0", "bm25")}
C1_DEV = "c1_v2"  # C1 congelado (V2 de T2.2b) en desarrollo


def metricas(sistema, escenario, dev):
    nombre = nombre_ejecucion(sistema, MODO, escenario, None, dev)
    return json.loads((DIR_RESULTADOS / f"metricas_{nombre}.json").read_text(encoding="utf-8"))


def sistema_dev(variante, escenario):
    return sistema_variante(variante if escenario == "caliente" else 1)


def tabla_metricas(variante):
    filas, por_sistema, idx = [], {}, None
    for esc in ESCENARIOS:
        c2 = metricas(sistema_dev(variante, esc), esc, True)
        idx = idx or c2["idx"]
        assert c2["idx"] == idx, "C2 debe tener las mismas consultas en los dos escenarios"
        sistemas = {f"c2_v{variante}": c2, "c1 (congelado)": metricas(C1_DEV, esc, True),
                    **{s: metricas(s, esc, False) for s in COMPARADOS[esc]}}
        for nombre, res in sistemas.items():
            v = dict(zip(res["idx"], res["por_consulta"][TAREA]["mrr"])), dict(
                zip(res["idx"], res["por_consulta"][TAREA]["precision@1"]))
            for m, valores in zip(METRICAS, v):
                lista = [valores[i] for i in idx]
                por_sistema.setdefault((nombre, m), {})[esc] = lista
                filas.append({"sistema": nombre, "escenario": esc, "metrica": m, **ic_bootstrap(lista)})
    for (nombre, m), v in por_sistema.items():
        if set(v) == set(ESCENARIOS):
            media = [(a + b) / 2 for a, b in zip(v["caliente"], v["frio"])]
            filas.append({"sistema": nombre, "escenario": "media", "metrica": m, **ic_bootstrap(media)})
    return filas, len(idx)


def tokens_prefijo(espec, consultas_por_idx, candidatos, idx):
    """prompt_eval_count del prefijo común de cada consulta (hasta "Article B:\\nTitle:"), sin el cierre."""
    marca = "Article B:\nTitle:"
    prefijos = {}
    for q in idx:
        c = consultas_por_idx[q]
        vista = Vista(c)
        construir = ro.prompts_c2(espec, {"idx": q, "escenario": c["escenario"]}, vista)
        texto_q, _ = ro.recortar(vista.texto(q), ro.MAX_PALABRAS_CONSULTA)
        prompt, _ = construir(candidatos[q][0], texto_q, "X")
        prefijos[q] = ro.llamar(prompt[:prompt.index(marca) + len(marca)]).prompt_eval_count - CIERRE_PLANTILLA
    return prefijos


def coste(diag, nombre, prefijos, cabecera, incluidas, picos):
    ok = [f for f in diag if "error" not in f and f["consulta"] in incluidas]
    filas_coste = sorted(leer_jsonl(DIR_RESULTADOS / f"coste_{nombre}.jsonl"), key=lambda f: f["ts"])
    presupuesto = comprobar_presupuesto(filas_coste, picos)
    filas_coste = [f for f in filas_coste[CALENTAMIENTO:] if f["consulta"] in incluidas]
    lat = [f["latencia_ms"] for f in filas_coste]
    agregado = agregar(filas_coste, descartar=0, picos=picos)
    primeras, resto, vistas = [], [], set()
    for f in ok:
        (resto if f["consulta"] in vistas else primeras).append(f)
        vistas.add(f["consulta"])
    con_cache = [f["prompt_eval_count"] - prefijos[f["consulta"]] for f in resto] + \
                [f["prompt_eval_count"] - cabecera for f in primeras]
    return {
        "presupuesto": presupuesto,
        "latencia_ms_por_consulta": agregado[0]["latencia_ms"],
        "latencia_ms_por_consulta_media": float(np.mean(lat)),
        "n_consultas_sin_calentamiento": len(lat),
        "tokens_in_por_consulta": agregado[0]["tokens_in"],
        "tokens_in_por_llamada_sin_cache_media": float(np.mean([f["prompt_eval_count"] for f in ok])),
        "tokens_in_por_llamada_sin_cache_max": int(max(f["prompt_eval_count"] for f in ok)),
        "tokens_in_por_llamada_con_cache_media_estimada": float(np.mean(con_cache)),
        "prompt_eval_ms_primera_llamada_media": float(np.mean([f["prompt_eval_ms"] for f in primeras])),
        "prompt_eval_ms_resto_media": float(np.mean([f["prompt_eval_ms"] for f in resto])),
        "vecinos_consulta_media": float(np.mean([f["n_vecinos_consulta"] for f in primeras])),
        "vecinos_candidato_media": float(np.mean([f["n_vecinos_candidato"] for f in ok])),
        "comunes_media": float(np.mean([f["n_comunes"] for f in ok])),
        "pct_llamadas_con_comunes": 100 * float(np.mean([f["n_comunes"] > 0 for f in ok])),
        "horas_estimadas_2000_consultas": float(np.mean(lat)) * 2000 / 3.6e6,
    }


def main():
    ap = argparse.ArgumentParser()
    grupo = ap.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--variante", type=int, choices=sorted(VARIANTES))
    grupo.add_argument("--sistema", choices=sorted(SISTEMA_MODELO.values()))
    ap.add_argument("--escenarios", nargs="+", choices=ESCENARIOS, default=list(ESCENARIOS))
    args = ap.parse_args()
    dev = args.variante is not None
    if dev:
        espec = VARIANTES[args.variante]
        filas, n = tabla_metricas(args.variante)
        sistema = sistema_variante(args.variante)
        salida = {"sistema": sistema, "variante": args.variante, "modo": MODO, "tarea": TAREA,
                  "muestra": "desarrollo", "n_consultas": n, "metricas": filas, "por_escenario": {}}
        rutas_vram = [DIR_RESULTADOS / f"vram_{sistema_dev(args.variante, e)}_dev.json" for e in args.escenarios]
        ruta_salida = DIR_RESULTADOS / f"{sistema}_analisis_dev.json"
    else:
        espec, sistema = ro.C2, args.sistema
        ro.MODELO = {s: m for m, s in SISTEMA_MODELO.items()}[sistema]
        salida = {"sistema": sistema, "modelo": ro.MODELO, "variante": "congelada", "modo": MODO,
                  "muestra": "principal_sin_desarrollo", "por_escenario": {}}
        # Una medida para los dos escenarios o una por escenario (T2.7, tras el corte de luz)
        rutas_vram = [DIR_RESULTADOS / f"vram_{sistema}.json"] + [
            DIR_RESULTADOS / f"vram_{sistema}_{e}.json" for e in args.escenarios]
        ruta_salida = DIR_RESULTADOS / f"{sistema}_analisis.json"
    rutas_vram = sorted({r for r in rutas_vram if r.exists()})
    picos = picos_vram(rutas_vram)
    if not dev:
        tag = "3b" if sistema == "c2" else sistema.replace("c2_", "")
        ruta_t26 = DIR_RESULTADOS / f"vram_presupuesto_{tag}.json"
        pico_t26 = json.loads(ruta_t26.read_text(encoding="utf-8"))["pico_mib"]
        salida["pico_ejecucion_mib"], salida["pico_t26_con_embeddings_mib"] = dict(picos), pico_t26
        picos = {i: max(p, pico_t26) for i, p in picos.items()}
        rutas_vram.append(ruta_t26)
    cabecera = ro.llamar("").prompt_eval_count - CIERRE_PLANTILLA
    for esc in args.escenarios:
        s = sistema_dev(args.variante, esc) if dev else sistema
        nombre = nombre_ejecucion(s, MODO, esc, None, dev)
        consultas = {c["idx"]: c for c in cargar_consultas(esc)}
        incluidas = {q for q, c in consultas.items() if c["es_desarrollo"] == dev}
        diag = [f for f in leer_jsonl(DIR_RESULTADOS / f"diagnostico_{nombre}.jsonl") if f["consulta"] in incluidas]
        punt = [l for l in leer_jsonl(DIR_RESULTADOS / f"puntuaciones_{nombre}.jsonl") if l["idx"] in incluidas]
        prefijos = tokens_prefijo(espec, consultas, cargar_candidatos("recuperador", esc), [l["idx"] for l in punt])
        salida["por_escenario"][esc] = {**saturacion_y_empates(diag, punt),
                                        "coste": coste(diag, nombre, prefijos, cabecera, incluidas, picos)}
    salida["horas_estimadas_2000_consultas_escenarios"] = sum(
        salida["por_escenario"][e]["coste"]["horas_estimadas_2000_consultas"] for e in args.escenarios)
    salida["tokens_cabecera_plantilla"] = cabecera
    salida["vram"] = {r.name: json.loads(r.read_text(encoding="utf-8")) for r in rutas_vram}
    ruta_salida.write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(salida, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
