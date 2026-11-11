"""Análisis de C1: saturación, casos, empates y coste; en desarrollo, también métricas.

En desarrollo (por defecto; T2.2), lee lo que deja experimentos/evaluar_c1.py
--desarrollo [--variante N] (modo recuperador) y, por escenario y en la media de los dos:
- MRR y precision@1 de la tarea implícita, junto a C0, BM25 y, en caliente,
  Adamic-Adar (B), y a los controles aleatorio y oráculo, sobre las mismas consultas
  (las de C1). La media de los dos escenarios se calcula por consulta (son las mismas
  200) y su IC por bootstrap, con el arnés;
- saturación, casos, empates y coste (ver abajo).
Escribe {sistema}_analisis_dev.json (c1_analisis_dev.json para V1).

Con --principal (T2.3), lee la ejecución del prompt congelado sobre la muestra
principal y lo calcula solo sobre las 1800 consultas que no son de desarrollo
(invariante 4); las métricas y las diferencias pareadas están en
experimentos/tabla_principal.py. Escribe c1_analisis.json.

Con --principal --modelo M (T2.7b y T2.7c; C1 con el 14B o el 7B, sistemas c1_14b y
c1_7b), como C2 en
experimentos/analizar_c2.py: el coste se agrega con el pico de VRAM, y arnes.agregar
falla si supera el tope del nivel de los registros. El pico que se comprueba es el mayor
entre el de la propia ejecución (vram_{sistema}_{escenario}.json) y el de T2.6 para el
mismo modelo con el de embeddings a la vez (vram_presupuesto_{tag}.json, medido con C2,
cuyo contexto es más largo). Escribe {sistema}_analisis.json.

Lo que se calcula en los dos casos:
- saturación: % de respuestas "Yes", percentiles de la confianza y % de llamadas en
  cada caso (ambos / falta uno / faltan ambos), y tasa de None;
- empates: valores distintos de confianza por lista y % de listas con el primer
  puesto empatado (P-cota);
- coste: latencia por consulta (p50/p95, sin las 3 de calentamiento), tokens de
  entrada por llamada sin caché (prompt_eval_count, que Ollama da completo aunque use
  la caché) y con caché (estimación: menos el prefijo común de la consulta, medido
  aparte con una llamada por consulta), VRAM (de arnes.medir_vram, si existe) y horas
  estimadas para 2000 consultas x 2 escenarios.

Uso (desde la raíz del repo, con Ollama en marcha para medir los prefijos):
    .venv/Scripts/python.exe experimentos/analizar_c1.py [--variante N | --principal [--modelo M]]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.agregar import agregar, comprobar_presupuesto, picos_vram  # noqa: E402
from arnes.candidatos import cargar_consultas  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, ic_bootstrap, nombre_ejecucion  # noqa: E402
from arnes.vista import Vista  # noqa: E402
from evaluar_c1 import SISTEMA_MODELO  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402
from variantes_c1 import VARIANTES, sistema as sistema_variante  # noqa: E402

MODO = "recuperador"
ESCENARIOS = ("caliente", "frio")
METRICAS = ("mrr", "precision@1")
TAREA = "implicita"
COMPARADOS = {"caliente": ("aleatorio", "oraculo", "c0", "bm25", "adamic_adar_B"),
              "frio": ("aleatorio", "oraculo", "c0", "bm25")}
CALENTAMIENTO = 3  # como arnes/agregar.py
CIERRE_PLANTILLA = 5  # <|im_end|> \n <|im_start|> assistant \n, tras el mensaje de usuario


def leer_jsonl(ruta):
    return [json.loads(l) for l in Path(ruta).read_text(encoding="utf-8").splitlines()]


def metricas(sistema, escenario, dev):
    nombre = nombre_ejecucion(sistema, MODO, escenario, None, dev)
    return json.loads((DIR_RESULTADOS / f"metricas_{nombre}.json").read_text(encoding="utf-8"))


def por_consulta(res, metrica, indices):
    v = dict(zip(res["idx"], res["por_consulta"][TAREA][metrica]))
    return [v[i] for i in indices]


def tabla_metricas(sistema):
    filas, por_sistema = [], {}
    idx = None
    for esc in ESCENARIOS:
        c1 = metricas(sistema, esc, True)
        idx = idx or c1["idx"]
        assert c1["idx"] == idx, "C1 debe tener las mismas consultas en los dos escenarios"
        sistemas = {sistema: c1, **{s: metricas(s, esc, False) for s in COMPARADOS[esc]}}
        for nombre, res in sistemas.items():
            for m in METRICAS:
                v = por_consulta(res, m, idx)
                por_sistema.setdefault((nombre, m), {})[esc] = v
                filas.append({"sistema": nombre, "escenario": esc, "metrica": m, **ic_bootstrap(v)})
    for (nombre, m), v in por_sistema.items():
        if set(v) == set(ESCENARIOS):
            media = [(a + b) / 2 for a, b in zip(v["caliente"], v["frio"])]
            filas.append({"sistema": nombre, "escenario": "media", "metrica": m, **ic_bootstrap(media)})
    return filas, len(idx)


def saturacion_y_empates(diag, puntuaciones):
    ok = [f for f in diag if "error" not in f]
    conf = np.asarray([f["conf_cota"] for f in ok if f["conf_cota"] is not None])
    vals_todas = [p for l in puntuaciones for _, p in l["puntuaciones"]]
    resultado = {
        "consultas": len(puntuaciones),
        "llamadas": len(diag), "errores": len(diag) - len(ok),
        "puntuaciones": len(vals_todas), "none": sum(p is None for p in vals_todas),
        "tasa_none": sum(p is None for p in vals_todas) / len(vals_todas),
        "pct_yes": 100 * np.mean([ro.normalizar(f["token"]) == "yes" for f in ok]),
        "pct_no_si_no": 100 * np.mean([ro.normalizar(f["token"]) not in ("yes", "no") for f in ok]),
        "casos": {c: sum(f["caso"] == c for f in ok) for c in ro.CASOS},
        "pct_casos": {c: 100 * np.mean([f["caso"] == c for f in ok]) for c in ro.CASOS},
        "truncados": sum(f["prompt_eval_count"] >= ro.NUM_CTX for f in ok),
        "confianza_cota_percentiles": {f"p{p}": float(np.percentile(conf, p)) for p in (0, 1, 5, 25, 50, 75, 95, 99, 100)},
        "confianza_cota_media": float(conf.mean()),
    }
    distintos, primero_empatado = [], 0
    for linea in puntuaciones:
        vals = [p for _, p in linea["puntuaciones"]]
        no_none = [v for v in vals if v is not None]
        distintos.append(len(set(no_none)) + (len(no_none) < len(vals)))
        if not no_none or no_none.count(max(no_none)) >= 2:
            primero_empatado += 1
    resultado["empates_P-cota"] = {
        "valores_distintos_por_lista_media": float(np.mean(distintos)),
        "valores_distintos_por_lista_min": int(min(distintos)),
        "listas_con_algun_empate": sum(d < len(l["puntuaciones"]) for d, l in zip(distintos, puntuaciones)),
        "tamano_lista_medio": float(np.mean([len(l["puntuaciones"]) for l in puntuaciones])),
        "tamano_lista_min": int(min(len(l["puntuaciones"]) for l in puntuaciones)),
        "pct_listas_primer_puesto_empatado": 100 * primero_empatado / len(puntuaciones)}
    return resultado


def tokens_prefijo(consultas_por_idx, idx):
    """prompt_eval_count del prefijo común de cada consulta (hasta "Article B:\\nTitle:"),
    sin el cierre de la plantilla. Una llamada por consulta."""
    marca = "Article B:\nTitle:"
    prefijos = {}
    for q in idx:
        vista = Vista(consultas_por_idx[q])
        texto_q, _ = ro.recortar(vista.texto(q), ro.MAX_PALABRAS_CONSULTA)
        prompt = ro.construir_prompt(vista.titulo(q), texto_q, "X", "X")
        prefijo = prompt[:prompt.index(marca) + len(marca)]
        prefijos[q] = ro.llamar(prefijo).prompt_eval_count - CIERRE_PLANTILLA
    return prefijos


def coste(diag, nombre, prefijos, cabecera, incluidas, picos=None):
    """incluidas: consultas que cuentan (las 200 en desarrollo, las 1800 en la principal).
    El calentamiento son las 3 primeras consultas de la ejecución, cuenten o no.
    picos: {id_ejecucion: pico_mib} de una ejecución con nivel (comprueba el tope)."""
    ok = [f for f in diag if "error" not in f and f["consulta"] in incluidas]
    filas_coste = sorted(leer_jsonl(DIR_RESULTADOS / f"coste_{nombre}.jsonl"), key=lambda f: f["ts"])
    presupuesto = comprobar_presupuesto(filas_coste, picos) if picos is not None else None
    filas_coste = [f for f in filas_coste[CALENTAMIENTO:] if f["consulta"] in incluidas]
    lat = [f["latencia_ms"] for f in filas_coste]
    agregado = agregar(filas_coste, descartar=0, picos=picos)
    primeras, resto = [], []
    vistas = set()
    for f in ok:
        (resto if f["consulta"] in vistas else primeras).append(f)
        vistas.add(f["consulta"])
    con_cache = [f["prompt_eval_count"] - prefijos[f["consulta"]] for f in resto] + \
                [f["prompt_eval_count"] - cabecera for f in primeras]
    return {
        **({"presupuesto": presupuesto} if presupuesto is not None else {}),
        "latencia_ms_por_consulta": agregado[0]["latencia_ms"] if agregado else None,
        "latencia_ms_por_consulta_media": float(np.mean(lat)),
        "n_consultas_sin_calentamiento": len(lat),
        "tokens_in_por_consulta": agregado[0]["tokens_in"] if agregado else None,
        "tokens_in_por_llamada_sin_cache_media": float(np.mean([f["prompt_eval_count"] for f in ok])),
        "tokens_in_por_llamada_sin_cache_max": int(max(f["prompt_eval_count"] for f in ok)),
        "tokens_in_por_llamada_con_cache_media_estimada": float(np.mean(con_cache)),
        "prompt_eval_ms_primera_llamada_media": float(np.mean([f["prompt_eval_ms"] for f in primeras])),
        "prompt_eval_ms_resto_media": float(np.mean([f["prompt_eval_ms"] for f in resto])),
        "horas_estimadas_2000_consultas": float(np.mean(lat)) * 2000 / 3.6e6,
    }


def main():
    ap = argparse.ArgumentParser()
    grupo = ap.add_mutually_exclusive_group()
    grupo.add_argument("--variante", type=int, choices=sorted(VARIANTES), default=1)
    grupo.add_argument("--principal", action="store_true")
    ap.add_argument("--modelo", choices=sorted(SISTEMA_MODELO), default=ro.MODELO,
                    help="con --principal: el modelo de la ejecución (por defecto, el 3B de T2.3)")
    args = ap.parse_args()
    if args.modelo != ro.MODELO and not args.principal:
        ap.error("--modelo solo con --principal")
    dev = not args.principal
    picos = None
    if dev:
        ro.PLANTILLA = VARIANTES[args.variante]
        sistema = sistema_variante(args.variante)
        filas, n = tabla_metricas(sistema)
        salida = {"sistema": sistema, "variante": args.variante, "modo": MODO, "tarea": TAREA,
                  "muestra": "desarrollo", "n_consultas": n, "metricas": filas, "por_escenario": {}}
        ruta_vram = DIR_RESULTADOS / f"vram_{sistema}_dev.json"
        ruta_salida = DIR_RESULTADOS / f"{sistema}_analisis_dev.json"
    elif args.modelo == ro.MODELO:
        sistema = "c1"
        salida = {"sistema": sistema, "variante": "congelada", "modo": MODO,
                  "muestra": "principal_sin_desarrollo", "por_escenario": {}}
        ruta_vram = DIR_RESULTADOS / "vram_c1.json"
        ruta_salida = DIR_RESULTADOS / "c1_analisis.json"
    else:
        sistema, ro.MODELO = SISTEMA_MODELO[args.modelo], args.modelo
        salida = {"sistema": sistema, "modelo": ro.MODELO, "variante": "congelada", "modo": MODO,
                  "muestra": "principal_sin_desarrollo", "por_escenario": {}}
        rutas_vram = [r for r in (DIR_RESULTADOS / f"vram_{sistema}_{e}.json" for e in ESCENARIOS) if r.exists()]
        ruta_t26 = DIR_RESULTADOS / f"vram_presupuesto_{sistema.replace('c1_', '')}.json"
        pico_t26 = json.loads(ruta_t26.read_text(encoding="utf-8"))["pico_mib"]
        picos_propios = picos_vram(rutas_vram)
        salida["pico_ejecucion_mib"], salida["pico_t26_con_embeddings_mib"] = picos_propios, pico_t26
        picos = {i: max(p, pico_t26) for i, p in picos_propios.items()}
        ruta_vram = None
        ruta_salida = DIR_RESULTADOS / f"{sistema}_analisis.json"
    cabecera = ro.llamar("").prompt_eval_count - CIERRE_PLANTILLA
    for esc in ESCENARIOS:
        nombre = nombre_ejecucion(sistema, MODO, esc, None, dev)
        consultas = {c["idx"]: c for c in cargar_consultas(esc)}
        incluidas = {q for q, c in consultas.items() if c["es_desarrollo"] == dev}
        diag = [f for f in leer_jsonl(DIR_RESULTADOS / f"diagnostico_{nombre}.jsonl")]
        punt = [l for l in leer_jsonl(DIR_RESULTADOS / f"puntuaciones_{nombre}.jsonl") if l["idx"] in incluidas]
        prefijos = tokens_prefijo(consultas, [l["idx"] for l in punt])
        salida["por_escenario"][esc] = {
            **saturacion_y_empates([f for f in diag if f["consulta"] in incluidas], punt),
            "coste": coste(diag, nombre, prefijos, cabecera, incluidas, picos)}
    horas = sum(salida["por_escenario"][e]["coste"]["horas_estimadas_2000_consultas"] for e in ESCENARIOS)
    salida["horas_estimadas_2000_consultas_x_2_escenarios"] = horas
    salida["tokens_cabecera_plantilla"] = cabecera
    if ruta_vram is not None:
        salida["vram"] = json.loads(ruta_vram.read_text(encoding="utf-8")) if ruta_vram.exists() else None
    else:
        salida["vram"] = {r.name: json.loads(r.read_text(encoding="utf-8")) for r in rutas_vram + [ruta_t26]}
    ruta_salida.write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(salida, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
