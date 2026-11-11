"""C1: reordenación con un modelo pequeño, con coste registrado (T2.2 en desarrollo; T2.3, T2.7b y T2.7c en la principal).

Puntúa con el scorer `c1` de pipeline/reordenacion.py (política P-cota). Desde T2.2b,
P-none ya no se calcula como sistema aparte; los casos (i)/(ii)/(iii) se siguen
contando en el diagnóstico.

Por defecto, qwen2.5:3b, el prompt congelado (pipeline/prompts/c1.txt), modo
recuperador y los dos escenarios. Con --desarrollo, solo las 200 consultas de
desarrollo (invariante 4: sirven para ajustar y no se reportan); con --desarrollo
--variante N, la variante N de experimentos/variantes_c1.py (sistema "c1" para V1,
"c1_vN" para las demás).

Con --modelo qwen2.5:14b-instruct-q4_K_M (T2.7b; sistema c1_14b) o
qwen2.5:7b-instruct-q4_K_M (T2.7c; sistema c1_7b), el prompt congelado con otro modelo: solo en la muestra principal, modo recuperador y con --nivel, el nivel
de presupuesto que se anota en cada registro de coste. Como en C2, el agregado de coste
no se escribe aquí, porque necesita el pico de VRAM que mide arnes.medir_vram desde
fuera: lo hace experimentos/analizar_c1.py --principal --modelo M.

Escribe en experimentos/resultados/, con nombre = {sistema}_{modo}_{escenario}[_s{s}][_dev]:
- puntuaciones_{nombre}.jsonl y metricas_{nombre}.json (no versionados);
- diagnostico_{nombre}.jsonl: una línea por llamada, con el token generado, el
  top_logprobs completo, el caso y la confianza (no versionado);
- coste_{nombre}.jsonl (se reescribe) y coste_{nombre}_agregado.json;
- c1_resumen[_dev].json: métricas y recuentos del diagnóstico, una fila por
  (sistema, modo, escenario, semilla, tarea); se actualiza por escenario;
- en la muestra principal, además, crudas/{id_ejecucion}/puntuaciones_{nombre}.jsonl.gz
  (versionado; no se sobrescribe).

No pisa puntuaciones de C1 ya guardadas salvo con --sobrescribir (son horas de
cómputo). Con --solo-resumen recalcula métricas y resumen desde lo guardado.

Uso (desde la raíz del repo, con Ollama en marcha):
    .venv/Scripts/python.exe experimentos/evaluar_c1.py [--desarrollo [--variante N]] [--escenario frio] [--sobrescribir]
    .venv/Scripts/python.exe experimentos/evaluar_c1.py --modelo qwen2.5:14b-instruct-q4_K_M --nivel alto [--escenario frio]
    .venv/Scripts/python.exe experimentos/evaluar_c1.py --modelo qwen2.5:7b-instruct-q4_K_M --nivel medio [--escenario frio]
"""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.agregar import agregar, leer  # noqa: E402
from arnes.candidatos import K_RECUPERADOR, N_CANDIDATOS  # noqa: E402
from arnes.coste import NIVELES, con_coste, config_ejecucion  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, evaluar, nombre_ejecucion, recalcular_metricas  # noqa: E402
from arnes.scorer import SCORERS  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402  (registra c1)
from variantes_c1 import VARIANTES, sistema as sistema_variante  # noqa: E402

ESCENARIOS = ("caliente", "frio")
DIR_CRUDAS = DIR_RESULTADOS / "crudas"
SISTEMA_MODELO = {"qwen2.5:3b": "c1", "qwen2.5:7b-instruct-q4_K_M": "c1_7b", "qwen2.5:14b-instruct-q4_K_M": "c1_14b"}


def leer_jsonl(ruta):
    return [json.loads(l) for l in Path(ruta).read_text(encoding="utf-8").splitlines()]


def contar(ruta_diagnostico):
    """Recuentos del diagnóstico: casos, errores, respuestas que no son yes/no, recortes y truncados."""
    filas = leer_jsonl(ruta_diagnostico)
    ok = [f for f in filas if "error" not in f]
    return {"llamadas": len(filas), "error": len(filas) - len(ok),
            **{c: sum(f["caso"] == c for f in ok) for c in ro.CASOS},
            "respuesta_yes": sum(ro.normalizar(f["token"]) == "yes" for f in ok),
            "respuesta_no": sum(ro.normalizar(f["token"]) == "no" for f in ok),
            "respuesta_no_si_no": sum(ro.normalizar(f["token"]) not in ("yes", "no") for f in ok),
            "consultas": len({f["consulta"] for f in filas}),
            "consultas_recortadas": len({f["consulta"] for f in filas if f["recorte_consulta"]}),
            "llamadas_candidato_recortado": sum(f["recorte_candidato"] for f in filas),
            "candidatos_distintos_recortados": len({f["candidato"] for f in filas if f["recorte_candidato"]}),
            "truncados": sum(f["prompt_eval_count"] >= ro.NUM_CTX for f in ok)}


def filas_resumen(res, politica, recuentos, variante):
    """Una fila por tarea. Con --desarrollo, el agregado es el de las 200 de desarrollo."""
    dev = res["solo_desarrollo"]
    agregado = res["agregado"] if dev else res["agregado_sin_desarrollo"]
    return [{"sistema": res["sistema"], "variante": variante, "politica": politica, "modo": res["modo"],
             "escenario": res["escenario"], "semilla": res["semilla"], "tarea": tarea,
             "id_ejecucion": res["id_ejecucion"],
             "muestra": "desarrollo" if dev else "principal_sin_desarrollo",
             "n_consultas": res["n_consultas"] if dev else res["n_consultas_sin_desarrollo"],
             "tasa_none": res["tasa_none"], "malformadas": res["malformadas"], "diagnostico": recuentos,
             **{m: {"media": v["media"], "ic95": v["ic95"]} for m, v in ms.items()}}
            for tarea, ms in agregado.items()]


def versionar_crudas(ruta_puntuaciones, id_ejecucion):
    """Copia comprimida de las puntuaciones crudas de la principal, en su carpeta por id_ejecucion."""
    destino = DIR_CRUDAS / id_ejecucion / f"{Path(ruta_puntuaciones).name}.gz"
    if destino.exists():
        sys.exit(f"Ya existe {destino}: no se sobrescribe")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_puntuaciones, "rb") as f, gzip.open(destino, "wb") as g:
        shutil.copyfileobj(f, g)
    return destino


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desarrollo", action="store_true", help="solo las 200 consultas de desarrollo")
    ap.add_argument("--variante", type=int, choices=sorted(VARIANTES), default=None,
                    help="variante del prompt (solo con --desarrollo); por defecto, el prompt congelado")
    ap.add_argument("--escenario", choices=ESCENARIOS, default=None, help="por defecto, los dos")
    ap.add_argument("--modo", choices=("recuperador", "aleatorio"), default="recuperador")
    ap.add_argument("--semilla", type=int, default=None, help="semilla de candidatos del modo aleatorio")
    ap.add_argument("--modelo", choices=sorted(SISTEMA_MODELO), default=ro.MODELO)
    ap.add_argument("--nivel", choices=sorted(NIVELES), default=None,
                    help="nivel de presupuesto (obligatorio con un modelo que no sea el 3B)")
    ap.add_argument("--sobrescribir", action="store_true", help="pisar puntuaciones de C1 ya guardadas")
    ap.add_argument("--solo-resumen", action="store_true",
                    help="recalcular métricas y resumen desde lo guardado, sin puntuar")
    args = ap.parse_args()
    if (args.modo == "aleatorio") != (args.semilla is not None):
        ap.error("--semilla va con --modo aleatorio, y solo con él")
    if args.variante is not None and not args.desarrollo:
        ap.error("--variante solo con --desarrollo: la muestra principal usa el prompt congelado")
    if args.modelo != ro.MODELO and (args.desarrollo or args.modo != "recuperador" or args.nivel is None):
        ap.error("con otro modelo, solo la muestra principal en modo recuperador y con --nivel")
    if args.nivel is not None and args.desarrollo:
        ap.error("--nivel solo en la muestra principal")
    ro.MODELO = args.modelo
    if args.variante is not None:
        ro.PLANTILLA = VARIANTES[args.variante]
        sistema = sistema_variante(args.variante)
    else:
        sistema = SISTEMA_MODELO[args.modelo]
    variante = args.variante if args.variante is not None else "congelada"
    escenarios = (args.escenario,) if args.escenario else ESCENARIOS
    nombres = {e: nombre_ejecucion(sistema, args.modo, e, args.semilla, args.desarrollo) for e in escenarios}
    if not args.solo_resumen and not args.sobrescribir:
        existentes = [n for n in nombres.values() if (DIR_RESULTADOS / f"puntuaciones_{n}.jsonl").exists()]
        if existentes:
            sys.exit(f"Ya hay puntuaciones de C1 ({existentes}); usa --sobrescribir para pisarlas")

    ruta_resumen = DIR_RESULTADOS / f"c1_resumen{'_dev' if args.desarrollo else ''}.json"
    resumen = json.loads(ruta_resumen.read_text(encoding="utf-8")) if ruta_resumen.exists() else []
    for escenario in escenarios:
        nombre = nombres[escenario]
        ruta_p = DIR_RESULTADOS / f"puntuaciones_{nombre}.jsonl"
        ruta_diag = DIR_RESULTADOS / f"diagnostico_{nombre}.jsonl"
        ruta_coste = DIR_RESULTADOS / f"coste_{nombre}.jsonl"
        if args.solo_resumen:
            res = recalcular_metricas(sistema, args.modo, escenario, args.semilla, args.desarrollo)
        else:
            ruta_diag.unlink(missing_ok=True)
            ruta_coste.unlink(missing_ok=True)
            ro.RUTA_DIAGNOSTICO = ruta_diag
            ro.reiniciar_contadores()
            config = config_ejecucion(
                modelo=ro.MODELO, K=K_RECUPERADOR if args.modo == "recuperador" else None,
                num_ctx=ro.NUM_CTX, prompt=ro.PLANTILLA,
                **({"N": N_CANDIDATOS} if args.modo == "aleatorio" else {}),
                variante=variante, top_logprobs=ro.TOP_LOGPROBS, max_palabras_consulta=ro.MAX_PALABRAS_CONSULTA,
                max_palabras_candidato=ro.MAX_PALABRAS_CANDIDATO, keep_alive=ro.KEEP_ALIVE,
                politica_devuelta="P-cota", via="/api/chat", **({"nivel": args.nivel} if args.nivel else {}))
            scorer = con_coste(ruta_coste, sistema=sistema, modo=args.modo, semilla=args.semilla,
                               escenario=escenario, config=config,
                               **({"nivel": args.nivel} if args.nivel else {}))(SCORERS["c1"])
            res = evaluar(sistema, args.modo, escenario, args.semilla, solo_desarrollo=args.desarrollo,
                          scorer=scorer)
            if args.nivel is None:
                (DIR_RESULTADOS / f"coste_{nombre}_agregado.json").write_text(
                    json.dumps(agregar(leer([ruta_coste])), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if not args.desarrollo:
                print(f"Crudas versionadas en {versionar_crudas(ruta_p, res['id_ejecucion'])}", flush=True)
        recuentos = contar(ruta_diag)
        clave = (sistema, args.modo, escenario, args.semilla)
        resumen = [r for r in resumen if (r["sistema"], r["modo"], r["escenario"], r["semilla"]) != clave]
        filas = filas_resumen(res, "P-cota", recuentos, variante)
        if args.nivel is not None:
            filas = [{**f, "modelo": ro.MODELO, "nivel": args.nivel} for f in filas]
        resumen += filas
        ruta_resumen.write_text(json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{escenario} {args.modo}: hecho; {recuentos}", flush=True)
    print(f"Escrito {ruta_resumen.name}")


if __name__ == "__main__":
    main()
