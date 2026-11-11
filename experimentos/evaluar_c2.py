"""C2: C1 congelado más el contexto estructural, con coste registrado (T2.4, T2.5, T2.7).

Puntúa con el scorer `c2` de pipeline/reordenacion.py (política P-cota).

- Desarrollo (T2.4): --desarrollo --variante N, con la especificación N de
  experimentos/variantes_c2.py (sistema c2_vN). En frío las tres variantes dan el
  mismo prompt: frío se ejecuta solo con V1 y vale para las tres; con V2 y V3 solo
  se acepta caliente.
- Principal (T2.5, T2.7): la especificación congelada (pipeline/prompts/c2.json), con
  --modelo (por defecto qwen2.5:3b; sistema c2, c2_7b o c2_14b) y --nivel, el nivel
  de presupuesto que se anota en cada registro de coste (obligatorio en la principal).
  El agregado de coste no se escribe aquí: necesita el pico de VRAM de la ejecución,
  que mide arnes.medir_vram desde fuera; lo hace experimentos/analizar_c2.py.

Escribe en experimentos/resultados/, con nombre = {sistema}_{modo}_{escenario}[_dev]:
puntuaciones_*, metricas_*, diagnostico_* (no versionados), coste_*.jsonl,
c2_resumen[_dev].json y, en la principal, crudas/{id_ejecucion}/puntuaciones_*.jsonl.gz.
No pisa puntuaciones ya guardadas salvo con --sobrescribir.

Uso (desde la raíz del repo, con Ollama en marcha):
    .venv/Scripts/python.exe experimentos/evaluar_c2.py --desarrollo --variante N [--escenario E]
    .venv/Scripts/python.exe experimentos/evaluar_c2.py --nivel bajo [--modelo M] [--escenario E]
"""

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.agregar import agregar, leer  # noqa: E402
from arnes.candidatos import K_RECUPERADOR  # noqa: E402
from arnes.coste import NIVELES, con_coste, config_ejecucion  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, evaluar, nombre_ejecucion  # noqa: E402
from arnes.scorer import SCORERS  # noqa: E402
from evaluar_c1 import contar, filas_resumen, versionar_crudas  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402  (registra c2)
from variantes_c2 import VARIANTES, sistema as sistema_variante  # noqa: E402

ESCENARIOS = ("caliente", "frio")
MODO = "recuperador"
SISTEMA_MODELO = {"qwen2.5:3b": "c2", "qwen2.5:7b-instruct-q4_K_M": "c2_7b",
                  "qwen2.5:14b-instruct-q4_K_M": "c2_14b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desarrollo", action="store_true", help="solo las 200 consultas de desarrollo")
    ap.add_argument("--variante", type=int, choices=sorted(VARIANTES), default=None,
                    help="variante del contexto (solo con --desarrollo)")
    ap.add_argument("--escenario", choices=ESCENARIOS, default=None)
    ap.add_argument("--modelo", choices=sorted(SISTEMA_MODELO), default=ro.MODELO)
    ap.add_argument("--nivel", choices=sorted(NIVELES), default=None,
                    help="nivel de presupuesto (obligatorio en la muestra principal)")
    ap.add_argument("--sobrescribir", action="store_true")
    args = ap.parse_args()
    if args.desarrollo != (args.variante is not None):
        ap.error("--variante va con --desarrollo, y solo con él")
    if args.desarrollo and (args.nivel or args.modelo != ro.MODELO):
        ap.error("en desarrollo solo qwen2.5:3b y sin nivel")
    if not args.desarrollo and args.nivel is None:
        ap.error("la muestra principal necesita --nivel")

    if args.desarrollo:
        espec, sistema, variante = VARIANTES[args.variante], sistema_variante(args.variante), args.variante
        por_defecto = ESCENARIOS if args.variante == 1 else ("caliente",)
        if args.variante != 1 and args.escenario == "frio":
            ap.error("frío es el mismo prompt en las tres variantes: se ejecuta solo con V1")
    else:
        if ro.C2 is None:
            sys.exit("C2 no está congelado (pipeline/prompts/c2.json)")
        espec, sistema, variante = ro.C2, SISTEMA_MODELO[args.modelo], "congelada"
        por_defecto = ESCENARIOS
    ro.C2, ro.MODELO = espec, args.modelo
    escenarios = (args.escenario,) if args.escenario else por_defecto
    nombres = {e: nombre_ejecucion(sistema, MODO, e, None, args.desarrollo) for e in escenarios}
    if not args.sobrescribir:
        existentes = [n for n in nombres.values() if (DIR_RESULTADOS / f"puntuaciones_{n}.jsonl").exists()]
        if existentes:
            sys.exit(f"Ya hay puntuaciones de C2 ({existentes}); usa --sobrescribir para pisarlas")

    ruta_resumen = DIR_RESULTADOS / f"c2_resumen{'_dev' if args.desarrollo else ''}.json"
    resumen = json.loads(ruta_resumen.read_text(encoding="utf-8")) if ruta_resumen.exists() else []
    for escenario in escenarios:
        nombre = nombres[escenario]
        ruta_p = DIR_RESULTADOS / f"puntuaciones_{nombre}.jsonl"
        ruta_diag = DIR_RESULTADOS / f"diagnostico_{nombre}.jsonl"
        ruta_coste = DIR_RESULTADOS / f"coste_{nombre}.jsonl"
        ruta_diag.unlink(missing_ok=True)
        ruta_coste.unlink(missing_ok=True)
        ro.RUTA_DIAGNOSTICO = ruta_diag
        ro.reiniciar_contadores()
        config = config_ejecucion(
            modelo=ro.MODELO, K=K_RECUPERADOR, num_ctx=ro.NUM_CTX, prompt=ro.texto_c2(espec),
            variante=variante, prompt_c1=ro.hash_texto(ro.PLANTILLA), top_logprobs=ro.TOP_LOGPROBS,
            max_palabras_consulta=ro.MAX_PALABRAS_CONSULTA, max_palabras_candidato=ro.MAX_PALABRAS_CANDIDATO,
            keep_alive=ro.KEEP_ALIVE, politica_devuelta="P-cota", via="/api/chat", nivel=args.nivel)
        scorer = con_coste(ruta_coste, sistema=sistema, modo=MODO, escenario=escenario, config=config,
                           **({"nivel": args.nivel} if args.nivel else {}))(SCORERS["c2"])
        res = evaluar(sistema, MODO, escenario, None, solo_desarrollo=args.desarrollo, scorer=scorer)
        if args.desarrollo:
            (DIR_RESULTADOS / f"coste_{nombre}_agregado.json").write_text(
                json.dumps(agregar(leer([ruta_coste])), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            print(f"Crudas versionadas en {versionar_crudas(ruta_p, res['id_ejecucion'])}", flush=True)
        recuentos = contar(ruta_diag)
        clave = (sistema, MODO, escenario, None)
        resumen = [r for r in resumen if (r["sistema"], r["modo"], r["escenario"], r["semilla"]) != clave]
        resumen += [{**f, "modelo": ro.MODELO, "nivel": args.nivel}
                    for f in filas_resumen(res, "P-cota", recuentos, variante)]
        ruta_resumen.write_text(json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{escenario} {MODO}: hecho; {recuentos}", flush=True)
    print(f"Escrito {ruta_resumen.name}")


if __name__ == "__main__":
    main()
