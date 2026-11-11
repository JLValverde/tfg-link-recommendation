"""Selección y congelación del contexto de C2 (T2.4), con la regla fijada antes de medir.

Regla (Roadmap Fase 2, T2.4, la misma que T2.2): gana la variante (V1, V2, V3) con
mayor MRR de la tarea implícita, modo recuperador, media de los dos escenarios
calculada por consulta, sobre las 200 consultas de desarrollo. Empate a tres
decimales: gana la de número menor.

En frío las tres variantes dan el mismo prompt, así que su frío es el de V1. Antes de
aplicar la regla se comprueba: para las llamadas de frío de V1 de las primeras
N_COMPROBAR consultas, el hash del prompt de V2 y de V3 coincide con el registrado.

Escribe experimentos/resultados/c2_seleccion_dev.json. Con --congelar, escribe la
especificación de la ganadora en pipeline/prompts/c2.json (texto canónico de
pipeline.reordenacion.texto_c2), que lee el scorer; se niega si ya existe.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/seleccionar_c2.py [--congelar]
"""

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.candidatos import cargar_consultas  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, ic_bootstrap, nombre_ejecucion  # noqa: E402
from arnes.vista import Vista  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402
from variantes_c2 import VARIANTES, sistema  # noqa: E402

MODO, TAREA, METRICA = "recuperador", "implicita", "mrr"
ESCENARIOS = ("caliente", "frio")
DECIMALES = 3
N_COMPROBAR = 5
RUTA_SALIDA = DIR_RESULTADOS / "c2_seleccion_dev.json"


def mrr_por_consulta(s, escenario):
    nombre = nombre_ejecucion(s, MODO, escenario, None, True)
    res = json.loads((DIR_RESULTADOS / f"metricas_{nombre}.json").read_text(encoding="utf-8"))
    assert res["solo_desarrollo"] and res["n_consultas"] == 200, nombre
    return dict(zip(res["idx"], res["por_consulta"][TAREA][METRICA])), res["id_ejecucion"]


def comprobar_frio():
    """Hashes de los prompts de V2 y V3 frente a los registrados en el frío de V1."""
    nombre = nombre_ejecucion(sistema(1), MODO, "frio", None, True)
    filas = [json.loads(l) for l in (DIR_RESULTADOS / f"diagnostico_{nombre}.jsonl").read_text(
        encoding="utf-8").splitlines()]
    consultas = {c["idx"]: c for c in cargar_consultas("frio")}
    elegidas = list(dict.fromkeys(f["consulta"] for f in filas))[:N_COMPROBAR]
    resultado = {"consultas": elegidas, "llamadas": 0, "coinciden": {2: 0, 3: 0}}
    for q in elegidas:
        vista = Vista(consultas[q])
        texto_q, _ = ro.recortar(vista.texto(q), ro.MAX_PALABRAS_CONSULTA)
        construir = {v: ro.prompts_c2(VARIANTES[v], {"idx": q, "escenario": "frio"}, vista) for v in (2, 3)}
        for f in (f for f in filas if f["consulta"] == q):
            texto_c, _ = ro.recortar(vista.texto(f["candidato"]), ro.MAX_PALABRAS_CANDIDATO)
            resultado["llamadas"] += 1
            for v in (2, 3):
                prompt, _ = construir[v](f["candidato"], texto_q, texto_c)
                resultado["coinciden"][v] += ro.hash_texto(prompt)[:16] == f["hash_prompt"]
    resultado["todas_coinciden"] = all(n == resultado["llamadas"] for n in resultado["coinciden"].values())
    return resultado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congelar", action="store_true", help="escribir la ganadora en pipeline/prompts/c2.json")
    args = ap.parse_args()
    frio = comprobar_frio()
    if not frio["todas_coinciden"]:
        sys.exit(f"El prompt de frío de V2 o V3 no coincide con el de V1: {frio}")
    frio_v1, id_frio = mrr_por_consulta(sistema(1), "frio")
    filas = []
    for v in sorted(VARIANTES):
        caliente, id_caliente = mrr_por_consulta(sistema(v), "caliente")
        assert set(caliente) == set(frio_v1), "las mismas 200 consultas en los dos escenarios"
        idx = sorted(caliente)
        filas.append({"variante": v, "sistema_caliente": sistema(v), "sistema_frio": sistema(1),
                      "id_ejecucion": {"caliente": id_caliente, "frio": id_frio},
                      "hash_prompt": ro.hash_texto(ro.texto_c2(VARIANTES[v])),
                      "mrr_caliente": ic_bootstrap([caliente[q] for q in idx]),
                      "mrr_frio": ic_bootstrap([frio_v1[q] for q in idx]),
                      "mrr_media": ic_bootstrap([(caliente[q] + frio_v1[q]) / 2 for q in idx])})
    ganadora = min(filas, key=lambda f: (-round(f["mrr_media"]["media"], DECIMALES), f["variante"]))
    redondeo = round(ganadora["mrr_media"]["media"], DECIMALES)
    salida = {"regla": "mayor MRR de la tarea implícita, modo recuperador, media de los dos escenarios "
                       "calculada por consulta, 200 consultas de desarrollo; empate a 3 decimales: "
                       "gana la de número menor",
              "comprobacion_frio": frio, "variantes": filas, "ganadora": ganadora["variante"],
              "empate_a_3_decimales": sum(round(f["mrr_media"]["media"], DECIMALES) == redondeo for f in filas) > 1}
    RUTA_SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(salida, ensure_ascii=False, indent=1))
    if args.congelar:
        if ro.RUTA_C2.exists():
            sys.exit(f"{ro.RUTA_C2} ya existe: el contexto congelado no se toca")
        ro.RUTA_C2.write_bytes(ro.texto_c2(VARIANTES[ganadora["variante"]]).encode("utf-8"))
        print(f"Congelada V{ganadora['variante']} en {ro.RUTA_C2.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
