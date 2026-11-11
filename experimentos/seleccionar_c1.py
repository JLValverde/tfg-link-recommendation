"""Selección y congelación del prompt de C1 (T2.2b), con la regla fijada antes de medir.

Regla (Roadmap Fase 2, T2.2): gana la variante (V1, V2, V3) con mayor MRR de la tarea
implícita, modo recuperador, media de los dos escenarios calculada por consulta,
sobre las 200 consultas de desarrollo. Empate a tres decimales: gana la de número menor.

Lee metricas_{sistema}_recuperador_{escenario}_dev.json de cada variante y escribe
experimentos/resultados/c1_seleccion_dev.json. Con --congelar, escribe el texto de la
ganadora en pipeline/prompts/c1.txt, que lee el scorer; se niega si ya existe (desde
la congelación ese fichero no se toca).

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/seleccionar_c1.py [--congelar]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.evaluar import DIR_RESULTADOS, ic_bootstrap, nombre_ejecucion  # noqa: E402
from variantes_c1 import VARIANTES, sistema  # noqa: E402

MODO, TAREA, METRICA = "recuperador", "implicita", "mrr"
ESCENARIOS = ("caliente", "frio")
DECIMALES = 3
RUTA_PROMPT = RAIZ / "pipeline" / "prompts" / "c1.txt"
RUTA_SALIDA = DIR_RESULTADOS / "c1_seleccion_dev.json"


def mrr_por_consulta(variante, escenario):
    nombre = nombre_ejecucion(sistema(variante), MODO, escenario, None, True)
    res = json.loads((DIR_RESULTADOS / f"metricas_{nombre}.json").read_text(encoding="utf-8"))
    assert res["solo_desarrollo"] and res["n_consultas"] == 200, nombre
    return dict(zip(res["idx"], res["por_consulta"][TAREA][METRICA])), res["id_ejecucion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congelar", action="store_true", help="escribir la ganadora en pipeline/prompts/c1.txt")
    args = ap.parse_args()
    filas = []
    for v in sorted(VARIANTES):
        por_esc, ids = {}, {}
        for esc in ESCENARIOS:
            por_esc[esc], ids[esc] = mrr_por_consulta(v, esc)
        assert set(por_esc["caliente"]) == set(por_esc["frio"]), "las mismas 200 consultas en los dos escenarios"
        idx = sorted(por_esc["caliente"])
        media = [(por_esc["caliente"][q] + por_esc["frio"][q]) / 2 for q in idx]
        filas.append({"variante": v, "sistema": sistema(v), "id_ejecucion": ids,
                      "hash_prompt": hashlib.sha256(VARIANTES[v].encode("utf-8")).hexdigest(),
                      **{f"mrr_{esc}": ic_bootstrap([por_esc[esc][q] for q in idx]) for esc in ESCENARIOS},
                      "mrr_media": ic_bootstrap(media)})
    # Mayor MRR medio redondeado a 3 decimales; en empate, la de número menor
    ganadora = min(filas, key=lambda f: (-round(f["mrr_media"]["media"], DECIMALES), f["variante"]))
    salida = {"regla": "mayor MRR de la tarea implícita, modo recuperador, media de los dos escenarios "
                       "calculada por consulta, 200 consultas de desarrollo; empate a 3 decimales: "
                       "gana la de número menor",
              "variantes": filas, "ganadora": ganadora["variante"],
              "empate_a_3_decimales": sum(round(f["mrr_media"]["media"], DECIMALES)
                                          == round(ganadora["mrr_media"]["media"], DECIMALES) for f in filas) > 1}
    RUTA_SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(salida, ensure_ascii=False, indent=1))
    if args.congelar:
        if RUTA_PROMPT.exists():
            sys.exit(f"{RUTA_PROMPT} ya existe: el prompt congelado no se toca")
        RUTA_PROMPT.parent.mkdir(parents=True, exist_ok=True)
        RUTA_PROMPT.write_bytes(VARIANTES[ganadora["variante"]].encode("utf-8"))
        print(f"Congelada V{ganadora['variante']} en {RUTA_PROMPT.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
