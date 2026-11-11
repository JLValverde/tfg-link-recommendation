"""Tabla de baselines (T1.11; fusión RRF añadida en T2.1).

Sobre las 1800 consultas de la muestra principal que no son de desarrollo, para
cada modo y escenario y en las tres tareas: aleatorio y oráculo (controles), C0,
BM25, vecinos comunes y Adamic-Adar sobre B y sobre A y la fusión RRF de BM25 y
Adamic-Adar (B) (estos tres últimos grupos, solo en caliente; en frío "no aplica"),
con precision@1, precision_norm@5, MRR y hits@5, media e IC 95 %.
El modo aleatorio se muestra con la semilla de candidatos 1 (las semillas 2 y 3
están en los *_resumen.json).

Diferencias pareadas (bootstrap pareado, IC 95 %) en modo recuperador:
- BM25 - C0 (los dos escenarios);
- cada heurística sobre B - C0 (caliente);
- cada heurística sobre A - la misma sobre B (caliente);
- fusión - Adamic-Adar (B) y fusión - BM25 (caliente).

No puntúa nada: lee los *_resumen.json y los metricas_*.json ya calculados.
Escribe experimentos/resultados/tabla_baselines.md y tabla_baselines.json.
`generar()` también la usa experimentos/tabla_principal.py, que añade C1, C2, las réplicas
y, aparte, el análisis exploratorio (a posteriori) de T2.8b.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/tabla_baselines.py
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from arnes.candidatos import cargar_consultas  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, diferencia_pareada, nombre_ejecucion  # noqa: E402

METRICAS = ("precision@1", "precision_norm@5", "mrr", "hits@5")
TAREAS = ("completa", "implicita", "implicita_solo_ancla")
NOMBRE_TAREA = {"completa": "completa", "implicita": "implícita", "implicita_solo_ancla": "implícita solo ancla"}
ESCENARIOS = ("caliente", "frio")
NOMBRE_ESC = {"caliente": "caliente", "frio": "frío"}
SISTEMAS = ("aleatorio", "oraculo", "c0", "bm25",
            "vecinos_comunes_B", "adamic_adar_B", "vecinos_comunes_A", "adamic_adar_A", "rrf_bm25_aa_B")
NOMBRE_SIS = {"aleatorio": "aleatorio (control)", "oraculo": "oráculo (control)", "c0": "C0", "bm25": "BM25",
              "vecinos_comunes_B": "vecinos comunes (B)", "adamic_adar_B": "Adamic-Adar (B)",
              "vecinos_comunes_A": "vecinos comunes (A)", "adamic_adar_A": "Adamic-Adar (A)",
              "rrf_bm25_aa_B": "fusión RRF BM25 + Adamic-Adar (B)"}
# Sin sentido en frío: se reportan como "no aplica"
ESTRUCTURALES = ("vecinos_comunes_B", "adamic_adar_B", "vecinos_comunes_A", "adamic_adar_A", "rrf_bm25_aa_B")
MODOS = (("recuperador", None), ("aleatorio", 1))
RESUMENES = ("controles_resumen.json", "baselines_resumen.json", "estructurales_resumen.json",
             "fusion_resumen.json")
# Diferencias pareadas, modo recuperador: (A, B, escenario)
COMPARACIONES = ([("bm25", "c0", esc) for esc in ESCENARIOS]
                 + [(h, "c0", "caliente") for h in ("vecinos_comunes_B", "adamic_adar_B")]
                 + [(f"{h}_A", f"{h}_B", "caliente") for h in ("vecinos_comunes", "adamic_adar")]
                 + [("rrf_bm25_aa_B", b, "caliente") for b in ("adamic_adar_B", "bm25")])
TITULO = "Tabla de baselines (T1.11; fusión RRF en T2.1)"


def cargar_filas(resumenes=RESUMENES):
    filas = {}
    for nombre in resumenes:
        for r in json.loads((DIR_RESULTADOS / nombre).read_text(encoding="utf-8")):
            filas[(r["sistema"], r["modo"], r["escenario"], r["semilla"], r["tarea"])] = r
    return filas


def cargar_metricas(sistema, modo, escenario, semilla):
    nombre = nombre_ejecucion(sistema, modo, escenario, semilla)
    return json.loads((DIR_RESULTADOS / f"metricas_{nombre}.json").read_text(encoding="utf-8"))


def fmt(v):
    return f"{v['media']:.3f} [{v['ic95'][0]:.3f}, {v['ic95'][1]:.3f}]".replace(".", ",")


def fmt_dif(d):
    s = lambda x: f"{x:+.3f}".replace(".", ",").replace("-", "−")  # noqa: E731
    marca = "" if d["ic95"][0] <= 0 <= d["ic95"][1] else " *"
    return f"{s(d['diferencia'])} [{s(d['ic95'][0])}, {s(d['ic95'][1])}]{marca}"


def generar(base, titulo=TITULO, sistemas=SISTEMAS, nombres=NOMBRE_SIS, resumenes=RESUMENES,
            comparaciones=COMPARACIONES, no_ejecutados=(), nota=None, exploratorio=None):
    """Escribe {base}.md y {base}.json en experimentos/resultados/.

    no_ejecutados: (sistema, modo) o (sistema, modo, escenario) que no se han ejecutado; salen
    como «no ejecutado».
    nota: párrafo extra para la cabecera del .md.
    exploratorio: sistemas fuera del protocolo (T2.8b), que van al final, en su propia sección
    "Análisis exploratorio (a posteriori)" del .md y en la clave "exploratorio_a_posteriori" del
    .json, nunca en "tabla" ni en "diferencias". Un dict con "sistemas", "nombres", "resumen",
    "escenarios", "comparaciones" y "nota". Sin él, la salida no cambia.
    """
    filas = cargar_filas(resumenes)
    sin_dev = {e: [c["idx"] for c in cargar_consultas(e) if not c["es_desarrollo"]] for e in ESCENARIOS}
    salida = {"muestra": "principal_sin_desarrollo", "n_consultas": {e: len(v) for e, v in sin_dev.items()},
              "semilla_modo_aleatorio": 1, "metricas": METRICAS, "tabla": [], "diferencias": []}
    md = [f"# {titulo}", "",
          "Media [IC 95 %] sobre las 1800 consultas de la muestra principal que no son de desarrollo. "
          f"Generada por `experimentos/{base}.py` a partir de los `*_resumen.json` y `metricas_*.json`.",
          "Las heurísticas estructurales y la fusión solo se evalúan en caliente; en frío, «no aplica». "
          "El modo aleatorio se muestra con la semilla de candidatos 1.", ""]
    if nota:
        md += [nota, ""]

    for modo, semilla in MODOS:
        md += [f"## Modo {modo}" + (" (semilla 1)" if semilla else " (las 50 candidatas del recuperador)"), ""]
        for esc in ESCENARIOS:
            for tarea in TAREAS:
                md += [f"### {NOMBRE_ESC[esc]} · tarea {NOMBRE_TAREA[tarea]}", "",
                       "| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |", "|---|---|---|---|---|"]
                for s in sistemas:
                    if esc == "frio" and s in ESTRUCTURALES:
                        md.append(f"| {nombres[s]} | no aplica | no aplica | no aplica | no aplica |")
                        salida["tabla"].append({"sistema": s, "modo": modo, "escenario": esc, "semilla": semilla,
                                                "tarea": tarea, "no_aplica": True})
                        continue
                    if (s, modo) in no_ejecutados or (s, modo, esc) in no_ejecutados:
                        md.append(f"| {nombres[s]} | no ejecutado | no ejecutado | no ejecutado | no ejecutado |")
                        salida["tabla"].append({"sistema": s, "modo": modo, "escenario": esc, "semilla": semilla,
                                                "tarea": tarea, "no_ejecutado": True})
                        continue
                    r = filas[(s, modo, esc, semilla, tarea)]
                    assert r["n_consultas"] == 1800, (s, modo, esc, tarea)
                    md.append(f"| {nombres[s]} | " + " | ".join(fmt(r[m]) for m in METRICAS) + " |")
                    salida["tabla"].append({"sistema": s, "modo": modo, "escenario": esc, "semilla": semilla,
                                            "tarea": tarea, "n_consultas": r["n_consultas"],
                                            **{m: r[m] for m in METRICAS}})
                md.append("")

    # Diferencias pareadas, modo recuperador, sobre las 1800
    md += ["## Diferencias pareadas, modo recuperador", "",
           "Diferencia media A − B por consulta [IC 95 % por bootstrap pareado, 1000 remuestreos], "
           "sobre las 1800 consultas. Con * las que no incluyen el 0.", ""]
    for tarea in TAREAS:
        md += [f"### Tarea {NOMBRE_TAREA[tarea]}", "",
               "| Comparación | Escenario | precision@1 | precision_norm@5 | MRR | hits@5 |",
               "|---|---|---|---|---|---|"]
        for a, b, esc in comparaciones:
            ra, rb = cargar_metricas(a, "recuperador", esc, None), cargar_metricas(b, "recuperador", esc, None)
            difs = {m: diferencia_pareada(ra, rb, tarea, m, indices=sin_dev[esc]) for m in METRICAS}
            assert all(d["n"] == 1800 for d in difs.values())
            md.append(f"| {nombres[a]} − {nombres[b]} | {NOMBRE_ESC[esc]} | "
                      + " | ".join(fmt_dif(difs[m]) for m in METRICAS) + " |")
            salida["diferencias"].append({"a": a, "b": b, "escenario": esc, "modo": "recuperador",
                                          "tarea": tarea, **difs})
        md.append("")

    if exploratorio:
        md += seccion_exploratoria(exploratorio, nombres, sin_dev, salida)

    (DIR_RESULTADOS / f"{base}.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (DIR_RESULTADOS / f"{base}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Escritos {base}.md y {base}.json")


def seccion_exploratoria(ex, nombres, sin_dev, salida):
    """Líneas del .md de la sección exploratoria; añade sus filas a salida["exploratorio_a_posteriori"]."""
    filas = cargar_filas((ex["resumen"],))
    nombres = {**nombres, **ex["nombres"]}
    datos = salida["exploratorio_a_posteriori"] = {"nota": ex["nota"], "tabla": [], "diferencias": []}
    md = ["## Análisis exploratorio (a posteriori)", "", ex["nota"], ""]
    for esc in ex["escenarios"]:
        for tarea in TAREAS:
            md += [f"### {NOMBRE_ESC[esc]} · tarea {NOMBRE_TAREA[tarea]} (modo recuperador)", "",
                   "| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |", "|---|---|---|---|---|"]
            for s in ex["sistemas"]:
                r = filas[(s, "recuperador", esc, None, tarea)]
                assert r["n_consultas"] == 1800, (s, esc, tarea)
                md.append(f"| {nombres[s]} | " + " | ".join(fmt(r[m]) for m in METRICAS) + " |")
                datos["tabla"].append({"sistema": s, "modo": "recuperador", "escenario": esc, "semilla": None,
                                       "tarea": tarea, "n_consultas": r["n_consultas"],
                                       **{m: r[m] for m in METRICAS}})
            md.append("")
    md += ["### Diferencias pareadas (modo recuperador)", "",
           "Diferencia media A − B por consulta [IC 95 % por bootstrap pareado, 1000 remuestreos], "
           "sobre las 1800 consultas. Con * las que no incluyen el 0.", "",
           "| Tarea | Comparación | Escenario | precision@1 | precision_norm@5 | MRR | hits@5 |",
           "|---|---|---|---|---|---|---|"]
    for tarea in TAREAS:
        for a, b, esc in ex["comparaciones"]:
            ra, rb = cargar_metricas(a, "recuperador", esc, None), cargar_metricas(b, "recuperador", esc, None)
            difs = {m: diferencia_pareada(ra, rb, tarea, m, indices=sin_dev[esc]) for m in METRICAS}
            assert all(d["n"] == 1800 for d in difs.values())
            md.append(f"| {NOMBRE_TAREA[tarea]} | {nombres[a]} − {nombres[b]} | {NOMBRE_ESC[esc]} | "
                      + " | ".join(fmt_dif(difs[m]) for m in METRICAS) + " |")
            datos["diferencias"].append({"a": a, "b": b, "escenario": esc, "modo": "recuperador",
                                         "tarea": tarea, **difs})
    md.append("")
    return md


def main():
    generar("tabla_baselines")


if __name__ == "__main__":
    main()
