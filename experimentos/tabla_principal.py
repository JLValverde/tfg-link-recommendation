"""Tabla principal (T2.3, T2.5, T2.7, T2.7b, T2.7c): la tabla de baselines más C1, C2 y las réplicas.

Las mismas filas, tareas, métricas y diferencias que experimentos/tabla_baselines.py,
más, solo en modo recuperador (en modo aleatorio, «no ejecutado»):
- C1 (qwen2.5:3b, prompt congelado), con C1 - C0 y C1 - BM25 (los dos escenarios) y
  C1 - Adamic-Adar (B) y C1 - fusión (caliente);
- C2 (qwen2.5:3b, contexto congelado), con C2 - C1, C2 - C0 y C2 - BM25 (los dos
  escenarios) y C2 - Adamic-Adar (B) y C2 - fusión (caliente);
- cada réplica de C2 elegida en T2.6 (c2_7b, c2_14b), con su nivel de presupuesto
  (presupuesto_niveles.json), con réplica - C2 (3B), réplica - C1, réplica - C0 y
  réplica - BM25 en los escenarios que tenga, y réplica - Adamic-Adar (B) y réplica -
  fusión en caliente. Un escenario sin ejecutar sale como «no ejecutado»;
- C1 con el 14B (T2.7b; c1_14b), con su nivel, con C2 (14B) - C1 (14B), C1 (14B) - C1 (3B),
  C1 (14B) - C0 y C1 (14B) - BM25 en los escenarios que tenga, y C1 (14B) - Adamic-Adar (B)
  y C1 (14B) - fusión en caliente;
- C1 con el 7B (T2.7c; c1_7b), con su nivel, con C2 (7B) - C1 (7B), C1 (7B) - C1 (3B),
  C1 (7B) - C0, C1 (7B) - BM25 y C1 (14B) - C1 (7B) en los escenarios que tenga, y
  C1 (7B) - Adamic-Adar (B) y C1 (7B) - fusión en caliente (la de la fusión, desde T2.8e).
C2 y las réplicas solo entran si están en c2_resumen.json, y C1 (7B) y C1 (14B), si están
en c1_resumen.json.

Al final, aparte, el análisis exploratorio (a posteriori) de T2.8b, si existe
fusion_exploratoria_resumen.json: la fusión RRF de C1 con cada modelo y Adamic-Adar (B),
en caliente, con sus diferencias frente a Adamic-Adar (B), C1 con el mismo modelo y la
fusión de BM25 y Adamic-Adar (B). Va en su propia sección del .md y en la clave
"exploratorio_a_posteriori" del .json, nunca con los sistemas del protocolo.

No puntúa nada: lee los *_resumen.json y los metricas_*.json.
Escribe experimentos/resultados/tabla_principal.md y tabla_principal.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/tabla_principal.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabla_baselines as tb  # noqa: E402

ESCENARIOS = tb.ESCENARIOS
RUTA_NIVELES = tb.DIR_RESULTADOS / "presupuesto_niveles.json"
MODELO_DE = {"c2": "qwen2.5:3b", "c2_7b": "qwen2.5:7b", "c2_14b": "qwen2.5:14b", "c1_7b": "qwen2.5:7b",
             "c1_14b": "qwen2.5:14b"}
REPLICAS_C1 = ("c1_7b", "c1_14b")
# Con qué se compara cada réplica de C1 en caliente, además de lo de los dos escenarios
CALIENTE_C1 = {"c1_7b": ("adamic_adar_B", "rrf_bm25_aa_B"), "c1_14b": ("adamic_adar_B", "rrf_bm25_aa_B")}
C1_DE_EXPLORATORIO = {"rrf_c1_3b_aa_B": "c1", "rrf_c1_7b_aa_B": "c1_7b", "rrf_c1_14b_aa_B": "c1_14b"}


def escenarios(resumen, sistemas):
    """{sistema: escenarios con filas en el resumen}, solo de los sistemas pedidos."""
    ruta = tb.DIR_RESULTADOS / resumen
    if not ruta.exists():
        return {}
    presentes = {}
    for r in json.loads(ruta.read_text(encoding="utf-8")):
        presentes.setdefault(r["sistema"], set()).add(r["escenario"])
    return {s: presentes[s] for s in sistemas if s in presentes}


def escenarios_c2():
    """{sistema: escenarios con filas en c2_resumen.json}."""
    return escenarios("c2_resumen.json", [s for s in MODELO_DE if s.startswith("c2")])


def nombre_c2(s, niveles):
    if s == "c2":
        return "C2 (qwen2.5:3b)"
    tag = s.split("_", 1)[1]
    lista = niveles.get("replicas_t27", {}).get(tag, {}).get("niveles")
    nombre = s.split("_")[0].upper()
    return f"{nombre} ({MODELO_DE[s]}; nivel {', '.join(lista)})" if lista else f"{nombre} ({MODELO_DE[s]})"


def main():
    niveles = json.loads(RUTA_NIVELES.read_text(encoding="utf-8")) if RUTA_NIVELES.exists() else {}
    c2 = escenarios_c2()
    c1r = escenarios("c1_resumen.json", REPLICAS_C1)
    sistemas = tb.SISTEMAS + ("c1",) + tuple(c2) + tuple(c1r)
    nombres = {**tb.NOMBRE_SIS, "c1": "C1 (qwen2.5:3b)", **{s: nombre_c2(s, niveles) for s in (*c2, *c1r)}}
    resumenes = tb.RESUMENES + ("c1_resumen.json",) + (("c2_resumen.json",) if c2 else ())
    no_ejecutados = {(s, "aleatorio") for s in ("c1", *c2, *c1r)}
    no_ejecutados |= {(s, "recuperador", e) for s, escs in (*c2.items(), *c1r.items())
                      for e in ESCENARIOS if e not in escs}
    comparaciones = (tb.COMPARACIONES
                     + [("c1", b, esc) for esc in ESCENARIOS for b in ("c0", "bm25")]
                     + [("c1", b, "caliente") for b in ("adamic_adar_B", "rrf_bm25_aa_B")])
    if "c2" in c2:
        comparaciones += [("c2", b, esc) for esc in ESCENARIOS if esc in c2["c2"] for b in ("c1", "c0", "bm25")]
        comparaciones += [("c2", b, "caliente") for b in ("adamic_adar_B", "rrf_bm25_aa_B") if "caliente" in c2["c2"]]
    for s in (s for s in c2 if s != "c2"):
        comparaciones += [(s, b, esc) for esc in ESCENARIOS if esc in c2[s] for b in ("c2", "c1", "c0", "bm25")]
        comparaciones += [(s, b, "caliente") for b in ("adamic_adar_B", "rrf_bm25_aa_B") if "caliente" in c2[s]]
    for s in c1r:
        par = s.replace("c1_", "c2_")
        comparaciones += [(par, s, esc) for esc in ESCENARIOS if esc in c1r[s] and esc in c2.get(par, ())]
        comparaciones += [(s, b, esc) for esc in ESCENARIOS if esc in c1r[s] for b in ("c1", "c0", "bm25")]
        comparaciones += [(s, b, "caliente") for b in CALIENTE_C1[s] if "caliente" in c1r[s]]
    if "c1_7b" in c1r and "c1_14b" in c1r:
        comparaciones += [("c1_14b", "c1_7b", esc) for esc in ESCENARIOS if esc in c1r["c1_7b"] and esc in c1r["c1_14b"]]
    replicas_c1 = {"c1_7b": "C1 (7B) (T2.7c) repite C1 con el modelo del nivel medio",
                   "c1_14b": "C1 (14B) (T2.7b) repite C1 con el modelo del nivel alto"}
    nota = ("C1: qwen2.5:3b con el prompt congelado en `pipeline/prompts/c1.txt` (T2.2). "
            "C2: el prompt de C1 más el contexto estructural congelado en `pipeline/prompts/c2.json` (T2.4); "
            "las réplicas (T2.7) repiten C2 con otro modelo, con el nivel de presupuesto de T2.6"
            + (", y " + " y ".join(replicas_c1[s] for s in c1r) + ". " if c1r else ". ")
            + "Solo en modo recuperador; en modo aleatorio, «no ejecutado».")
    tb.generar("tabla_principal", titulo="Tabla principal (baselines, C1, C2 y réplicas)", sistemas=sistemas,
               nombres=nombres, resumenes=resumenes, comparaciones=comparaciones,
               no_ejecutados=no_ejecutados, nota=nota, exploratorio=exploratorio(nombres))


def exploratorio(nombres):
    """La sección exploratoria (T2.8b) para tb.generar, o None si no hay fusión exploratoria."""
    presentes = escenarios("fusion_exploratoria_resumen.json", list(C1_DE_EXPLORATORIO))
    sistemas = [s for s in C1_DE_EXPLORATORIO if s in presentes and C1_DE_EXPLORATORIO[s] in nombres]
    if not sistemas:
        return None
    nombres_ex = {s: f"fusión RRF {nombres[C1_DE_EXPLORATORIO[s]].split(';')[0].rstrip(')')})"
                     f" + Adamic-Adar (B)" for s in sistemas}
    return {"sistemas": sistemas, "nombres": nombres_ex, "resumen": "fusion_exploratoria_resumen.json",
            "escenarios": ("caliente",),
            "comparaciones": [(s, b, "caliente") for s in sistemas
                              for b in ("adamic_adar_B", C1_DE_EXPLORATORIO[s], "rrf_bm25_aa_B")],
            "nota": ("**No son sistemas del protocolo.** Se añaden en T2.8b, por decisión de Jose, después de "
                     "ver los resultados de C1 y de los baselines. Fusión RRF (k = 60) de C1 con cada modelo y "
                     "Adamic-Adar (B), como la de BM25 y Adamic-Adar (B): rangos del arnés, desempate fijo y lista "
                     "entera. Se calcula desde las puntuaciones guardadas de C1, sin volver a llamar al modelo "
                     "(`experimentos/evaluar_fusion_exploratoria.py`). Solo caliente y modo recuperador.")}


if __name__ == "__main__":
    main()
