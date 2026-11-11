"""Tablas finales de T2.8 (protocolo §6), en markdown, en experimentos/resultados/.

No puntúa nada: lee tabla_principal.json, los *_analisis.json, los agregados de coste,
los ficheros de VRAM y el techo del recuperador (experimentos/figuras/datos.py).
Todas sobre las 1800 consultas de la muestra principal que no son de desarrollo.

- tabla_resumen.md: una fila por sistema; por escenario, precision@1, precision_norm@5,
  MRR y hits@5 (tarea implícita, con IC), latencia p50 por consulta y, en los sistemas con
  modelo de lenguaje, nivel de presupuesto y picos de VRAM. Al final, aparte, el análisis
  exploratorio (a posteriori) de T2.8b, si está en tabla_principal.json.
- tabla_ablacion.md: las diferencias pareadas de fig_8_ablacion, por bloques (efecto de la
  reordenación, del contexto y del tamaño), MRR y precision@1.
- tabla_ablacion_subconjunto.md: el análisis secundario de la ablación (decisión del
  2026-10-28, protocolo §6): C1 − C0 y C2 − C1 con cada modelo, C1 (14B) − C1 (3B) y
  C1 (14B) − C1 (7B), solo sobre las consultas cuya lista trae algún enlace de referencia de
  la tarea (con los filtros de arnes/rangos.py), cada una debajo de su cifra sobre las 1800
  y con el n del subconjunto; por escenario, las tres tareas, precision@1, precision@5,
  precision_norm@5 y MRR. Lee los metricas_*.json y las listas del recuperador
  (datos/procesados/).
- tabla_techo.md: recall@K del recuperador por escenario y tarea, y % de listas sin
  ningún enlace de referencia entre las K = 50 candidatas.
- tabla_A_vs_B.md: vecinos comunes y Adamic-Adar sobre A junto a los mismos sobre B, y
  sus diferencias A − B (caliente; en frío no aplican).
- tabla_malformadas.md: por modelo y configuración, casos (i)/(ii)/(iii), None,
  respuestas que no son yes ni no y respuestas "Yes".
- tabla_secundaria.md: el modo aleatorio (semilla 1), tarea implícita; C1, C2 y las
  réplicas, «no ejecutado (recortado)».

Uso (desde la raíz del repo, después de experimentos/tabla_principal.py):
    .venv/Scripts/python.exe experimentos/tablas_finales.py
"""

import json
import sys
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "figuras"))
import datos  # noqa: E402
from fig_8_ablacion import BLOQUES  # noqa: E402

# Al final: datos/ de la raíz no debe tapar a figuras/datos.py
sys.path.append(str(datos.RAIZ))
from arnes.candidatos import cargar_candidatos, cargar_consultas  # noqa: E402
from arnes.evaluar import diferencia_pareada  # noqa: E402
from arnes.rangos import rankings_por_tarea  # noqa: E402

DIR = datos.DIR_RESULTADOS
CONTROLES = ("aleatorio", "oraculo")
BASELINES = ("c0", "bm25", "vecinos_comunes_B", "adamic_adar_B", "rrf_bm25_aa_B")
CON_MODELO = ("c1", "c1_7b", "c1_14b", "c2", "c2_7b", "c2_14b")
NOMBRE = {"aleatorio": "aleatorio (control)", "oraculo": "oráculo (control)", "c0": "C0", "bm25": "BM25",
          "vecinos_comunes_B": "vecinos comunes (B)", "adamic_adar_B": "Adamic-Adar (B)",
          "vecinos_comunes_A": "vecinos comunes (A)", "adamic_adar_A": "Adamic-Adar (A)",
          "rrf_bm25_aa_B": "fusión RRF BM25 + Adamic-Adar (B)", "c1": "C1 (qwen2.5:3b)",
          "c1_7b": "C1 (qwen2.5:7b)", "c1_14b": "C1 (qwen2.5:14b)", "c2": "C2 (qwen2.5:3b)", "c2_7b": "C2 (qwen2.5:7b)",
          "c2_14b": "C2 (qwen2.5:14b)"}
TAREAS_TABLA = datos.TAREAS
TAREA = "implicita"
K_USADO = "50"
# Análisis secundario de la ablación (decisión del 2026-10-28)
COMPARACIONES_SECUNDARIO = (("c1", "c0", "C1 − C0 (3B)"), ("c1_7b", "c0", "C1 − C0 (7B)"),
                            ("c1_14b", "c0", "C1 − C0 (14B)"), ("c2", "c1", "C2 − C1 (3B)"),
                            ("c2_7b", "c1_7b", "C2 − C1 (7B)"), ("c2_14b", "c1_14b", "C2 − C1 (14B)"),
                            ("c1_14b", "c1", "C1 (14B) − C1 (3B)"), ("c1_14b", "c1_7b", "C1 (14B) − C1 (7B)"))
TAREAS_SECUNDARIO = datos.TAREAS
METRICAS_SECUNDARIO = ("precision@1", "precision@5", "precision_norm@5", "mrr")


def n(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def fmt(v):
    return f"{n(v['media'])} [{n(v['ic95'][0])}, {n(v['ic95'][1])}]"


def fmt_dif(d):
    s = lambda x: f"{x:+.3f}".replace(".", ",").replace("-", "−")  # noqa: E731
    marca = "" if d["ic95"][0] <= 0 <= d["ic95"][1] else " *"
    return f"{s(d['diferencia'])} [{s(d['ic95'][0])}, {s(d['ic95'][1])}]{marca}"


def fmt_ms(ms):
    if ms is None:
        return "—"
    return f"{n(ms / 1000, 2)} s" if ms >= 1000 else f"{n(ms, 3 if ms < 1 else 1)} ms"


def gb(mib):
    return n(mib / 1000, 1)


def celdas(s, esc, metricas=datos.METRICAS, modo="recuperador"):
    e = datos.estado(s, esc, modo)
    if e != "ok":
        return [e or "no ejecutado"] * len(metricas)
    return [fmt(datos.metrica(s, esc, m, TAREA, modo)) for m in metricas]


def escribir(nombre, lineas):
    (DIR / nombre).write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Escrito {nombre}")


def cabecera(titulo, texto):
    return [f"# {titulo}", "", texto + " Generada por `experimentos/tablas_finales.py`.", ""]


def vram(s):
    """(nivel, tope_mib, pico de la ejecución, pico de T2.6 con el modelo de embeddings), en MiB."""
    a = datos.analisis(s)
    nivel = datos.nivel_de(s)
    tope = next(t for nv, t, _ in datos.niveles() if nv == nivel)
    if "pico_ejecucion_mib" in a:
        return nivel, tope, max(a["pico_ejecucion_mib"].values()), a["pico_t26_con_embeddings_mib"]
    # C1 (3B), T2.3: un solo fichero de VRAM para los dos escenarios, y el pico de T2.6 del 3B
    t26 = datos.leer(f"vram_presupuesto_{datos.MODELO[s]}.json")["pico_mib"]
    return nivel, tope, a["vram"]["pico_mib"], t26


def tabla_resumen():
    lineas = cabecera(
        "Tabla resumen (modo recuperador, tarea implícita)",
        "Media [IC 95 %] sobre las 1800 consultas de la muestra principal que no son de desarrollo, "
        "y latencia p50 por consulta (en los sistemas con modelo de lenguaje, sin las 3 consultas de "
        "calentamiento y sobre las 1800; en los demás, lo medido: las 2000 menos las 3 de calentamiento).")
    lineas += [
        "VRAM: nivel de presupuesto (tope) y, en GB, el pico de la propia ejecución (el mayor de sus "
        "escenarios) / el pico de T2.6 con el modelo de embeddings cargado a la vez, que es el que se "
        "comprueba contra el tope. Los sistemas sin modelo de lenguaje no tienen nivel. "
        "Las tres tareas y todas las diferencias, en `tabla_principal.md`.", ""]
    esc_cab = []
    for esc in datos.ESCENARIOS:
        e = datos.NOMBRE_ESC[esc]
        esc_cab += [f"{e}: {datos.NOMBRE_METRICA[m]}" for m in datos.METRICAS] + [f"{e}: latencia p50"]
    lineas += ["| Sistema | VRAM: nivel (tope): pico / pico T2.6 | " + " | ".join(esc_cab) + " |",
               "|---|---|" + "---|" * len(esc_cab)]
    for s in CONTROLES + BASELINES + CON_MODELO:
        if datos.estado(s, "frio") is None:
            continue
        if s in CON_MODELO:
            nivel, tope, pico, t26 = vram(s)
            col_vram = f"{nivel} ({gb(tope)} GB): {gb(pico)} / {gb(t26)} GB"
        else:
            col_vram = "—"
        fila = []
        for esc in datos.ESCENARIOS:
            fila += celdas(s, esc)
            fila.append("—" if s in CONTROLES or not datos.disponible(s, esc)
                        else fmt_ms(datos.latencia_p50_ms(s, esc)))
        lineas.append(f"| {NOMBRE[s]} | {col_vram} | " + " | ".join(fila) + " |")
    lineas += exploratorio_resumen()
    escribir("tabla_resumen.md", lineas)


def exploratorio_resumen():
    """Sección aparte con la fusión exploratoria de T2.8b (solo caliente), o nada si no está."""
    sistemas, _ = datos.exploratorio()
    if not sistemas:
        return []
    nota = datos.leer("tabla_principal.json")["exploratorio_a_posteriori"]["nota"]
    lineas = ["", "## Análisis exploratorio (a posteriori)", "", nota + " Sin latencia propia: no se ha medido "
              "(su coste es el de C1 con ese modelo más el de Adamic-Adar (B)).", "",
              "| Sistema | " + " | ".join(f"caliente: {datos.NOMBRE_METRICA[m]}" for m in datos.METRICAS) + " |",
              "|---|" + "---|" * len(datos.METRICAS)]
    for s in sistemas:
        modelo = s.split("_")[2]
        lineas.append(f"| fusión RRF C1 (qwen2.5:{modelo}) + Adamic-Adar (B) | " + " | ".join(
            fmt(datos.metrica_exploratoria(s, "caliente", m)) for m in datos.METRICAS) + " |")
    return lineas


def tabla_ablacion():
    lineas = cabecera(
        "Ablación: diferencias pareadas (modo recuperador, tarea implícita)",
        "Diferencia media A − B por consulta [IC 95 % por bootstrap pareado, 1000 remuestreos], sobre "
        "las 1800 consultas. Con * las que no incluyen el 0. Son las de `fig_8_ablacion`, con sus bloques; "
        "las tres tareas y las cuatro métricas, en `tabla_principal.md`.")
    lineas += ["| Bloque | Comparación | Escenario | MRR | precision@1 |", "|---|---|---|---|---|"]
    for bloque, comparaciones in BLOQUES:
        for a, b, nombre in comparaciones:
            for esc in datos.ESCENARIOS:
                if datos.diferencia(a, b, esc) is None:
                    continue
                lineas.append(f"| {bloque} | {nombre} | {datos.NOMBRE_ESC[esc]} | "
                              f"{fmt_dif(datos.diferencia(a, b, esc, 'mrr'))} | "
                              f"{fmt_dif(datos.diferencia(a, b, esc, 'precision@1'))} |")
    lineas += ["", "El análisis secundario (solo las consultas con algún enlace de referencia entre las candidatas) "
               "está en `tabla_ablacion_subconjunto.md`, siempre junto a estas cifras."]
    escribir("tabla_ablacion.md", lineas)


def con_referencia_en_lista(esc):
    """{tarea: [idx]} de las 1800 consultas cuya lista del recuperador trae algún enlace de referencia.

    Con los filtros de arnes/rangos.py: sin neutros y, en la tarea implícita, sin explícitos.
    Solo depende de la lista, así que es el mismo subconjunto para todos los sistemas.
    """
    candidatos = cargar_candidatos("recuperador", esc)
    salida = {t: [] for t in TAREAS_SECUNDARIO}
    for c in cargar_consultas(esc):
        if c["es_desarrollo"]:
            continue
        for t, (ranking, relevantes) in rankings_por_tarea(c, dict.fromkeys(candidatos[c["idx"]])).items():
            if t in salida and relevantes & set(ranking):
                salida[t].append(c["idx"])
    # Mismo recuento que la última columna de tabla_techo.md (recall@50 > 0)
    por_consulta = [json.loads(l) for l in (DIR / "recall_recuperador_por_consulta.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
    for t, idx in salida.items():
        assert set(idx) == {f["idx"] for f in por_consulta if f["escenario"] == esc and not f["es_desarrollo"]
                            and f["recall"][t][K_USADO] > 0}, (esc, t)
    return salida


@cache
def metricas_recuperador(s, esc):
    """metricas_{s}_recuperador_{esc}.json, o None si el sistema aún no tiene puntuaciones."""
    ruta = DIR / f"metricas_{s}_recuperador_{esc}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def tabla_ablacion_subconjunto():
    """Análisis secundario de la ablación sobre las consultas con algún enlace de referencia en la lista."""
    lineas = cabecera(
        "Ablación: análisis secundario sobre las consultas con algún enlace de referencia entre las candidatas",
        "**Secundario: la cifra que se reporta es la de las 1800 consultas.** Las diferencias pareadas de la "
        "ablación (C1 − C0 y C2 − C1 con cada modelo, y C1 (14B) − C1 (3B) y C1 (14B) − C1 (7B)), solo sobre "
        f"las consultas cuya lista de K = {K_USADO} candidatas del recuperador trae al menos un enlace de "
        "referencia de la tarea, después de los filtros de `arnes/rangos.py` (sin neutros y, en las tareas "
        "implícitas, sin los explícitos de cada definición). Cada fila va debajo de su cifra sobre las 1800. "
        "Modo recuperador; mismo bootstrap pareado (1000 remuestreos, IC 95 %), con * las que no incluyen el 0. "
        "Se calcula desde las puntuaciones guardadas, sin llamar al modelo (decisión del 2026-10-28, "
        "protocolo §6). **El subconjunto lo elige el recuperador y no tiene el mismo tamaño en cada escenario: "
        "sirve solo para comparar configuraciones dentro de un escenario, no escenarios entre sí.**")
    lineas += ["| Escenario | Tarea | Consultas con algún enlace de referencia en la lista |", "|---|---|---|"]
    subconjuntos = {esc: con_referencia_en_lista(esc) for esc in datos.ESCENARIOS}
    for esc in datos.ESCENARIOS:
        for t in TAREAS_SECUNDARIO:
            k = len(subconjuntos[esc][t])
            lineas.append(f"| {datos.NOMBRE_ESC[esc]} | {datos.NOMBRE_TAREA[t]} | {k} de 1800 "
                          f"({n(100 * k / 1800, 1)} %) |")
    for esc in datos.ESCENARIOS:
        sin_dev = [c["idx"] for c in cargar_consultas(esc) if not c["es_desarrollo"]]
        lineas += ["", f"## {datos.NOMBRE_ESC[esc].capitalize()} (secundario)", "",
                   "| Tarea | Comparación | Consultas | " + " | ".join(
                       datos.NOMBRE_METRICA.get(m, m) for m in METRICAS_SECUNDARIO) + " |",
                   "|---|---|---|" + "---|" * len(METRICAS_SECUNDARIO)]
        for t in TAREAS_SECUNDARIO:
            for a, b, nombre in COMPARACIONES_SECUNDARIO:
                ra, rb = metricas_recuperador(a, esc), metricas_recuperador(b, esc)
                if ra is None or rb is None:
                    continue
                for etiqueta, indices in (("todas", sin_dev), ("con enlace de referencia", subconjuntos[esc][t])):
                    difs = [diferencia_pareada(ra, rb, t, m, indices=indices) for m in METRICAS_SECUNDARIO]
                    assert all(d["n"] == len(indices) for d in difs), (a, b, esc, t)
                    primera = etiqueta == "todas"
                    lineas.append(f"| {datos.NOMBRE_TAREA[t] if primera else ''} | {nombre if primera else ''} | "
                                  f"{etiqueta} ({len(indices)}) | " + " | ".join(fmt_dif(d) for d in difs) + " |")
    escribir("tabla_ablacion_subconjunto.md", lineas)


def tabla_techo():
    r = datos.leer("recall_recuperador.json")
    por_consulta = [json.loads(l) for l in (DIR / "recall_recuperador_por_consulta.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
    ks = list(r["recall"]["frio"]["implicita"])
    lineas = cabecera(
        "Techo del recuperador (recall@K)",
        f"Recall@K del recuperador ({r['modelo']}): fracción de los enlaces de referencia de cada consulta "
        f"que entra entre los K primeros, media [IC 95 %] sobre las 1800 consultas. Los sistemas puntúan "
        f"las K = {K_USADO} primeras, así que recall@{K_USADO} es el máximo que puede recuperar cualquiera. "
        f"La última columna es el % de consultas sin ningún enlace de referencia entre esas {K_USADO}.")
    lineas += ["| Escenario | Tarea | " + " | ".join(
        f"**recall@{k}**" if k == K_USADO else f"recall@{k}" for k in ks)
        + f" | Listas sin enlace de referencia (K = {K_USADO}) |", "|---|---|" + "---|" * (len(ks) + 1)]
    for esc in datos.ESCENARIOS:
        filas = [c for c in por_consulta if c["escenario"] == esc and not c["es_desarrollo"]]
        assert len(filas) == r["n_consultas"][esc] == 1800
        for t in TAREAS_TABLA:
            sin = sum(c["recall"][t][K_USADO] == 0 for c in filas)
            lineas.append(f"| {datos.NOMBRE_ESC[esc]} | {datos.NOMBRE_TAREA[t]} | "
                          + " | ".join(fmt(r["recall"][esc][t][k]) for k in ks)
                          + f" | {n(100 * sin / len(filas), 1)} % ({sin}) |")
    escribir("tabla_techo.md", lineas)


def tabla_a_vs_b():
    lineas = cabecera(
        "Baselines estructurales sobre A y sobre B (caliente)",
        "Media [IC 95 %] sobre las 1800 consultas. B son los enlaces de WikiCS escritos en el artículo "
        "de origen (de donde salen los enlaces de referencia); A, todos los enlaces de WikiCS, incluidos "
        "los de plantillas de navegación. Solo en caliente: en frío no aplican. Las diferencias A − B "
        "son pareadas, en modo recuperador.")
    for modo, titulo in (("recuperador", "Modo recuperador"), ("aleatorio", "Modo aleatorio (semilla 1)")):
        lineas += [f"## {titulo}", ""]
        for t in TAREAS_TABLA:
            lineas += [f"### Tarea {datos.NOMBRE_TAREA[t]}", "",
                       "| Sistema | " + " | ".join(datos.NOMBRE_METRICA[m] for m in datos.METRICAS) + " |",
                       "|---|" + "---|" * len(datos.METRICAS)]
            for s in ("vecinos_comunes_B", "vecinos_comunes_A", "adamic_adar_B", "adamic_adar_A"):
                lineas.append(f"| {NOMBRE[s]} | " + " | ".join(
                    fmt(datos.metrica(s, "caliente", m, t, modo)) for m in datos.METRICAS) + " |")
            lineas.append("")
    lineas += ["## Diferencias A − B, modo recuperador", "",
               "Con * las que no incluyen el 0.", "",
               "| Tarea | Comparación | " + " | ".join(datos.NOMBRE_METRICA[m] for m in datos.METRICAS) + " |",
               "|---|---|" + "---|" * len(datos.METRICAS)]
    for t in TAREAS_TABLA:
        for h in ("vecinos_comunes", "adamic_adar"):
            a, b = f"{h}_A", f"{h}_B"
            lineas.append(f"| {datos.NOMBRE_TAREA[t]} | {NOMBRE[a]} − {NOMBRE[b]} | " + " | ".join(
                fmt_dif(datos.diferencia(a, b, "caliente", m, t)) for m in datos.METRICAS) + " |")
    escribir("tabla_A_vs_B.md", lineas)


def tabla_malformadas():
    lineas = cabecera(
        "Salidas malformadas y casos del sí y el no (modo recuperador)",
        "Sobre las 1800 consultas. Una llamada por candidato. Casos de la confianza P-cota: (i) "
        "\"yes\" y \"no\" están entre los 20 primeros *tokens*; (ii) falta uno y se acota con el menor del "
        "top; (iii) faltan los dos (puntuación None). None es la peor puntuación posible.")
    cab = ["Llamadas", "Errores", "(i) ambos", "(ii) falta uno", "(iii) faltan ambos", "None (tasa)",
           "Respuestas que no son yes ni no", "Truncados", "Respuestas \"Yes\""]
    lineas += ["| Configuración | Modelo | Escenario | " + " | ".join(cab) + " |", "|---|---|---|" + "---|" * len(cab)]
    for s in CON_MODELO:
        a = datos.analisis(s)
        if a is None:
            continue
        for esc in datos.ESCENARIOS:
            v = a["por_escenario"][esc]
            mil = lambda x: f"{x:,}".replace(",", " ")  # noqa: E731
            lineas.append(
                f"| {s[:2].upper()} | {NOMBRE[s].split('(')[1].rstrip(')')} | {datos.NOMBRE_ESC[esc]} | "
                f"{mil(v['llamadas'])} | {v['errores']} | {mil(v['casos']['ambos'])} | "
                f"{mil(v['casos']['falta_uno'])} ({n(v['pct_casos']['falta_uno'], 1)} %) | "
                f"{mil(v['casos']['faltan_ambos'])} | {v['none']} ({n(100 * v['tasa_none'], 2)} %) | "
                f"{round(v['pct_no_si_no'] * v['llamadas'] / 100)} ({n(v['pct_no_si_no'], 4)} %) | "
                f"{v['truncados']} | {n(v['pct_yes'], 2)} % |")
    escribir("tabla_malformadas.md", lineas)


def tabla_secundaria():
    lineas = cabecera(
        "Tabla secundaria: modo aleatorio (semilla 1, tarea implícita)",
        "Media [IC 95 %] sobre las 1800 consultas, con las 50 candidatas del modo aleatorio (enlaces de "
        "referencia, como mucho G = 10, más negativos al azar): mide la capacidad de ordenar, aislada, y "
        "no se mezcla con el modo recuperador. Las semillas 2 y 3 están en los `*_resumen.json`; las tres "
        "tareas, en `tabla_principal.md`. C1, C2 y las réplicas no se han ejecutado en este modo (recortado).")
    esc_cab = [f"{datos.NOMBRE_ESC[e]}: {datos.NOMBRE_METRICA[m]}" for e in datos.ESCENARIOS for m in datos.METRICAS]
    lineas += ["| Sistema | " + " | ".join(esc_cab) + " |", "|---|" + "---|" * len(esc_cab)]
    for s in CONTROLES + ("c0", "bm25", "vecinos_comunes_B", "adamic_adar_B", "vecinos_comunes_A",
                          "adamic_adar_A", "rrf_bm25_aa_B") + CON_MODELO:
        if datos.estado(s, "frio", "aleatorio") is None:
            continue
        fila = [c for esc in datos.ESCENARIOS for c in celdas(s, esc, modo="aleatorio")]
        if s in CON_MODELO:
            assert set(fila) == {"no ejecutado"}, s
            fila = ["no ejecutado (recortado)"] * len(fila)
        lineas.append(f"| {NOMBRE[s]} | " + " | ".join(fila) + " |")
    escribir("tabla_secundaria.md", lineas)


def main():
    tabla_resumen()
    tabla_ablacion()
    tabla_ablacion_subconjunto()
    tabla_techo()
    tabla_a_vs_b()
    tabla_malformadas()
    tabla_secundaria()


if __name__ == "__main__":
    main()
