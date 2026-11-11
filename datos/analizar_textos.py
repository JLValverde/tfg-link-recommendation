"""Análisis de la descarga completa de textos (T0.3).

Lee datos/procesados/textos.jsonl y datos/raw/wikics/data.json y escribe
experimentos/resultados/textos_completo.json. Solo mide: no modifica el grafo
ni genera consultas.

A = links[i] sin duplicados ni autobucles.
B = A ∩ (destinos_cuerpo ∪ destinos_cuerpo_redir): enlaces de WikiCS que están
    escritos en el wikitexto del origen (directos o a través de una redirección
    actual). Los nodos sin texto (no ok) tienen B vacío.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe datos\\analizar_textos.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RUTA_TEXTOS = RAIZ / "datos" / "procesados" / "textos.jsonl"
RUTA_DATA = RAIZ / "datos" / "raw" / "wikics" / "data.json"
RUTA_META = RAIZ / "datos" / "raw" / "wikics" / "metadata.json"
RUTA_EJECUCIONES = RAIZ / "experimentos" / "resultados" / "descarga_textos_ejecuciones.jsonl"
RUTA_SALIDA = RAIZ / "experimentos" / "resultados" / "textos_completo.json"
UMBRALES_GRADO = [1, 2, 3, 5, 10, 20]


def resumen(v, claves=("min", "mediana", "media", "p90", "p99", "max")):
    a = np.asarray(v, dtype=float)
    todo = {"min": a.min(), "mediana": np.percentile(a, 50), "media": a.mean(),
            "p90": np.percentile(a, 90), "p99": np.percentile(a, 99), "max": a.max()}
    return {k: round(float(todo[k]), 4) for k in claves}


def pct(a, b):
    return round(100 * a / b, 2) if b else None


def describir_grafo(ady, titulos):
    """ady: lista de conjuntos de sucesores (sin autobucles)."""
    n = len(ady)
    aristas = {(u, v) for u in range(n) for v in ady[u]}
    reciprocos = sum((v, u) in aristas for u, v in aristas)
    sal = [len(s) for s in ady]
    ent = [0] * n
    for _, v in aristas:
        ent[v] += 1
    top = sorted(range(n), key=lambda i: (-ent[i], i))[:10]
    return {
        "enlaces": len(aristas),
        "pct_reciprocidad": pct(reciprocos, len(aristas)),
        "grado_saliente": resumen(sal),
        "grado_entrante": resumen(ent),
        "nodos_grado_saliente_0": sal.count(0),
        "nodos_grado_saliente_al_menos": {str(k): sum(g >= k for g in sal) for k in UMBRALES_GRADO},
        "top10_grado_entrante": [{"idx": i, "titulo": titulos[i], "grado_entrante": ent[i]}
                                 for i in top],
    }, sal, ent


def main():
    filas = [json.loads(l) for l in RUTA_TEXTOS.read_text(encoding="utf-8").splitlines()]
    links = json.loads(RUTA_DATA.read_text(encoding="utf-8"))["links"]
    titulos = [nd["title"] for nd in json.loads(RUTA_META.read_text(encoding="utf-8"))["nodes"]]
    n = len(links)
    assert [f["idx"] for f in filas] == list(range(n)), "textos.jsonl no cubre 0..n-1 en orden"
    res = {"n_nodos": n}

    A = [set(links[i]) - {i} for i in range(n)]
    cuerpo = [set(f["destinos_cuerpo"]) - {f["idx"]} for f in filas]
    redir = [set(f["destinos_cuerpo_redir"]) - {f["idx"]} for f in filas]
    plantilla = [set(f["destinos_solo_plantilla"]) for f in filas]
    B = [A[i] & (cuerpo[i] | redir[i]) for i in range(n)]
    ok = [f["estado"] == "ok" for f in filas]

    # c) primero, porque (a) usa los grados de A
    info_A, sal_A, ent_A = describir_grafo(A, titulos)
    info_B, sal_B, ent_B = describir_grafo(B, titulos)

    # a) Estados y centralidad de los nodos sin texto
    estados = Counter(f["estado"] for f in filas)
    res["a_estados"] = dict(sorted(estados.items()))
    res["a_no_ok"] = {}
    for estado in sorted(set(estados) - {"ok"}):
        idx = [f["idx"] for f in filas if f["estado"] == estado]
        res["a_no_ok"][estado] = {
            "n": len(idx),
            "grado_saliente_A": resumen([sal_A[i] for i in idx]),
            "grado_entrante_A": resumen([ent_A[i] for i in idx]),
            "enlaces_de_A_que_tocan_estos_nodos": sum(sal_A[i] + ent_A[i] for i in idx),
            "mayor_grado_entrante": sorted(
                [{"idx": i, "titulo": titulos[i], "entrante": ent_A[i], "saliente": sal_A[i]}
                 for i in idx], key=lambda x: (-x["entrante"], x["idx"]))[:5],
            "detalles": dict(Counter(f.get("detalle") for f in filas
                                     if f["estado"] == estado and f.get("detalle"))),
        }
    res["a_referencia_todos"] = {"grado_saliente_A": resumen(sal_A), "grado_entrante_A": resumen(ent_A)}
    ts = sorted(f["timestamp"] for f in filas if f["estado"] == "ok")
    res["a_fechas"] = {"min": ts[0], "mediana": ts[len(ts) // 2], "max": ts[-1],
                       "por_anio": dict(sorted(Counter(t[:4] for t in ts).items())),
                       "posteriores_al_corte": sum(t > "2019-08-20T00:00:00Z" for t in ts)}

    # b) Cobertura
    tot_A = sum(map(len, A))
    tot_A_ok = sum(len(A[i]) for i in range(n) if ok[i])
    tot_B = sum(map(len, B))
    solo_redir = sum(len(B[i] - cuerpo[i]) for i in range(n))
    solo_plant = sum(len(B[i] & plantilla[i]) for i in range(n))
    medias = [len(B[i]) / len(A[i]) for i in range(n) if A[i]]
    medias_ok = [len(B[i]) / len(A[i]) for i in range(n) if A[i] and ok[i]]
    extra = sum(len((cuerpo[i] | redir[i]) - A[i]) for i in range(n))
    extra_directo = sum(len(cuerpo[i] - A[i]) for i in range(n))
    res["b_cobertura"] = {
        "enlaces_A": tot_A, "enlaces_B": tot_B,
        "pct_micro_A_en_B": pct(tot_B, tot_A),
        "pct_micro_A_en_B_solo_nodos_ok": pct(tot_B, tot_A_ok),
        "pct_medio_A_en_B": round(100 * float(np.mean(medias)), 2),
        "pct_medio_A_en_B_solo_nodos_ok": round(100 * float(np.mean(medias_ok)), 2),
        "enlaces_B_solo_por_redireccion_actual": solo_redir,
        "pct_B_solo_por_redireccion_actual": pct(solo_redir, tot_B),
        "enlaces_B_solo_en_plantillas": solo_plant,
        "pct_B_solo_en_plantillas": pct(solo_plant, tot_B),
        "destinos_escritos_fuera_de_A": extra,
        "destinos_escritos_fuera_de_A_directos": extra_directo,
        "destinos_escritos_fuera_de_A_por_redireccion": extra - extra_directo,
        "pct_destinos_escritos_fuera_de_A": pct(extra, sum(len(cuerpo[i] | redir[i]) for i in range(n))),
    }

    # c) Grafos
    res["c_grafo_A"] = info_A
    res["c_grafo_B"] = info_B

    # d) Menciones en la introducción
    def menciones(grafo):
        directo = con_redir = 0
        for f in filas:
            i = f["idx"]
            d = {a["destino_idx"] for a in f["anclas_intro"]} - {None}
            r = d | ({a["destino_idx_redir"] for a in f["anclas_intro"]} - {None})
            directo += len(grafo[i] & d)
            con_redir += len(grafo[i] & r)
        tot = sum(map(len, grafo))
        return {"enlaces": tot, "con_ancla_directo": directo, "pct_directo": pct(directo, tot),
                "con_ancla_con_redireccion": con_redir, "pct_con_redireccion": pct(con_redir, tot)}
    res["d_menciones_intro"] = {"A": menciones(A), "B": menciones(B)}

    # e) Longitud de la introducción
    palabras = [f["n_palabras_intro"] for f in filas if f["estado"] == "ok"]
    res["e_longitud_intro"] = {**resumen(palabras, ("min", "mediana", "p90", "max")),
                               "menos_de_20": sum(p < 20 for p in palabras),
                               "cero": sum(p == 0 for p in palabras)}
    desal = sum(f["texto_intro"][a["inicio"]:a["fin"]] != a["texto"]
                for f in filas if f["estado"] == "ok" for a in f["anclas_intro"])
    res["e_anclas_desalineadas"] = desal

    # f) Tiempo y peticiones
    if RUTA_EJECUCIONES.exists():
        ej = [json.loads(l) for l in RUTA_EJECUCIONES.read_text(encoding="utf-8").splitlines() if l]
        res["f_ejecuciones"] = ej
        res["f_total"] = {k: round(sum(e[k] for e in ej), 1) for k in
                          ("peticiones_descarga", "peticiones_redirecciones", "peticiones_total",
                           "segundos_red", "segundos_total")}

    RUTA_SALIDA.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Escrito {RUTA_SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
