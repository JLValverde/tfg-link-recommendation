"""Medición de los tipos de enlace para la decisión de T0.5.

Lee datos/procesados/textos.jsonl y datos/raw/wikics/ y escribe
experimentos/resultados/tipos_enlace.json. Solo mide: no construye el grafo
limpio ni genera consultas. Determinista (sin aleatoriedad, salida ordenada).

Nodos válidos: estado == ok. Se excluyen los demás y todos los enlaces que los
tocan.
A = links[i] sin duplicados ni autobucles, entre nodos válidos.
B = A ∩ (destinos_cuerpo ∪ destinos_cuerpo_redir).
Tipos (precedencia: el de menor número):
  1. Mencionado: el destino tiene un ancla en la introducción del origen
     (destino_idx o destino_idx_redir).
  2. Escrito fuera de la introducción: en B, no es 1 y no está en
     destinos_solo_plantilla.
  3. De infobox: en B y en destinos_solo_plantilla (y no es 1).
  4. De plantilla compartida: en A y no en B.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe datos\\medir_tipos_enlace.py
"""

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RUTA_TEXTOS = RAIZ / "datos" / "procesados" / "textos.jsonl"
RUTA_DATA = RAIZ / "datos" / "raw" / "wikics" / "data.json"
RUTA_SALIDA = RAIZ / "experimentos" / "resultados" / "tipos_enlace.json"
UMBRALES_M = [1, 2, 3, 5, 10]
UMBRALES_PALABRAS = [0, 20, 30, 50]


def pct(a, b):
    return round(100 * a / b, 2) if b else None


def distribucion(v):
    a = np.asarray(v, dtype=float)
    return {"mediana": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": int(a.max()),
            "media": round(float(a.mean()), 4)}


def tabla_elegibilidad(conteo, palabras, validos):
    """Nodos válidos con al menos m enlaces del conteo dado y al menos p palabras de introducción."""
    return {f"intro>={p}": {str(m): sum(1 for i in validos if conteo[i] >= m and palabras[i] >= p)
                            for m in UMBRALES_M}
            for p in UMBRALES_PALABRAS}


def main():
    filas = [json.loads(l) for l in RUTA_TEXTOS.read_text(encoding="utf-8").splitlines()]
    links = json.loads(RUTA_DATA.read_text(encoding="utf-8"))["links"]
    n = len(links)
    assert [f["idx"] for f in filas] == list(range(n))
    valido = [f["estado"] == "ok" for f in filas]
    validos = [i for i in range(n) if valido[i]]
    titulos = [f["titulo"] for f in filas]
    palabras = [f["n_palabras_intro"] or 0 for f in filas]

    A = [{j for j in set(links[i]) if j != i and valido[j]} if valido[i] else set()
         for i in range(n)]
    escritos = [set(f["destinos_cuerpo"]) | set(f["destinos_cuerpo_redir"]) for f in filas]
    B = [A[i] & escritos[i] for i in range(n)]
    solo_plantilla = [set(f["destinos_solo_plantilla"]) for f in filas]
    anclados = [({a["destino_idx"] for a in f["anclas_intro"]}
                 | {a["destino_idx_redir"] for a in f["anclas_intro"]}) - {None, f["idx"]}
                for f in filas]

    t1 = [A[i] & anclados[i] for i in range(n)]
    t2 = [B[i] - t1[i] - solo_plantilla[i] for i in range(n)]
    t3 = [(B[i] & solo_plantilla[i]) - t1[i] for i in range(n)]
    t4 = [A[i] - B[i] for i in range(n)]

    tot = {k: sum(map(len, t)) for k, t in (("1", t1), ("2", t2), ("3", t3), ("4", t4))}
    tot_A, tot_B = sum(map(len, A)), sum(map(len, B))
    t1_fuera_de_B = sum(len(t1[i] - B[i]) for i in range(n))
    cuadra_B = tot["1"] + tot["2"] + tot["3"] == tot_B and t1_fuera_de_B == 0
    cuadra_A = sum(tot.values()) == tot_A
    res = {
        "n_nodos": n, "n_validos": len(validos), "n_excluidos": n - len(validos),
        "a_totales": {
            "enlaces_A": tot_A, "enlaces_B": tot_B,
            "enlaces_A_originales": sum(len(set(links[i]) - {i}) for i in range(n)),
            "tipos": tot,
            "pct_sobre_B": {k: pct(tot[k], tot_B) for k in ("1", "2", "3")},
            "pct_sobre_A": {k: pct(v, tot_A) for k, v in tot.items()},
            "tipo1_fuera_de_B": t1_fuera_de_B,
            "comprobacion_1+2+3=B": cuadra_B, "comprobacion_1+2+3+4=A": cuadra_A,
        },
    }
    if not (cuadra_B and cuadra_A):
        RUTA_SALIDA.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"Los totales no cuadran: {res['a_totales']}")

    # b) Elegibilidad
    c23 = [len(t2[i]) + len(t3[i]) for i in range(n)]
    c123 = [len(t1[i]) + c23[i] for i in range(n)]
    res["b_elegibilidad"] = {
        "tipos_2+3": tabla_elegibilidad(c23, palabras, validos),
        "tipos_1+2+3": tabla_elegibilidad(c123, palabras, validos),
    }

    # c) Fuga inversa: nº de OTROS nodos válidos con un ancla en su introducción hacia cada nodo
    entrante_t1 = [0] * n
    entrante_anclas = [0] * n
    for i in validos:
        for j in t1[i]:
            entrante_t1[j] += 1
        for j in anclados[i]:
            if valido[j]:
                entrante_anclas[j] += 1
    v = [entrante_t1[j] for j in validos]
    top = sorted(validos, key=lambda j: (-entrante_t1[j], j))[:10]
    res["c_fuga_inversa"] = {
        "nota": "grado entrante de tipo 1 sobre los nodos válidos",
        "distribucion": distribucion(v),
        "pct_nodos_con_al_menos_1": pct(sum(x >= 1 for x in v), len(v)),
        "nodos_con_al_menos_1": sum(x >= 1 for x in v),
        "top10": [{"idx": j, "titulo": titulos[j], "entrante_tipo1": entrante_t1[j],
                   "entrante_A": sum(1 for i in validos if j in A[i])} for j in top],
        "comparacion_contando_todas_las_anclas": {
            "nota": "anclas de la introducción hacia el nodo aunque el enlace no esté en A",
            "distribucion": distribucion([entrante_anclas[j] for j in validos]),
            "pct_nodos_con_al_menos_1": pct(sum(entrante_anclas[j] >= 1 for j in validos), len(validos)),
        },
    }

    # d) Enlaces neutros por documento (tipos 1+4)
    neutros = [len(t1[i]) + len(t4[i]) for i in validos]
    res["d_neutros_por_documento"] = {
        "distribucion": distribucion(neutros),
        "solo_tipo1": distribucion([len(t1[i]) for i in validos]),
        "solo_tipo4": distribucion([len(t4[i]) for i in validos]),
    }

    RUTA_SALIDA.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Escrito {RUTA_SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
