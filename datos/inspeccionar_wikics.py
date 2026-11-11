"""Inspección descriptiva de WikiCS (T0.2).

Lee datos/raw/wikics/ sin modificarlo y escribe:
- experimentos/resultados/inspeccion_wikics.json con todas las cifras.
- experimentos/resultados/figuras/grado_saliente_wikics.png con el histograma
  del grado saliente.

Solo mide: no preprocesa, no filtra y no toma decisiones sobre los datos.
Es determinista: no usa aleatoriedad y el JSON se escribe con claves ordenadas.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe datos\\inspeccionar_wikics.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "datos" / "raw" / "wikics"
DIR_RES = RAIZ / "experimentos" / "resultados"
RUTA_JSON = DIR_RES / "inspeccion_wikics.json"
RUTA_FIG = DIR_RES / "figuras" / "grado_saliente_wikics.png"

UMBRALES_GRADO = [1, 2, 3, 5, 10, 20]
RE_DESAMBIGUACION = re.compile(r"\(([^()]*)\)\s*$")
N_EJEMPLOS = 5


def resumen(valores):
    """Mín, mediana, media, p90, p99 y máx (percentiles con interpolación lineal de numpy)."""
    a = np.asarray(valores, dtype=float)
    return {
        "min": int(a.min()),
        "mediana": float(np.percentile(a, 50)),
        "media": round(float(a.mean()), 4),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def describir_clave(valor):
    """Tipo y forma de un valor de primer nivel de data.json."""
    d = {"tipo": type(valor).__name__, "longitud": len(valor)}
    if isinstance(valor, list) and valor:
        d["tipo_elemento"] = type(valor[0]).__name__
        if isinstance(valor[0], list):
            longitudes = sorted({len(x) for x in valor})
            d["longitudes_internas"] = (
                longitudes if len(longitudes) <= 3 else {"min": longitudes[0], "max": longitudes[-1]}
            )
    return d


def componentes_debiles(n, aristas):
    """Componentes débilmente conexas con union-find; devuelve los tamaños ordenados de mayor a menor."""
    padre = list(range(n))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for u, v in aristas:
        ru, rv = raiz(u), raiz(v)
        if ru != rv:
            padre[max(ru, rv)] = min(ru, rv)
    return sorted(Counter(raiz(i) for i in range(n)).values(), reverse=True)


def main():
    data = json.loads((DIR_RAW / "data.json").read_text(encoding="utf-8"))
    meta = json.loads((DIR_RAW / "metadata.json").read_text(encoding="utf-8"))
    nodos = meta["nodes"]
    links = data["links"]
    n = len(links)
    res = {}

    # a) Estructura de los ficheros
    res["a_estructura"] = {
        "data_json": {k: describir_clave(v) for k, v in data.items()},
        "metadata_json": {
            "labels": {"tipo": "dict", "longitud": len(meta["labels"])},
            "nodes": {"tipo": "list", "longitud": len(nodos), "claves_nodo": sorted(nodos[0].keys())},
        },
        "n_nodos": {
            "data_features": len(data["features"]),
            "data_labels": len(data["labels"]),
            "data_links": n,
            "metadata_nodes": len(nodos),
        },
        "dim_features": sorted({len(f) for f in data["features"]}),
        "clases_data_labels": sorted(set(data["labels"])),
        "mascaras": {
            "train_masks_n_particiones": len(data["train_masks"]),
            "stopping_masks_n_particiones": len(data["stopping_masks"]),
            "val_masks_n_particiones": len(data["val_masks"]),
            "train_true_por_particion": sorted({sum(m) for m in data["train_masks"]}),
            "stopping_true_por_particion": sorted({sum(m) for m in data["stopping_masks"]}),
            "val_true_por_particion": sorted({sum(m) for m in data["val_masks"]}),
            "test_true": sum(data["test_mask"]),
        },
    }

    # b) Correspondencia entre ficheros
    etiquetas_coinciden = sum(
        meta["labels"][str(data["labels"][i])] == nodos[i]["label"] for i in range(n)
    )
    pageids = [nd["id"] for nd in nodos]
    idx_por_pageid = {p: i for i, p in enumerate(pageids)}
    coincide_lista = coincide_multiconjunto = coincide_conjunto = 0
    outlinks_total = outlinks_fuera = 0
    solo_en_metadata = solo_en_data = 0
    nodos_con_outlinks_fuera = 0
    for i, nd in enumerate(nodos):
        dentro = [idx_por_pageid[o] for o in nd["outlinks"] if o in idx_por_pageid]
        fuera = len(nd["outlinks"]) - len(dentro)
        outlinks_total += len(nd["outlinks"])
        outlinks_fuera += fuera
        nodos_con_outlinks_fuera += fuera > 0
        coincide_lista += dentro == links[i]
        coincide_multiconjunto += Counter(dentro) == Counter(links[i])
        coincide_conjunto += set(dentro) == set(links[i])
        solo_en_metadata += len(set(dentro) - set(links[i]))
        solo_en_data += len(set(links[i]) - set(dentro))
    res["b_correspondencia"] = {
        "etiquetas_coinciden_por_indice": etiquetas_coinciden,
        "pageids_unicos": len(set(pageids)),
        "outlinks_total": outlinks_total,
        "outlinks_fuera_del_dataset": outlinks_fuera,
        "nodos_con_outlinks_fuera": nodos_con_outlinks_fuera,
        "nodos_coincidencia_lista_exacta_mismo_orden": coincide_lista,
        "nodos_coincidencia_multiconjunto": coincide_multiconjunto,
        "nodos_coincidencia_conjunto": coincide_conjunto,
        "pct_coincidencia_conjunto": round(100 * coincide_conjunto / n, 4),
        "enlaces_solo_en_metadata": solo_en_metadata,
        "enlaces_solo_en_data": solo_en_data,
    }

    # c) ¿Es dirigido?
    pares = [(u, v) for u in range(n) for v in links[u]]
    autobucles = sum(u == v for u, v in pares)
    unicos = set(pares)
    unicos_sin_bucle = {(u, v) for u, v in unicos if u != v}
    reciprocos = sum((v, u) in unicos_sin_bucle for u, v in unicos_sin_bucle)
    no_dirigidas = {(min(u, v), max(u, v)) for u, v in unicos_sin_bucle}
    res["c_direccion"] = {
        "enlaces_pares_ordenados": len(pares),
        "autobucles": autobucles,
        "duplicados": len(pares) - len(unicos),
        "enlaces_unicos_sin_autobucle": len(unicos_sin_bucle),
        "enlaces_con_reciproco": reciprocos,
        "pct_reciprocidad": round(100 * reciprocos / len(unicos_sin_bucle), 4),
        "aristas_no_dirigidas_equivalentes": len(no_dirigidas),
    }

    # d) Grados (sobre los enlaces únicos sin autobucle)
    grado_sal = [0] * n
    grado_ent = [0] * n
    for u, v in unicos_sin_bucle:
        grado_sal[u] += 1
        grado_ent[v] += 1
    tamanos_cc = componentes_debiles(n, unicos_sin_bucle)
    res["d_grados"] = {
        "nota": "calculados sobre enlaces únicos sin autobucles",
        "saliente": resumen(grado_sal),
        "entrante": resumen(grado_ent),
        "nodos_grado_saliente_0": grado_sal.count(0),
        "nodos_grado_entrante_0": grado_ent.count(0),
        "nodos_grado_0_ambos": sum(s == 0 and e == 0 for s, e in zip(grado_sal, grado_ent)),
        "nodos_grado_saliente_al_menos": {str(k): sum(g >= k for g in grado_sal) for k in UMBRALES_GRADO},
        "componentes_debiles": {
            "numero": len(tamanos_cc),
            "tamano_mayor": tamanos_cc[0],
            "tamanos_10_mayores": tamanos_cc[:10],
            "componentes_de_tamano_1": tamanos_cc.count(1),
        },
    }

    # e) Texto y títulos
    long_tokens = [len(nd["tokens"]) for nd in nodos]
    titulos = [nd["title"] for nd in nodos]
    repetidos = sorted(t for t, c in Counter(titulos).items() if c > 1)
    con_parentesis = [t for t in titulos if RE_DESAMBIGUACION.search(t)]
    sufijos = Counter(RE_DESAMBIGUACION.search(t).group(1) for t in con_parentesis)
    res["e_texto"] = {
        "tokens_por_nodo": {
            k: v for k, v in resumen(long_tokens).items() if k in ("min", "mediana", "p90", "max")
        },
        "nodos_sin_tokens": long_tokens.count(0),
        "titulos_distintos": len(set(titulos)),
        "titulos_repetidos": len(repetidos),
        "titulos_repetidos_ejemplos": repetidos[:N_EJEMPLOS],
        "titulos_con_guion_bajo": sum("_" in t for t in titulos),
        "titulos_con_espacio": sum(" " in t for t in titulos),
        "titulos_con_porcentaje": sum("%" in t for t in titulos),
        "titulos_con_parentesis_final": len(con_parentesis),
        "titulos_con_parentesis_ejemplos": sorted(con_parentesis)[:N_EJEMPLOS],
        "sufijos_parentesis_mas_frecuentes": [
            [s, c] for s, c in sorted(sufijos.items(), key=lambda x: (-x[1], x[0]))[:10]
        ],
        "n_clases": len(meta["labels"]),
        "nodos_por_clase": {
            f'{k}: {meta["labels"][str(k)]}': c for k, c in sorted(Counter(data["labels"]).items())
        },
    }

    DIR_RES.mkdir(parents=True, exist_ok=True)
    RUTA_JSON.write_text(
        json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Figura: histograma del grado saliente, eje y logarítmico, en escala de grises
    RUTA_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ancho = 5
    bordes = np.arange(0, max(grado_sal) + ancho + 1, ancho)
    ax.hist(grado_sal, bins=bordes, color="0.6", edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(f"Grado saliente (intervalos de {ancho})")
    ax.set_ylabel("Número de nodos (escala log)")
    ax.set_title(f"WikiCS: grado saliente (n = {n})")
    ax.grid(axis="y", which="major", color="0.85", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RUTA_FIG, metadata={"Software": None})
    plt.close(fig)

    print(f"Escrito {RUTA_JSON.relative_to(RAIZ)}")
    print(f"Escrito {RUTA_FIG.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
