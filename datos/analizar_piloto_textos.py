"""Análisis del piloto de textos (T0.3).

Lee datos/procesados/textos_piloto.jsonl y datos/raw/wikics/data.json y mide:
estados, fechas de revisión, recall de los enlaces de WikiCS, enlaces nombrados
en la introducción, anclas sin resolver, longitudes y restos de marcado.

Para atribuir los enlaces que faltan hace dos consultas auxiliares a la API,
cacheadas en datos/raw/api_piloto/:
- Redirecciones ACTUALES de los destinos que no casan con ningún título (en
  lotes de 50). Son las de hoy, no las de 2019.
- action=parse&prop=links sobre la revisión de 2019: enlaces del artículo
  renderizado, con las plantillas expandidas (con las plantillas de hoy).

Escribe experimentos/resultados/piloto_textos.json.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe datos\\analizar_piloto_textos.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from descargar_textos import API, USER_AGENT, normalizar_titulo  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
RUTA_PILOTO = RAIZ / "datos" / "procesados" / "textos_piloto.jsonl"
RUTA_DATA = RAIZ / "datos" / "raw" / "wikics" / "data.json"
RUTA_META = RAIZ / "datos" / "raw" / "wikics" / "metadata.json"
DIR_API = RAIZ / "datos" / "raw" / "api_piloto"
RUTA_SALIDA = RAIZ / "experimentos" / "resultados" / "piloto_textos.json"
RESTOS = ["{{", "}}", "[[", "]]", "<ref", "|", "'''"]


def pedir(sesion, params):
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    for intento in range(6):
        try:
            r = sesion.get(API, params=params, timeout=60)
            if r.status_code == 200:
                d = r.json()
                if d.get("error", {}).get("code") == "maxlag":
                    time.sleep(float(r.headers.get("Retry-After", 2 * 2**intento)))
                    continue
                return d
        except requests.RequestException:
            pass
        time.sleep(2 * 2**intento)
    raise RuntimeError(f"La API no responde: {params}")


def resolver_redirecciones(sesion, titulos):
    """titulo normalizado -> {'tipo': redireccion|no_existe|pagina|otro_espacio, 'destino': ...}. Cacheado."""
    ruta = DIR_API / "redirecciones.json"
    cache = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    pendientes = sorted(t for t in set(titulos) if t not in cache)
    for k in range(0, len(pendientes), 50):
        lote = pendientes[k:k + 50]
        d = pedir(sesion, {"action": "query", "titles": "|".join(lote), "redirects": "1"})["query"]
        norm = {x["from"]: x["to"] for x in d.get("normalized", [])}
        redir = {x["from"]: x["to"] for x in d.get("redirects", [])}
        interwiki = {x["title"] for x in d.get("interwiki", [])}
        paginas = {p["title"]: p for p in d.get("pages", [])}
        for t in lote:
            t2 = norm.get(t, t)
            if t2 in interwiki:
                cache[t] = {"tipo": "otro_espacio", "destino": None}
                continue
            final = redir.get(t2, t2)
            p = paginas.get(final, {})
            if p.get("missing") or p.get("invalid"):
                tipo = "no_existe"
            elif p.get("ns", 0) != 0:
                tipo = "otro_espacio"
            elif t2 in redir:
                tipo = "redireccion"
            else:
                tipo = "pagina"
            cache[t] = {"tipo": tipo, "destino": final}
    ruta.write_text(json.dumps(cache, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    return cache


def enlaces_renderizados(sesion, revid):
    """Títulos (ns 0) enlazados desde la revisión renderizada con plantillas expandidas. Cacheado."""
    ruta = DIR_API / f"parse_{revid}.json"
    if not ruta.exists():
        d = pedir(sesion, {"action": "parse", "oldid": str(revid), "prop": "links"})
        ruta.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    d = json.loads(ruta.read_text(encoding="utf-8"))
    return sorted({normalizar_titulo(l["title"]) for l in d["parse"]["links"] if l["ns"] == 0})


def resumen(v, claves=("min", "mediana", "media", "p90", "max")):
    a = np.asarray(v, dtype=float)
    todo = {"min": float(a.min()), "mediana": float(np.percentile(a, 50)), "media": float(a.mean()),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}
    return {k: round(todo[k], 4) for k in claves}


def marcar(texto, anclas):
    """Introducción con las anclas entre corchetes."""
    out, previo = [], 0
    for a in sorted(anclas, key=lambda a: a["inicio"]):
        out += [texto[previo:a["inicio"]], "[", texto[a["inicio"]:a["fin"]], "]"]
        previo = a["fin"]
    return "".join(out + [texto[previo:]])


def main():
    filas = [json.loads(l) for l in RUTA_PILOTO.read_text(encoding="utf-8").splitlines()]
    links = json.loads(RUTA_DATA.read_text(encoding="utf-8"))["links"]
    nodos = json.loads(RUTA_META.read_text(encoding="utf-8"))["nodes"]
    idx_por_titulo = {normalizar_titulo(nd["title"]): i for i, nd in enumerate(nodos)}
    ok = [f for f in filas if f["estado"] == "ok"]
    DIR_API.mkdir(parents=True, exist_ok=True)
    sesion = requests.Session()
    sesion.headers["User-Agent"] = USER_AGENT
    res = {"n_piloto": len(filas)}

    # a) Estados y fechas
    res["a_estados"] = dict(sorted(Counter(f["estado"] for f in filas).items()))
    res["a_no_ok"] = [{k: f.get(k) for k in ("idx", "pageid", "titulo", "estado", "detalle")}
                      for f in filas if f["estado"] != "ok"]
    fechas = sorted(f["timestamp"] for f in ok)
    anios = Counter(t[:4] for t in fechas)
    res["a_fechas"] = {"min": fechas[0], "mediana": fechas[len(fechas) // 2], "max": fechas[-1],
                       "por_anio": dict(sorted(anios.items())),
                       "en_2019_07_08": sum(t >= "2019-07-01" for t in fechas)}

    # Consultas auxiliares: enlaces renderizados y redirecciones
    renderizados = {f["idx"]: enlaces_renderizados(sesion, f["revid"]) for f in ok}
    titulos_sin_casar = set()
    for f in ok:
        titulos_sin_casar.update(f["destinos_articulo_sin_resolver"])
        titulos_sin_casar.update(t for t in renderizados[f["idx"]] if t not in idx_por_titulo)
        titulos_sin_casar.update(a["destino"] for a in f["anclas_intro"] if a["destino_idx"] is None)
    redir = resolver_redirecciones(sesion, titulos_sin_casar)

    def a_idx(titulo):
        """Índice WikiCS de un título, directo o a través de su redirección actual."""
        if titulo in idx_por_titulo:
            return idx_por_titulo[titulo]
        r = redir.get(titulo)
        if r and r["tipo"] == "redireccion":
            return idx_por_titulo.get(normalizar_titulo(r["destino"]))
        return None

    # b) Reproducción de los enlaces de WikiCS
    por_nodo = []
    for f in ok:
        i = f["idx"]
        L = set(links[i]) - {i}
        D_wt = set(f["destinos_articulo"]) - {i}
        D_wt_r = (D_wt | {a_idx(t) for t in f["destinos_articulo_sin_resolver"]}) - {None, i}
        D_rend = {a_idx(t) for t in renderizados[i]} - {None, i}
        def rec(D):
            return len(L & D) / len(L) if L else None
        def extra(D):
            return len(D - L) / len(D) if D else None
        por_nodo.append({
            "idx": i, "titulo": f["titulo"], "n_links_wikics": len(L),
            "recall_wikitexto": rec(D_wt), "recall_wikitexto_redir": rec(D_wt_r),
            "recall_renderizado": rec(D_rend),
            "extra_wikitexto": extra(D_wt), "extra_wikitexto_redir": extra(D_wt_r),
            "extra_renderizado": extra(D_rend),
            "faltan_renderizado": len(L - D_rend), "sobran_renderizado": len(D_rend - L),
        })

    def media(clave):
        v = [p[clave] for p in por_nodo if p[clave] is not None]
        return {"media_pct": round(100 * float(np.mean(v)), 2),
                "mediana_pct": round(100 * float(np.median(v)), 2), "n_nodos": len(v)}

    def micro(fuente):
        tot = sum(p["n_links_wikics"] for p in por_nodo)
        enc = sum(round(p[f"recall_{fuente}"] * p["n_links_wikics"])
                  for p in por_nodo if p[f"recall_{fuente}"] is not None)
        return round(100 * enc / tot, 2)

    res["b_reproduccion"] = {
        "nota": "L = links[i] sin duplicados ni autobucles; recall = |L∩D|/|L|; extra = |D\\L|/|D|",
        "wikitexto": {"recall": media("recall_wikitexto"), "extra": media("extra_wikitexto"),
                      "recall_micro_pct": micro("wikitexto")},
        "wikitexto_con_redirecciones_actuales": {
            "recall": media("recall_wikitexto_redir"), "extra": media("extra_wikitexto_redir"),
            "recall_micro_pct": micro("wikitexto_redir")},
        "renderizado_con_redirecciones_actuales": {
            "recall": media("recall_renderizado"), "extra": media("extra_renderizado"),
            "recall_micro_pct": micro("renderizado")},
        "peores_recall_renderizado": sorted(
            [p for p in por_nodo if p["recall_renderizado"] is not None],
            key=lambda p: (p["recall_renderizado"], p["idx"]))[:5],
        "peores_recall_wikitexto_redir": sorted(
            [p for p in por_nodo if p["recall_wikitexto_redir"] is not None],
            key=lambda p: (p["recall_wikitexto_redir"], p["idx"]))[:5],
        "nodos_sin_links_wikics": sum(p["n_links_wikics"] == 0 for p in por_nodo),
    }

    # c) Enlaces de WikiCS nombrados en la introducción
    nombrados = nombrados_r = total = 0
    for f in ok:
        i = f["idx"]
        L = set(links[i]) - {i}
        directos = {a["destino_idx"] for a in f["anclas_intro"]} - {None}
        con_redir = directos | ({a_idx(a["destino"]) for a in f["anclas_intro"]} - {None})
        total += len(L)
        nombrados += len(L & directos)
        nombrados_r += len(L & con_redir)
    res["c_enlaces_en_intro"] = {
        "links_wikics_total": total,
        "con_ancla_en_intro_directo": nombrados,
        "pct_directo": round(100 * nombrados / total, 2),
        "con_ancla_en_intro_con_redirecciones": nombrados_r,
        "pct_con_redirecciones": round(100 * nombrados_r / total, 2),
    }

    # d) Anclas de la introducción sin resolver
    anclas = [a for f in ok for a in f["anclas_intro"]]
    sin = [a for a in anclas if a["destino_idx"] is None]
    def clase(a):
        r = redir.get(a["destino"], {"tipo": "desconocido"})
        if r["tipo"] == "redireccion":
            return "redireccion_a_wikics" if a_idx(a["destino"]) is not None else "redireccion_fuera"
        return {"pagina": "pagina_fuera_de_wikics"}.get(r["tipo"], r["tipo"])
    clases = Counter(clase(a) for a in sin)
    ejemplos = []
    for c in sorted(clases):
        for a in [a for a in sin if clase(a) == c][:3]:
            r = redir.get(a["destino"], {})
            ejemplos.append({"clase": c, "texto": a["texto"], "destino": a["destino"],
                             "redirige_a": r.get("destino") if r.get("tipo") == "redireccion" else None})
    res["d_anclas_sin_resolver"] = {
        "anclas_intro": len(anclas), "sin_resolver": len(sin),
        "pct_sin_resolver": round(100 * len(sin) / len(anclas), 2),
        "por_clase": dict(sorted(clases.items())), "ejemplos": ejemplos[:12],
    }

    # e) Longitud de la introducción
    palabras = [f["n_palabras_intro"] for f in ok]
    res["e_longitud_intro"] = {**resumen(palabras, ("min", "mediana", "p90", "max")),
                               "menos_de_20": sum(p < 20 for p in palabras),
                               "cero": sum(p == 0 for p in palabras)}

    # f) Calidad del parseo
    desalineadas = sum(f["texto_intro"][a["inicio"]:a["fin"]] != a["texto"]
                       for f in ok for a in f["anclas_intro"])
    restos = {}
    for r in RESTOS:
        casos = [(f["idx"], f["texto_intro"]) for f in ok if r in f["texto_intro"]]
        restos[r] = {"nodos": len(casos), "ejemplos": [
            {"idx": i, "contexto": t[max(0, t.find(r) - 60):t.find(r) + 60]} for i, t in casos[:3]]}
    con_20 = [f for f in ok if f["n_palabras_intro"] >= 20]
    res["f_calidad"] = {"anclas_desalineadas": desalineadas, "restos_de_marcado": restos,
                        "ejemplos_marcados": [{"idx": f["idx"], "titulo": f["titulo"],
                                               "texto": marcar(f["texto_intro"], f["anclas_intro"])}
                                              for f in con_20[:3]]}

    RUTA_SALIDA.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
