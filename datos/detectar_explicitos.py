"""Detector de destinos explícitos (T0.5).

Un destino j es EXPLÍCITO para un documento i (j != i) si en la introducción de
i hay un ancla a j, o si alguno de los nombres de j aparece en ella como palabra
completa. Si no, es IMPLÍCITO. Es una propiedad del par documento-destino.

Nombres de cada destino j (nodo válido), con su fuente. Por defecto solo se usan
las fuentes título y redirección (decisión de T0.5: la fuente ancla metía
palabras corrientes en el diccionario). Con --con-anclas se añade la fuente
ancla y se reproduce la primera versión del detector:
  titulo      su título completo normalizado ("_" -> espacio). Ya no se añade el
              título sin el sufijo "(...)": ese nombre es el que Wikipedia
              considera ambiguo. Afecta también a --con-anclas;
  redireccion títulos que redirigen a j según datos/raw/redirecciones_actuales.json
              (solo cubre títulos que aparecían enlazados en algún artículo);
  ancla       textos visibles de todos los wikilinks que, en cualquier parte del
              wikitexto de cualquier artículo válido, apuntan a j (directo o por
              redirección), incluido el sufijo de enlace ([[x]]s).

Dos reglas de detección por nombre:
  estricta  distingue mayúsculas; nombres de >= 3 caracteres, o siglas en
            mayúsculas de >= 2; descarta nombres con más de un destino.
  laxa      sin distinguir mayúsculas; nombres de >= 2 caracteres; un nombre
            ambiguo cuenta para todos sus destinos.

Todo offline. Escribe:
- datos/procesados/explicitos.jsonl (una línea por nodo válido)
- datos/procesados/nombres_destinos.json (diccionario nombre -> destinos y fuentes)
- experimentos/resultados/explicitos.json (medidas)
- experimentos/resultados/muestra_explicitos_sin_anclas.csv (muestra estratificada
  por fuente, regla estricta, para revisión manual)
Con --con-anclas los tres primeros llevan el sufijo _con_anclas y la muestra es
la de la primera versión (100 pares de la regla laxa, muestra_explicitos.csv).

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe datos\\detectar_explicitos.py [--con-anclas]
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
import mwparserfromhell
import numpy as np
from mwparserfromhell.nodes import Text, Wikilink

sys.path.insert(0, str(Path(__file__).resolve().parent))
from descargar_textos import RE_SUFIJO_ENLACE, es_descartado, normalizar_titulo  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
RUTA_TEXTOS = RAIZ / "datos" / "procesados" / "textos.jsonl"
RUTA_DATA = RAIZ / "datos" / "raw" / "wikics" / "data.json"
DIR_WIKITEXT = RAIZ / "datos" / "raw" / "wikitext"
RUTA_REDIRECCIONES = RAIZ / "datos" / "raw" / "redirecciones_actuales.json"
RUTA_EXPLICITOS = RAIZ / "datos" / "procesados" / "explicitos.jsonl"
RUTA_NOMBRES = RAIZ / "datos" / "procesados" / "nombres_destinos.json"
RUTA_MEDIDAS = RAIZ / "experimentos" / "resultados" / "explicitos.json"
RUTA_MUESTRA = RAIZ / "experimentos" / "resultados" / "muestra_explicitos.csv"
RUTA_MUESTRA_SIN_ANCLAS = RAIZ / "experimentos" / "resultados" / "muestra_explicitos_sin_anclas.csv"

FUENTES = ("titulo", "redireccion", "ancla")  # orden de prioridad al elegir la fuente
UMBRALES_M = [1, 2, 3, 5, 10]
UMBRALES_PALABRAS = [0, 20, 30]
TAM_MUESTRA = 100
SEMILLA = 42
TAM_ESTRATO = 30  # muestra sin anclas: pares por fuente (título, redirección)
SEMILLA_ESTRATOS = 44
CONTEXTO = 60


def minusculas(s):
    """Minúsculas carácter a carácter sin cambiar la longitud (las posiciones se conservan)."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in s)


def limpiar_nombre(s):
    return re.sub(r"\s+", " ", s).strip()


def pct(a, b):
    return round(100 * a / b, 2) if b else None


def distribucion(v):
    a = np.asarray(v, dtype=float)
    return {"mediana": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "max": int(a.max()), "media": round(float(a.mean()), 4)}


# ---------------------------------------------------------------- diccionario

def anclas_del_articulo(wikicodigo):
    """(texto visible, título destino normalizado) de todos los wikilinks, a cualquier profundidad."""
    salida = []

    def recorrer(wc):
        nodos = wc.nodes
        for k, n in enumerate(nodos):
            if isinstance(n, Wikilink) and not es_descartado(n):
                destino = normalizar_titulo(str(n.title))
                texto = (n.text.strip_code() if n.text is not None
                         else str(n.title).strip().lstrip(":"))
                if k + 1 < len(nodos) and isinstance(nodos[k + 1], Text):
                    m = RE_SUFIJO_ENLACE.match(str(nodos[k + 1].value))
                    if m:
                        texto += m.group()
                texto = limpiar_nombre(texto)
                if destino and texto:
                    salida.append((texto, destino))
            for hijo in n.__children__():
                recorrer(hijo)

    recorrer(wikicodigo)
    return salida


def construir_diccionario(filas, valido, idx_por_titulo, redir, con_anclas):
    """nombre -> {destino: conjunto de fuentes}, solo con destinos válidos."""
    nombres = defaultdict(lambda: defaultdict(set))

    def resolver(titulo):
        if titulo in idx_por_titulo:
            return idx_por_titulo[titulo]
        r = redir.get(titulo)
        if r and r["tipo"] == "redireccion":
            return idx_por_titulo.get(normalizar_titulo(r["destino"]))
        return None

    for f in filas:
        if not valido[f["idx"]]:
            continue
        # Solo el título completo: sin el sufijo "(...)", el nombre es justo el que
        # Wikipedia considera ambiguo (decisión de T0.5).
        titulo = limpiar_nombre(f["titulo"].replace("_", " "))
        nombres[titulo][f["idx"]].add("titulo")
    for origen, r in redir.items():
        if r["tipo"] == "redireccion":
            j = idx_por_titulo.get(normalizar_titulo(r["destino"]))
            if j is not None and valido[j]:
                nombres[limpiar_nombre(origen)][j].add("redireccion")
    for k, f in enumerate(filas):
        if not con_anclas:
            break
        if not valido[f["idx"]]:
            continue
        contenido = json.loads((DIR_WIKITEXT / f"{f['pageid']}.json")
                               .read_text(encoding="utf-8"))["contenido"]
        for texto, destino in anclas_del_articulo(mwparserfromhell.parse(contenido)):
            j = resolver(destino)
            if j is not None and valido[j]:
                nombres[texto][j].add("ancla")
        if (k + 1) % 2000 == 0:
            print(f"  anclas: {k + 1}/{len(filas)} artículos", flush=True)
    return nombres


# ---------------------------------------------------------------- reglas

def es_sigla(nombre):
    return len(nombre) >= 2 and " " not in nombre and nombre.isupper()


def automata(entradas):
    """entradas: clave (texto a buscar) -> valor. Devuelve un autómata Aho-Corasick."""
    a = ahocorasick.Automaton()
    for clave, valor in entradas.items():
        a.add_word(clave, (clave, valor))
    a.make_automaton()
    return a


def preparar_reglas(nombres):
    """Construye los autómatas de las dos reglas.

    Valor de cada clave: lista de (destino, nombre original) que esa clave activa.
    """
    estricta = {}
    for nombre, destinos in nombres.items():
        if len(destinos) != 1:
            continue
        if len(nombre) < 3 and not es_sigla(nombre):
            continue
        (j,) = destinos
        estricta[nombre] = [(j, nombre)]
    laxa = defaultdict(set)
    for nombre, destinos in nombres.items():
        if len(nombre) < 2:
            continue
        for j in destinos:
            laxa[minusculas(nombre)].add((j, nombre))
    laxa = {k: sorted(v) for k, v in laxa.items()}
    return automata(estricta), automata(laxa), len(estricta), len(laxa)


def palabra_completa(texto, inicio, fin):
    antes = texto[inicio - 1] if inicio > 0 else " "
    despues = texto[fin] if fin < len(texto) else " "
    return not antes.isalnum() and not despues.isalnum()


def buscar(aut, texto):
    """{destino: [(inicio, fin, nombre original), ...]} de los nombres hallados como palabra completa."""
    hallados = defaultdict(list)
    for fin_incl, (clave, valores) in aut.iter(texto):
        inicio, fin = fin_incl - len(clave) + 1, fin_incl + 1
        if palabra_completa(texto, inicio, fin):
            for j, nombre in valores:
                hallados[j].append((inicio, fin, nombre))
    return hallados


def elegir(nombres, j, ocurrencias):
    """Ocurrencia representativa: mejor fuente, luego nombre más largo, luego primera posición."""
    def clave(o):
        fuentes = nombres[o[2]][j]
        prioridad = min(FUENTES.index(s) for s in fuentes)
        return (prioridad, -(o[1] - o[0]), o[0], o[2])
    o = min(ocurrencias, key=clave)
    fuentes = sorted(nombres[o[2]][j], key=FUENTES.index)
    return {"inicio": o[0], "fin": o[1], "nombre": o[2], "fuente": fuentes[0], "fuentes": fuentes}


# ---------------------------------------------------------------- muestras

def contexto_de(texto, r):
    return re.sub(r"\s+", " ", texto[max(0, r["inicio"] - CONTEXTO):r["inicio"]] + "["
                  + texto[r["inicio"]:r["fin"]] + "]" + texto[r["fin"]:r["fin"] + CONTEXTO])


def muestra_laxa(explicitos, validos, A, filas, titulos):
    """Primera versión: 100 pares de la regla laxa, vía nombre, fuera de A."""
    candidatos = sorted((i, j) for i in validos for j, r in explicitos["laxa"][i].items()
                        if r["via"] == "nombre" and j not in A[i])
    muestra = sorted(random.Random(SEMILLA).sample(candidatos, TAM_MUESTRA))
    with RUTA_MUESTRA.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx_origen", "titulo_origen", "titulo_destino", "nombre_detectado",
                    "fuente_nombre", "regla_estricta", "contexto", "correcto"])
        for i, j in muestra:
            r = explicitos["laxa"][i][j]
            texto = filas[i]["texto_intro"]
            contexto = (texto[max(0, r["inicio"] - CONTEXTO):r["inicio"]] + "["
                        + texto[r["inicio"]:r["fin"]] + "]" + texto[r["fin"]:r["fin"] + CONTEXTO])
            w.writerow([i, titulos[i], titulos[j], r["nombre"], r["fuente"],
                        "sí" if j in explicitos["estricta"][i] else "no",
                        re.sub(r"\s+", " ", contexto), ""])
    return {"candidatos": len(candidatos), "tamano": TAM_MUESTRA, "semilla": SEMILLA,
                        "fuentes": dict(Counter(explicitos["laxa"][i][j]["fuente"] for i, j in muestra)),
                        "tambien_en_estricta": sum(j in explicitos["estricta"][i] for i, j in muestra)}


def muestra_estratificada(explicitos, validos, A, filas, titulos):
    """TAM_ESTRATO pares por fuente (título, redirección): regla estricta, vía nombre, fuera de A."""
    estratos = {"titulo": [], "redireccion": []}
    for i in validos:
        for j, r in explicitos["estricta"][i].items():
            if r["via"] == "nombre" and j not in A[i] and r["fuente"] in estratos:
                estratos[r["fuente"]].append((i, j))
    rng = random.Random(SEMILLA_ESTRATOS)
    muestra = []
    for fuente in ("titulo", "redireccion"):
        pares = sorted(estratos[fuente])
        muestra += [(fuente, i, j) for i, j in sorted(rng.sample(pares, min(TAM_ESTRATO, len(pares))))]
    with RUTA_MUESTRA_SIN_ANCLAS.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx_origen", "titulo_origen", "titulo_destino", "nombre_detectado",
                    "fuente_nombre", "contexto", "correcto"])
        for fuente, i, j in muestra:
            r = explicitos["estricta"][i][j]
            w.writerow([i, titulos[i], titulos[j], r["nombre"], fuente,
                        contexto_de(filas[i]["texto_intro"], r), ""])
    return {"regla": "estricta", "semilla": SEMILLA_ESTRATOS, "tam_estrato": TAM_ESTRATO,
            "candidatos_por_fuente": {k: len(v) for k, v in estratos.items()},
            "tamano": len(muestra),
            "ruta": str(RUTA_MUESTRA_SIN_ANCLAS.relative_to(RAIZ))}


# ---------------------------------------------------------------- principal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--con-anclas", action="store_true",
                    help="incluir la fuente ancla (primera versión del detector)")
    args = ap.parse_args()
    sufijo = "_con_anclas" if args.con_anclas else ""
    ruta_explicitos = RUTA_EXPLICITOS.with_stem(RUTA_EXPLICITOS.stem + sufijo)
    ruta_nombres = RUTA_NOMBRES.with_stem(RUTA_NOMBRES.stem + sufijo)
    ruta_medidas = RUTA_MEDIDAS.with_stem(RUTA_MEDIDAS.stem + sufijo)
    fuentes_dic = FUENTES if args.con_anclas else ("titulo", "redireccion")

    filas = [json.loads(l) for l in RUTA_TEXTOS.read_text(encoding="utf-8").splitlines()]
    links = json.loads(RUTA_DATA.read_text(encoding="utf-8"))["links"]
    n = len(links)
    assert [f["idx"] for f in filas] == list(range(n))
    valido = [f["estado"] == "ok" for f in filas]
    validos = [i for i in range(n) if valido[i]]
    titulos = [f["titulo"] for f in filas]
    idx_por_titulo = {normalizar_titulo(t): i for i, t in enumerate(titulos)}
    redir = json.loads(RUTA_REDIRECCIONES.read_text(encoding="utf-8"))["titulos"]

    A = [{j for j in set(links[i]) if j != i and valido[j]} if valido[i] else set()
         for i in range(n)]
    escritos = [set(f["destinos_cuerpo"]) | set(f["destinos_cuerpo_redir"]) for f in filas]
    B = [A[i] & escritos[i] for i in range(n)]
    anclados = [({a["destino_idx"] for a in f["anclas_intro"]}
                 | {a["destino_idx_redir"] for a in f["anclas_intro"]}) - {None, f["idx"]}
                for f in filas]

    print("Construyendo el diccionario de nombres...", flush=True)
    nombres = construir_diccionario(filas, valido, idx_por_titulo, redir, args.con_anclas)
    aut_estricta, aut_laxa, n_estricta, n_laxa = preparar_reglas(nombres)

    print("Detectando destinos explícitos...", flush=True)
    explicitos = {"estricta": {}, "laxa": {}}  # regla -> i -> j -> registro
    for i in validos:
        texto = filas[i]["texto_intro"]
        for regla, aut, t in (("estricta", aut_estricta, texto), ("laxa", aut_laxa, minusculas(texto))):
            hallados = buscar(aut, t)
            d = {}
            for j in anclados[i]:
                if valido[j]:
                    d[j] = {"via": "ancla"}
            for j, ocurrencias in hallados.items():
                if j != i and j not in d:
                    d[j] = {"via": "nombre", **elegir(nombres, j, ocurrencias)}
            explicitos[regla][i] = d

    # Comprobación: todo par con ancla en la introducción es explícito con las dos reglas
    fallos = [(regla, i, j) for regla in explicitos for i in validos for j in anclados[i]
              if valido[j] and j not in explicitos[regla][i]]
    if fallos:
        raise SystemExit(f"Pares con ancla que no salen como explícitos: {fallos[:10]}")

    # Salidas de datos
    with ruta_explicitos.open("w", encoding="utf-8", newline="\n") as f:
        for i in validos:
            fila = {"idx": i}
            for regla in ("estricta", "laxa"):
                fila[regla] = [{"destino": j, **explicitos[regla][i][j]}
                               for j in sorted(explicitos[regla][i])]
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    ruta_nombres.write_text(json.dumps(
        {nombre: {str(j): sorted(fs, key=FUENTES.index) for j, fs in sorted(ds.items())}
         for nombre, ds in sorted(nombres.items())}, ensure_ascii=False) + "\n", encoding="utf-8")

    res = {"n_validos": len(validos), "fuentes_diccionario": list(fuentes_dic)}

    # a) Diccionario
    por_fuente = {s: sum(1 for ds in nombres.values() if any(s in fs for fs in ds.values()))
                  for s in fuentes_dic}
    nombres_por_destino = Counter()
    for ds in nombres.values():
        for j in ds:
            nombres_por_destino[j] += 1
    laxo = defaultdict(set)
    for nombre, ds in nombres.items():
        laxo[minusculas(nombre)].update(ds)
    def top_ambiguos(dic):
        return [{"nombre": k, "n_destinos": len(v),
                 "ejemplos": [titulos[j] for j in sorted(v)[:5]]}
                for k, v in sorted(dic.items(), key=lambda x: (-len(x[1]), x[0]))[:20]]
    res["a_diccionario"] = {
        "nombres_distintos": len(nombres),
        "nombres_por_fuente": por_fuente,
        "pares_nombre_destino": sum(len(ds) for ds in nombres.values()),
        "nombres_por_destino": distribucion([nombres_por_destino[j] for j in validos]),
        "ambiguos_distinguiendo_mayusculas": sum(len(ds) > 1 for ds in nombres.values()),
        "ambiguos_sin_distinguir_mayusculas": sum(len(v) > 1 for v in laxo.values()),
        "top20_ambiguos": top_ambiguos(nombres),
        "top20_ambiguos_sin_mayusculas": top_ambiguos(laxo),
        "claves_regla_estricta": n_estricta,
        "claves_regla_laxa": n_laxa,
    }

    # b) Cobertura sobre B y A \ B
    def cobertura(grafo, regla):
        total = sum(map(len, grafo))
        via = Counter()
        for i in validos:
            for j in grafo[i]:
                r = explicitos[regla][i].get(j)
                if r is None:
                    via["implicito"] += 1
                elif r["via"] == "ancla":
                    via["ancla"] += 1
                else:
                    via["nombre"] += 1
                    via[f"nombre_{r['fuente']}"] += 1
        exp = via["ancla"] + via["nombre"]
        return {"enlaces": total, "explicitos": exp, "pct_explicitos": pct(exp, total),
                "via_ancla": via["ancla"], "pct_via_ancla": pct(via["ancla"], total),
                "via_nombre": via["nombre"], "pct_via_nombre": pct(via["nombre"], total),
                "via_nombre_por_fuente": {s: via[f"nombre_{s}"] for s in FUENTES},
                "pct_via_nombre_por_fuente": {s: pct(via[f"nombre_{s}"], total) for s in FUENTES},
                "implicitos": via["implicito"]}
    A_menos_B = [A[i] - B[i] for i in range(n)]
    res["b_cobertura"] = {regla: {"B": cobertura(B, regla), "A_menos_B": cobertura(A_menos_B, regla)}
                          for regla in ("estricta", "laxa")}

    # c) Explícitos sin enlace (no están en A)
    res["c_explicitos_sin_enlace"] = {}
    for regla in ("estricta", "laxa"):
        v = [sum(1 for j in explicitos[regla][i] if j not in A[i]) for i in validos]
        res["c_explicitos_sin_enlace"][regla] = {
            **distribucion(v), "total": sum(v),
            "documentos_con_al_menos_1": sum(x >= 1 for x in v)}

    # d) Elegibilidad: enlaces de B con destino implícito
    palabras = [f["n_palabras_intro"] or 0 for f in filas]
    res["d_elegibilidad"] = {}
    for regla in ("estricta", "laxa"):
        impl = {i: sum(1 for j in B[i] if j not in explicitos[regla][i]) for i in validos}
        res["d_elegibilidad"][regla] = {
            f"intro>={p}": {str(m): sum(1 for i in validos if impl[i] >= m and palabras[i] >= p)
                            for m in UMBRALES_M} for p in UMBRALES_PALABRAS}

    # e) Muestra para revisión manual
    if args.con_anclas:
        res["e_muestra"] = muestra_laxa(explicitos, validos, A, filas, titulos)
    else:
        res["e_muestra"] = muestra_estratificada(explicitos, validos, A, filas, titulos)

    ruta_medidas.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Escritos {ruta_explicitos.relative_to(RAIZ)}, {ruta_nombres.relative_to(RAIZ)}, "
          f"{ruta_medidas.relative_to(RAIZ)} y la muestra")


if __name__ == "__main__":
    main()
