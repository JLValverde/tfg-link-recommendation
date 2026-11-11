"""Reconstrucción del texto de los nodos de WikiCS desde Wikipedia (T0.3).

Para cada nodo descarga el wikitexto de la última revisión anterior a la fecha
de corte (dump enwiki-20190820 del que sale WikiCS), extrae la introducción como
texto limpio, conserva las anclas de los enlaces con sus posiciones y registra
los destinos de WikiCS enlazados desde todo el wikitexto.

Tres fases, todas reanudables:
1. Descarga: datos/raw/wikitext/<pageid>.json. Si el fichero existe no se vuelve
   a descargar; los errores no se guardan, así que se reintentan al relanzar.
2. Parseo y redirecciones actuales de los títulos que no casan con WikiCS, en
   lotes de 50, cacheadas en datos/raw/redirecciones_actuales.json.
3. Escritura: datos/procesados/textos.jsonl (o textos_piloto.jsonl con
   --piloto), una línea por nodo, ordenado por idx.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe datos\\descargar_textos.py --piloto 50 --semilla 42
    .venv\\Scripts\\python.exe datos\\descargar_textos.py --log experimentos\\resultados\\descarga_textos.log
"""

import argparse
import html
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mwparserfromhell
import requests
from mwparserfromhell.nodes import (
    Comment,
    ExternalLink,
    Heading,
    HTMLEntity,
    Tag,
    Template,
    Text,
    Wikilink,
)

RAIZ = Path(__file__).resolve().parent.parent
RUTA_METADATA = RAIZ / "datos" / "raw" / "wikics" / "metadata.json"
DIR_WIKITEXT = RAIZ / "datos" / "raw" / "wikitext"
RUTA_REDIRECCIONES = RAIZ / "datos" / "raw" / "redirecciones_actuales.json"
DIR_PROCESADOS = RAIZ / "datos" / "procesados"
DIR_RESULTADOS = RAIZ / "experimentos" / "resultados"

# Dump enwiki-20190820: data_processing/category-sanitizer/run.sh del repo original
FECHA_CORTE = "2019-08-20T00:00:00Z"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "tejedor-de-redes/0.1 (TFG UGR; https://github.com/JLValverde/tfg-link-recommendation)"
MAX_REINTENTOS = 6
ESPERA_BASE_S = 2.0
MAX_FALLOS_SEGUIDOS = 10  # rechazo sostenido de la API: se detiene la descarga
PROGRESO_CADA = 500
LOTE_REDIRECCIONES = 50

PREFIJOS_DESCARTADOS = ("file:", "image:", "category:")
# Etiquetas cuyo contenido no es prosa: se eliminan con su contenido
TAGS_DESCARTADOS = {
    "ref", "references", "gallery", "timeline", "imagemap", "templatestyles",
    "score", "graph", "mapframe", "maplink", "table",
}
RE_REDIRECCION = re.compile(r"^\s*#redirect", re.IGNORECASE)
RE_SUFIJO_ENLACE = re.compile(r"[a-z]+")  # linktrail de la Wikipedia en inglés
CARACTERES_INVALIDOS = set("|{}[]<>#")

log = logging.getLogger("descarga")


class APIRechaza(RuntimeError):
    """La API rechaza las peticiones de forma sostenida."""


# ---------------------------------------------------------------- API

class ClienteAPI:
    """Peticiones en serie con maxlag=5, reintentos con espera exponencial y contadores."""

    def __init__(self):
        self.sesion = requests.Session()
        self.sesion.headers["User-Agent"] = USER_AGENT
        self.peticiones = 0
        self.segundos_red = 0.0

    def pedir(self, params, metodo="GET"):
        """Devuelve (json, None) o (None, descripción del error)."""
        params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
        ultimo = None
        for intento in range(MAX_REINTENTOS):
            t0 = time.perf_counter()
            try:
                if metodo == "POST":
                    r = self.sesion.post(API, data=params, timeout=60)
                else:
                    r = self.sesion.get(API, params=params, timeout=60)
                self.peticiones += 1
                self.segundos_red += time.perf_counter() - t0
                if r.status_code == 200:
                    d = r.json()
                    if d.get("error", {}).get("code") == "maxlag":
                        ultimo = "maxlag"
                        time.sleep(float(r.headers.get("Retry-After", ESPERA_BASE_S * 2**intento)))
                        continue
                    if "error" in d:
                        return None, f"API: {d['error'].get('code')}: {d['error'].get('info')}"
                    return d, None
                ultimo = f"HTTP {r.status_code}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    return None, ultimo
            except requests.RequestException as e:
                self.peticiones += 1
                ultimo = repr(e)
            time.sleep(ESPERA_BASE_S * 2**intento)
        return None, f"reintentos agotados: {ultimo}"


def descargar(cliente, pageid):
    d, error = cliente.pedir({
        "action": "query", "prop": "revisions", "pageids": str(pageid), "rvlimit": "1",
        "rvdir": "older", "rvstart": FECHA_CORTE, "rvprop": "ids|timestamp|content",
        "rvslots": "main",
    })
    if error:
        # Fallo de red, HTTP o de la API: cuenta para detectar un rechazo sostenido
        return {"pageid": pageid, "estado": "error", "detalle": error, "fallo_api": True}
    return interpretar(pageid, d)


def interpretar(pageid, d):
    pagina = d["query"]["pages"][0]
    base = {"pageid": pageid, "titulo_api": pagina.get("title")}
    if pagina.get("missing") or pagina.get("invalid"):
        return {**base, "estado": "no_existe"}
    revs = pagina.get("revisions") or []
    if not revs:
        return {**base, "estado": "sin_revision"}
    rev = revs[0]
    slot = rev.get("slots", {}).get("main", {})
    base.update({"revid": rev.get("revid"), "timestamp": rev.get("timestamp")})
    if "content" not in slot:
        return {**base, "estado": "error", "detalle": "revisión sin contenido (¿oculto?)"}
    base["contenido"] = slot["content"]
    base["estado"] = "redireccion" if RE_REDIRECCION.match(slot["content"]) else "ok"
    return base


def resolver_redirecciones(cliente, titulos):
    """Resuelve títulos con la API actual (lotes de 50) y los añade a la caché en disco.

    Cada entrada: {"tipo": redireccion|pagina|no_existe|otro_espacio|invalido,
                   "destino": título final o None, "consultado": fecha ISO}.
    """
    cache = (json.loads(RUTA_REDIRECCIONES.read_text(encoding="utf-8"))
             if RUTA_REDIRECCIONES.exists() else {"titulos": {}})
    tabla = cache["titulos"]
    pendientes = sorted(t for t in set(titulos) if t not in tabla)
    hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    invalidos = [t for t in pendientes if CARACTERES_INVALIDOS & set(t) or len(t.encode()) > 255]
    for t in invalidos:
        tabla[t] = {"tipo": "invalido", "destino": None, "consultado": hoy}
    pendientes = [t for t in pendientes if t not in tabla]
    log.info("Redirecciones: %d títulos en caché, %d por consultar (%d lotes)",
             len(tabla), len(pendientes), -(-len(pendientes) // LOTE_REDIRECCIONES))
    fallos_seguidos = 0
    for k in range(0, len(pendientes), LOTE_REDIRECCIONES):
        lote = pendientes[k:k + LOTE_REDIRECCIONES]
        d, error = cliente.pedir(
            {"action": "query", "titles": "|".join(lote), "redirects": "1"}, metodo="POST")
        if error:
            fallos_seguidos += 1
            log.error("Redirecciones: lote %d falla: %s", k // LOTE_REDIRECCIONES, error)
            if fallos_seguidos >= MAX_FALLOS_SEGUIDOS:
                guardar_redirecciones(cache)
                raise APIRechaza(f"{fallos_seguidos} lotes de redirecciones seguidos con error")
            continue
        fallos_seguidos = 0
        q = d["query"]
        norm = {x["from"]: x["to"] for x in q.get("normalized", [])}
        redir = {x["from"]: x["to"] for x in q.get("redirects", [])}
        interwiki = {x["title"] for x in q.get("interwiki", [])}
        paginas = {p["title"]: p for p in q.get("pages", [])}
        for t in lote:
            t2 = norm.get(t, t)
            if t2 in interwiki:
                tabla[t] = {"tipo": "otro_espacio", "destino": None, "consultado": hoy}
                continue
            final = redir.get(t2, t2)
            p = paginas.get(final, {})
            if p.get("invalid"):
                tipo = "invalido"
            elif p.get("missing"):
                tipo = "no_existe"
            elif p.get("ns", 0) != 0:
                tipo = "otro_espacio"
            elif t2 in redir:
                tipo = "redireccion"
            else:
                tipo = "pagina"
            tabla[t] = {"tipo": tipo, "destino": final, "consultado": hoy}
        n_lote = k // LOTE_REDIRECCIONES + 1
        if n_lote % 100 == 0:
            guardar_redirecciones(cache)
            log.info("Redirecciones: %d/%d títulos consultados", k + len(lote), len(pendientes))
    guardar_redirecciones(cache)
    return tabla


def guardar_redirecciones(cache):
    tmp = RUTA_REDIRECCIONES.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(RUTA_REDIRECCIONES)


# ---------------------------------------------------------------- parseo

def normalizar_titulo(t):
    """Normaliza un título como MediaWiki: sin #fragmento, '_' -> espacio, primera letra en mayúscula."""
    t = html.unescape(t).split("#", 1)[0].replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip().lstrip(":").strip()
    return t[:1].upper() + t[1:]


def es_descartado(link):
    return str(link.title).strip().lstrip(":").strip().lower().startswith(PREFIJOS_DESCARTADOS)


class Renderizador:
    """Convierte wikicódigo en texto plano acumulando anclas con posiciones sobre ese texto."""

    def __init__(self, parar_en=None):
        self.partes = []
        self.longitud = 0
        self.anclas = []
        self.parar_en = parar_en  # al llegar a este nodo se deja de renderizar
        self.parado = False

    def _emitir(self, s):
        self.partes.append(s)
        self.longitud += len(s)

    def nodos(self, nodos):
        nodos = list(nodos)
        recorte = 0  # caracteres del Text siguiente ya absorbidos como sufijo de enlace
        for k, n in enumerate(nodos):
            if n is self.parar_en:
                self.parado = True
            if self.parado:
                return
            if isinstance(n, Text) and recorte:
                self._emitir(str(n.value)[recorte:])
                recorte = 0
                continue
            recorte = 0
            ancla = self.nodo(n)
            if self.parado:
                return
            # Sufijo de enlace: [[vector processor]]s -> "vector processors"
            if isinstance(n, Wikilink) and not es_descartado(n) and k + 1 < len(nodos) \
                    and isinstance(nodos[k + 1], Text):
                m = RE_SUFIJO_ENLACE.match(str(nodos[k + 1].value))
                if m:
                    self._emitir(m.group())
                    recorte = m.end()
                    if ancla is not None:
                        ancla["fin"] = self.longitud

    def nodo(self, n):
        """Emite el nodo; si es un wikilink que genera ancla, la devuelve."""
        if isinstance(n, Text):
            self._emitir(str(n.value))
        elif isinstance(n, HTMLEntity):
            self._emitir(n.normalize())
        elif isinstance(n, Wikilink):
            if es_descartado(n):
                return None
            inicio = self.longitud
            if n.text is not None:
                self.nodos(n.text.nodes)
            else:
                self._emitir(str(n.title).strip().lstrip(":"))
            destino = normalizar_titulo(str(n.title))
            if destino:  # [[#Sección]] enlaza a la propia página: no es ancla
                ancla = {"inicio": inicio, "fin": self.longitud, "destino": destino}
                self.anclas.append(ancla)
                return ancla
        elif isinstance(n, ExternalLink):
            if n.title is not None:
                self.nodos(n.title.nodes)
        elif isinstance(n, Tag):
            nombre = str(n.tag).strip().lower()
            if nombre in TAGS_DESCARTADOS or n.wiki_markup == "{|":
                return None
            if nombre == "br":
                self._emitir(" ")
            elif n.contents is not None:
                self.nodos(n.contents.nodes)
        # Plantillas, comentarios, encabezados, argumentos {{{x}}}: se omiten
        return None

    def texto(self):
        return "".join(self.partes)


def reescribir(texto, anclas, patron, reemplazo):
    """Aplica una sustitución regex y reubica las posiciones de las anclas."""
    nuevas, mapa = [], [0] * (len(texto) + 1)
    pos_nueva, previo = 0, 0
    for m in re.finditer(patron, texto):
        for p in range(previo, m.start()):
            mapa[p] = pos_nueva + (p - previo)
        pos_nueva += m.start() - previo
        nuevas.append(texto[previo:m.start()])
        r = reemplazo(m) if callable(reemplazo) else reemplazo
        for p in range(m.start(), m.end()):
            mapa[p] = pos_nueva + min(p - m.start(), len(r))
        nuevas.append(r)
        pos_nueva += len(r)
        previo = m.end()
    for p in range(previo, len(texto) + 1):
        mapa[p] = pos_nueva + (p - previo)
    nuevas.append(texto[previo:])
    for a in anclas:
        a["inicio"], a["fin"] = mapa[a["inicio"]], mapa[a["fin"]]
    return "".join(nuevas), anclas


def limpiar(texto, anclas):
    """Normaliza espacios y restos de plantillas eliminadas, manteniendo las anclas alineadas."""
    pasos = [
        (r"[ \t ]+", " "),                        # espacios horizontales
        (r" *\n[\n ]*", "\n"),                          # saltos de línea (párrafos)
        (r"\( *[,;:]? *\)", ""),                        # paréntesis vaciados: "( )", "(; )"
        (r"\(\s*[,;]\s*", "("),                         # "(; nacido..." -> "(nacido..."
        (r" +(?=[,.;:)])", ""),                         # espacio antes de puntuación
        (r"(?<=\() +", ""),                             # espacio tras "("
        (r"[ \t]{2,}", " "),
    ]
    for patron, repl in pasos:
        texto, anclas = reescribir(texto, anclas, patron, repl)
    izq = len(texto) - len(texto.lstrip())
    texto = texto.strip()
    for a in anclas:
        a["inicio"] = min(max(a["inicio"] - izq, 0), len(texto))
        a["fin"] = min(max(a["fin"] - izq, 0), len(texto))
    finales = []
    for a in anclas:
        while a["inicio"] < a["fin"] and texto[a["inicio"]].isspace():
            a["inicio"] += 1
        while a["fin"] > a["inicio"] and texto[a["fin"] - 1].isspace():
            a["fin"] -= 1
        if a["fin"] > a["inicio"]:
            a["texto"] = texto[a["inicio"]:a["fin"]]
            finales.append(a)
    return texto, finales


def primer_encabezado_2(wc):
    """Primer encabezado de nivel 2 (== ... ==) en orden de documento, a cualquier profundidad.

    Con cursivas o negritas sin cerrar, mwparserfromhell anida el encabezado dentro de
    un <i>/<b>, así que no basta con mirar el nivel superior del árbol. Se ignoran los
    encabezados dentro de plantillas o de etiquetas descartadas, que no se renderizan.
    Los comentarios no tienen nodos hijos, así que nunca contienen encabezados.
    """
    for h in wc.filter_headings(recursive=True):
        if h.level != 2:
            continue
        ancestros = wc.get_ancestors(h)
        if not any(isinstance(p, Template)
                   or (isinstance(p, Tag) and str(p.tag).strip().lower() in TAGS_DESCARTADOS)
                   for p in ancestros):
            return h
    return None


def parsear(contenido):
    """Parseo sin resolver destinos: introducción limpia, anclas y títulos enlazados.

    Devuelve (texto_intro, anclas, titulos_corrido, titulos_plantilla): los títulos
    normalizados de todos los wikilinks del artículo (salvo File/Image/Category),
    separados según estén en el texto corrido o dentro de parámetros de plantillas.
    """
    wc = mwparserfromhell.parse(contenido)
    rend = Renderizador(parar_en=primer_encabezado_2(wc))
    rend.nodos(wc.nodes)
    texto, anclas = limpiar(rend.texto(), rend.anclas)
    en_plantilla = set()
    for t in wc.filter_templates(recursive=True):
        for p in t.params:
            en_plantilla.update(id(l) for l in p.value.filter_wikilinks(recursive=True))
    corrido, plantilla = set(), set()
    for link in wc.filter_wikilinks(recursive=True):
        if es_descartado(link):
            continue
        titulo = normalizar_titulo(str(link.title))
        if titulo:
            (plantilla if id(link) in en_plantilla else corrido).add(titulo)
    return texto, anclas, sorted(corrido), sorted(plantilla)


# ---------------------------------------------------------------- principal

def configurar_log(ruta):
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    manejadores = [logging.StreamHandler(sys.stdout)]
    if ruta:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        manejadores.append(logging.FileHandler(ruta, encoding="utf-8"))
    for h in manejadores:
        h.setFormatter(fmt)
        log.addHandler(h)


def fase_descarga(cliente, nodos, indices):
    """Asegura el wikitexto en caché para cada índice. Devuelve {idx: registro}."""
    registros, descargados, errores, fallos_seguidos = {}, 0, 0, 0
    for k, i in enumerate(indices, 1):
        pageid = nodos[i]["id"]
        ruta = DIR_WIKITEXT / f"{pageid}.json"
        if ruta.exists():
            registros[i] = json.loads(ruta.read_text(encoding="utf-8"))
        else:
            reg = descargar(cliente, pageid)
            descargados += 1
            if reg["estado"] == "error":
                errores += 1
                log.error("Descarga: idx %d pageid %d: %s", i, pageid, reg["detalle"])
                # Una respuesta válida sin contenido (texthidden) no es un rechazo de la API
                fallos_seguidos = fallos_seguidos + 1 if reg.get("fallo_api") else 0
                if fallos_seguidos >= MAX_FALLOS_SEGUIDOS:
                    raise APIRechaza(f"{fallos_seguidos} descargas seguidas con error "
                                     f"(última: pageid {pageid}: {reg['detalle']})")
            else:
                fallos_seguidos = 0
                tmp = ruta.with_suffix(".tmp")
                tmp.write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
                tmp.replace(ruta)
            registros[i] = reg
        if k % PROGRESO_CADA == 0 or k == len(indices):
            log.info("Descarga: %d/%d nodos (%d descargados en esta ejecución, %d errores)",
                     k, len(indices), descargados, errores)
    return registros, descargados, errores


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--piloto", type=int, default=None, help="nº de nodos al azar (sin él: todos)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--log", default=None, help="fichero de log (además de la salida estándar)")
    args = ap.parse_args()
    configurar_log(args.log)

    nodos = json.loads(RUTA_METADATA.read_text(encoding="utf-8"))["nodes"]
    idx_por_titulo = {normalizar_titulo(nd["title"]): i for i, nd in enumerate(nodos)}
    if len(idx_por_titulo) != len(nodos):
        sys.exit("Títulos de metadata que colisionan tras normalizar: revisar antes de seguir.")

    if args.piloto:
        indices = sorted(random.Random(args.semilla).sample(range(len(nodos)), args.piloto))
        salida = DIR_PROCESADOS / "textos_piloto.jsonl"
    else:
        indices = list(range(len(nodos)))
        salida = DIR_PROCESADOS / "textos.jsonl"
    DIR_WIKITEXT.mkdir(parents=True, exist_ok=True)
    DIR_PROCESADOS.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    cliente = ClienteAPI()
    log.info("Inicio: %d nodos, salida %s", len(indices), salida.relative_to(RAIZ))
    try:
        registros, descargados, errores = fase_descarga(cliente, nodos, indices)
        peticiones_descarga = cliente.peticiones

        # Fase 2: parseo
        parseados = {}
        for k, i in enumerate(indices, 1):
            if registros[i]["estado"] == "ok":
                parseados[i] = parsear(registros[i]["contenido"])
            if k % PROGRESO_CADA == 0 or k == len(indices):
                log.info("Parseo: %d/%d nodos", k, len(indices))
        sin_casar = set()
        for texto, anclas, corrido, plantilla in parseados.values():
            sin_casar.update(t for t in corrido + plantilla if t not in idx_por_titulo)
            sin_casar.update(a["destino"] for a in anclas if a["destino"] not in idx_por_titulo)
        redir = resolver_redirecciones(cliente, sin_casar)
    except APIRechaza as e:
        log.error("DETENIDO: la API rechaza las peticiones de forma sostenida: %s", e)
        sys.exit(2)

    def via_redireccion(titulo):
        r = redir.get(titulo)
        if r and r["tipo"] == "redireccion":
            return idx_por_titulo.get(normalizar_titulo(r["destino"]))
        return None

    def resolver(titulos):
        directos = {idx_por_titulo[t] for t in titulos if t in idx_por_titulo}
        por_redir = {via_redireccion(t) for t in titulos if t not in idx_por_titulo} - {None}
        return directos, por_redir

    # Fase 3: escritura
    estados = {}
    tmp = salida.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for i in indices:
            reg = registros[i]
            estados[reg["estado"]] = estados.get(reg["estado"], 0) + 1
            fila = {
                "idx": i, "pageid": nodos[i]["id"], "titulo": nodos[i]["title"],
                "revid": reg.get("revid"), "timestamp": reg.get("timestamp"),
                "estado": reg["estado"], "texto_intro": None, "n_palabras_intro": None,
                "anclas_intro": [], "destinos_articulo": [], "destinos_articulo_sin_resolver": [],
                "destinos_cuerpo": [], "destinos_cuerpo_redir": [], "destinos_solo_plantilla": [],
            }
            if i in parseados:
                texto, anclas, corrido, plantilla = parseados[i]
                dir_c, red_c = resolver(corrido)
                dir_p, red_p = resolver(plantilla)
                cuerpo = dir_c | dir_p
                cuerpo_redir = (red_c | red_p) - cuerpo
                solo_plantilla = (cuerpo | cuerpo_redir) - (dir_c | red_c)
                for a in anclas:
                    a["destino_idx"] = idx_por_titulo.get(a["destino"])
                    a["destino_idx_redir"] = (via_redireccion(a["destino"])
                                              if a["destino_idx"] is None else None)
                fila.update({
                    "texto_intro": texto, "n_palabras_intro": len(texto.split()),
                    "anclas_intro": [{k: a[k] for k in ("inicio", "fin", "texto", "destino",
                                                        "destino_idx", "destino_idx_redir")}
                                     for a in anclas],
                    "destinos_articulo": sorted(cuerpo),
                    "destinos_articulo_sin_resolver": sorted(
                        {t for t in corrido + plantilla if t not in idx_por_titulo}),
                    "destinos_cuerpo": sorted(cuerpo),
                    "destinos_cuerpo_redir": sorted(cuerpo_redir),
                    "destinos_solo_plantilla": sorted(solo_plantilla),
                })
            elif "detalle" in reg:
                fila["detalle"] = reg["detalle"]
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    tmp.replace(salida)

    resumen = {
        "fecha": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nodos": len(indices), "estados": dict(sorted(estados.items())),
        "descargados_en_esta_ejecucion": descargados, "errores_de_descarga": errores,
        "peticiones_descarga": peticiones_descarga,
        "peticiones_redirecciones": cliente.peticiones - peticiones_descarga,
        "peticiones_total": cliente.peticiones,
        "segundos_red": round(cliente.segundos_red, 1),
        "segundos_total": round(time.perf_counter() - t0, 1),
    }
    log.info("Fin: %s", json.dumps(resumen, ensure_ascii=False))
    if not args.piloto:
        ruta = DIR_RESULTADOS / "descarga_textos_ejecuciones.jsonl"
        with ruta.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(resumen, ensure_ascii=False) + "\n")
    log.info("Escrito %s", salida.relative_to(RAIZ))


if __name__ == "__main__":
    main()
