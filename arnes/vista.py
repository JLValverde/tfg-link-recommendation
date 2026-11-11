"""Vistas de una consulta: qué parte del grafo y del texto puede ver el sistema.

- grafo_visible(consulta, grafo="B"): el grafo B (o A) menos los enlaces ocultos
  de la consulta (ocultos_B u ocultos_A). No copia el grafo: filtra al consultar
  los vecinos. Da sucesores visibles (vecinos/sucesores), si un enlace es visible
  (contiene) y los vecinos de la vista no dirigida (vecinos_no_dirigidos), que
  usan las heurísticas estructurales.
- texto_visible(doc, consulta): la introducción de doc con "[...]" en lugar del
  texto de cada ancla que apunta a la consulta: en frío, todas; en caliente,
  solo si el enlace doc -> consulta está oculto. El texto de la propia consulta
  no se toca.
- documentos_alterados(consulta): los documentos cuyo texto_visible difiere del
  original para esa consulta (los que tienen alguna ancla tapada). Permite
  precalcular representaciones del texto original y recalcular solo estos.
- Vista(consulta): lo único que recibe un scorer además de los candidatos. Da el
  grafo visible (B o A), el texto visible de cualquier documento, los títulos y
  los documentos alterados.
  Los scorers no deben leer la consulta completa (referencia, ocultos...): el
  arnés solo les pasa su idx y su escenario.

Los datos (grafo.json y textos.jsonl) se cargan una vez y se reutilizan.
"""

import json
from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTA_GRAFO = RAIZ / "datos" / "procesados" / "grafo.json"
RUTA_TEXTOS = RAIZ / "datos" / "procesados" / "textos.jsonl"
MARCA = "[...]"


@lru_cache(maxsize=None)
def cargar_adyacencia(grafo="B", ruta=RUTA_GRAFO):
    """idx -> frozenset de sucesores en el grafo limpio A o B."""
    if grafo not in ("A", "B"):
        raise ValueError(f"grafo debe ser 'A' o 'B', no {grafo!r}")
    nodos = json.loads(Path(ruta).read_text(encoding="utf-8"))["nodos"]
    return {int(i): frozenset(d[grafo]) for i, d in nodos.items()}


@lru_cache(maxsize=None)
def cargar_textos(ruta=RUTA_TEXTOS):
    """idx -> (texto_intro, anclas_intro) de los nodos válidos."""
    textos = {}
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        f = json.loads(linea)
        if f["estado"] == "ok":
            textos[f["idx"]] = (f["texto_intro"], f["anclas_intro"])
    return textos


@lru_cache(maxsize=None)
def cargar_titulos(ruta=RUTA_TEXTOS):
    """idx -> título legible ("_" -> espacio) de los nodos válidos."""
    titulos = {}
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        f = json.loads(linea)
        if f["estado"] == "ok":
            titulos[f["idx"]] = f["titulo"].replace("_", " ")
    return titulos


@lru_cache(maxsize=None)
def cargar_anclas_hacia(ruta=RUTA_TEXTOS):
    """destino -> frozenset de documentos (distintos del destino) con un ancla hacia él en la introducción."""
    indice = {}
    for doc, (_, anclas) in cargar_textos(ruta).items():
        for a in anclas:
            for q in {a["destino_idx"], a["destino_idx_redir"]} - {None, doc}:
                indice.setdefault(q, set()).add(doc)
    return {q: frozenset(docs) for q, docs in indice.items()}


def documentos_alterados(consulta):
    """Documentos cuyo texto_visible para la consulta difiere del texto original."""
    q = consulta["idx"]
    docs = cargar_anclas_hacia().get(q, frozenset())
    if consulta["escenario"] == "frio":
        return docs
    ocultos = {u for u, v in consulta["ocultos_B"] if v == q}
    return frozenset(d for d in docs if d in ocultos)


@lru_cache(maxsize=None)
def cargar_no_dirigido(grafo="B", ruta=RUTA_GRAFO):
    """idx -> frozenset de vecinos no dirigidos (sucesores ∪ predecesores) en el grafo completo A o B."""
    ady = cargar_adyacencia(grafo, ruta)
    nd = {u: set(vec) for u, vec in ady.items()}
    for u, vec in ady.items():
        for v in vec:
            nd.setdefault(v, set()).add(u)
    return {u: frozenset(vs) for u, vs in nd.items()}


class GrafoVisible:
    """Grafo completo menos los enlaces ocultos de una consulta, sin copiarlo."""

    def __init__(self, adyacencia, ocultos, no_dirigido=None):
        self._ady = adyacencia
        self._nd = no_dirigido
        self._ocultos = {}
        self._tocados = {}  # nodo -> nodos con los que comparte algún enlace oculto (en cualquier sentido)
        for u, v in ocultos:
            self._ocultos.setdefault(u, set()).add(v)
            self._tocados.setdefault(u, set()).add(v)
            self._tocados.setdefault(v, set()).add(u)

    def nodos(self):
        return self._ady.keys()

    def vecinos(self, u):
        """Sucesores visibles de u."""
        vec = self._ady.get(u, frozenset())
        ocultos = self._ocultos.get(u)
        return vec - ocultos if ocultos else vec

    def sucesores(self, u):
        """Sucesores visibles de u (el mismo conjunto que vecinos).

        Es el nombre que se usa al excluir candidatos: en caliente nunca son candidatos
        los sucesores visibles en B de la consulta; los predecesores visibles sí pueden
        serlo (Protocolo de evaluación v2.0, §1).
        """
        return self.vecinos(u)

    def contiene(self, u, v):
        """¿Es visible el enlace u -> v?"""
        return v in self._ady.get(u, ()) and v not in self._ocultos.get(u, ())

    def vecinos_no_dirigidos(self, u):
        """Vecinos de u en la vista no dirigida del grafo visible: v tal que u -> v o v -> u es visible.

        Un enlace oculto solo quita a v si tampoco es visible el enlace en el otro sentido.
        """
        if self._nd is None:
            raise ValueError("Este GrafoVisible se creó sin la vista no dirigida")
        base = self._nd.get(u, frozenset())
        tocados = self._tocados.get(u)
        if not tocados:
            return base
        quitar = {v for v in tocados & base if not (self.contiene(u, v) or self.contiene(v, u))}
        return base - quitar if quitar else base


def grafo_visible(consulta, grafo="B", adyacencia=None):
    """Vista del grafo B (o A) para la consulta: el grafo menos ocultos_B (u ocultos_A)."""
    if adyacencia is not None:
        return GrafoVisible(adyacencia, consulta[f"ocultos_{grafo}"])
    return GrafoVisible(cargar_adyacencia(grafo), consulta[f"ocultos_{grafo}"], cargar_no_dirigido(grafo))


def texto_visible(doc, consulta, textos=None):
    """Introducción de doc con las anclas a la consulta tapadas.

    En frío se tapan todas las anclas a la consulta, haya enlace oculto o no (una
    ancla sin enlace en A también delataría el nombre de un documento nuevo). En
    caliente, solo si el enlace doc -> consulta está en ocultos_B.
    """
    textos = textos if textos is not None else cargar_textos()
    texto, anclas = textos[doc]
    q = consulta["idx"]
    if doc == q:
        return texto
    if consulta["escenario"] != "frio" and not any(
            u == doc and v == q for u, v in consulta["ocultos_B"]):
        return texto
    a_tapar = sorted((a for a in anclas if q in (a["destino_idx"], a["destino_idx_redir"])),
                     key=lambda a: a["inicio"], reverse=True)
    for a in a_tapar:
        texto = texto[:a["inicio"]] + MARCA + texto[a["fin"]:]
    return texto


class Vista:
    """Acceso de un scorer a los datos de una consulta: grafo y texto visibles, títulos."""

    def __init__(self, consulta):
        self._consulta = consulta
        self._grafos = {}

    def grafo(self, nombre="B"):
        """Grafo visible B (o A) para esta consulta."""
        if nombre not in self._grafos:
            self._grafos[nombre] = grafo_visible(self._consulta, nombre)
        return self._grafos[nombre]

    def texto(self, doc):
        """Introducción visible de doc para esta consulta."""
        return texto_visible(doc, self._consulta)

    def titulo(self, doc):
        return cargar_titulos()[doc]

    def documentos_alterados(self):
        """Documentos cuyo texto visible difiere del original para esta consulta."""
        return documentos_alterados(self._consulta)
