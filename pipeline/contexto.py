"""Capa 2 del pipeline: contexto estructural de C2 (T2.4).

Todo sale de la Vista (invariante 1): vecinos = vista.grafo("B").vecinos_no_dirigidos(doc),
los visibles para la consulta en la vista no dirigida de B. Grado = número de esos
vecinos.
- Lista de un documento: títulos de sus vecinos, por grado descendente (empate:
  título), como mucho MAX_VECINOS, separados por SEPARADOR. Sin vecinos: VACIO.
- Línea de comunes: vecinos comunes de la consulta y del candidato, por grado
  ascendente (empate: título), con el mismo tope, separador y VACIO.
En frío la consulta no tiene vecinos visibles: las plantillas de frío no llevan la
lista de la consulta ni los comunes.

Los parámetros (grafo, tope, separador, texto vacío) se leen de la especificación
congelada de C2 (pipeline/prompts/c2.json) cuando existe; los valores de aquí son
los de las variantes de desarrollo (experimentos/variantes_c2.py).
"""

GRAFO = "B"
MAX_VECINOS = 20
SEPARADOR = "; "
VACIO = "none"


class Contexto:
    """Listas de vecinos de una consulta; cachea grados y listas por documento."""

    def __init__(self, vista, grafo=GRAFO, max_vecinos=MAX_VECINOS, separador=SEPARADOR, vacio=VACIO):
        self.vista = vista
        self.g = vista.grafo(grafo)
        self.max_vecinos, self.separador, self.vacio = max_vecinos, separador, vacio
        self._vecinos, self._grados, self._listas = {}, {}, {}

    def vecinos(self, doc):
        if doc not in self._vecinos:
            self._vecinos[doc] = self.g.vecinos_no_dirigidos(doc)
        return self._vecinos[doc]

    def grado(self, doc):
        if doc not in self._grados:
            self._grados[doc] = len(self.vecinos(doc))
        return self._grados[doc]

    def _unir(self, docs):
        titulos = [self.vista.titulo(d) for d in docs[:self.max_vecinos]]
        return self.separador.join(titulos) if titulos else self.vacio

    def ordenados(self, doc):
        """Vecinos de doc por grado descendente y título (todos, sin tope)."""
        return sorted(self.vecinos(doc), key=lambda v: (-self.grado(v), self.vista.titulo(v)))

    def comunes_ordenados(self, a, b):
        """Vecinos comunes de a y b por grado ascendente y título (todos, sin tope)."""
        return sorted(self.vecinos(a) & self.vecinos(b), key=lambda v: (self.grado(v), self.vista.titulo(v)))

    def lista(self, doc):
        if doc not in self._listas:
            self._listas[doc] = self._unir(self.ordenados(doc))
        return self._listas[doc]

    def comunes(self, a, b):
        return self._unir(self.comunes_ordenados(a, b))
