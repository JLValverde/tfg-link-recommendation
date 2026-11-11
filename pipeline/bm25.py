"""Índice BM25 (bm25s) sobre título + introducción, con texto visible por consulta.

- Tokenizador propio: palabras (\\w+) en minúsculas, sin stopwords en inglés
  (bm25s.stopwords.STOPWORDS_EN) y sin stemming.
- El índice se construye una vez con el texto ORIGINAL de todos los documentos
  válidos (bm25s, variante lucene, k1 = 1,5, b = 0,75).
- Para una consulta, los documentos cuyo texto_visible difiere del original (los
  que tienen un ancla tapada) se puntúan aparte con su texto visible, con la misma
  fórmula y las mismas estadísticas de colección (IDF y longitud media del
  índice original). Así el texto tapado no puede influir y no se reconstruye el
  índice por consulta.
"""

import math
import re
from functools import lru_cache

import bm25s
import numpy as np
from bm25s.stopwords import STOPWORDS_EN

from arnes.vista import cargar_textos, cargar_titulos

K1 = 1.5
B = 0.75
_STOP = frozenset(STOPWORDS_EN)
_PALABRA = re.compile(r"\w+")


def tokenizar(texto):
    return [t for t in _PALABRA.findall(texto.lower()) if t not in _STOP]


def texto_documento(titulo, texto):
    return f"{titulo}\n{texto}"


class IndiceBM25:
    def __init__(self):
        textos, titulos = cargar_textos(), cargar_titulos()
        self.ids = sorted(textos)
        self.pos = {d: k for k, d in enumerate(self.ids)}
        corpus = [tokenizar(texto_documento(titulos[d], textos[d][0])) for d in self.ids]
        self.modelo = bm25s.BM25(k1=K1, b=B, method="lucene")
        self.modelo.index(corpus, show_progress=False)
        # Estadísticas de colección para puntuar a mano los documentos alterados
        self.n_docs = len(corpus)
        self.longitud_media = float(np.mean([len(t) for t in corpus]))
        df = {}
        for toks in corpus:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.df = df

    def idf(self, termino):
        df = self.df.get(termino, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def puntuar_tokens(self, consulta_tokens, doc_tokens):
        """BM25 (lucene) de un documento dado por sus tokens, con las estadísticas del índice."""
        if not doc_tokens:
            return 0.0
        tf = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        norm = K1 * (1 - B + B * len(doc_tokens) / self.longitud_media)
        return float(sum(self.idf(t) * tf[t] / (tf[t] + norm)
                         for t in consulta_tokens if t in tf and t in self.df))

    def puntuar_todos(self, consulta_tokens, vista):
        """Puntuación de todos los documentos (en el orden de self.ids) para la consulta."""
        conocidos = [t for t in consulta_tokens if t in self.df]
        if conocidos:
            punt = np.asarray(self.modelo.get_scores(conocidos), dtype=np.float64)
        else:
            punt = np.zeros(self.n_docs)
        for d in vista.documentos_alterados():
            punt[self.pos[d]] = self.puntuar_tokens(conocidos, tokenizar(texto_documento(vista.titulo(d), vista.texto(d))))
        return punt


@lru_cache(maxsize=1)
def indice():
    return IndiceBM25()


def tokens_consulta(vista, q):
    """Consulta = título + introducción de la propia consulta (su texto nunca se tapa)."""
    return tokenizar(texto_documento(vista.titulo(q), vista.texto(q)))
