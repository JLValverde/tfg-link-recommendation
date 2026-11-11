"""Capa 1 del pipeline: embeddings -> top-K candidatos (A2).

- Texto de cada documento: título + introducción.
- Los embeddings del texto ORIGINAL de todos los documentos válidos se calculan
  una vez (normalizados, similitud = producto escalar) y se guardan en
  datos/procesados/embeddings/<modelo>.npz.
- Para cada consulta, solo se recalculan los embeddings de los documentos cuyo
  texto_visible difiere del original (vista.documentos_alterados()).
- La consulta se representa con su propio título + introducción (su texto nunca
  se tapa), que es su embedding precalculado.
- Recuperador: ordena toda la colección por similitud coseno con la consulta,
  sin la consulta ni sus sucesores visibles en B. Empates por idx.
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from arnes.vista import cargar_textos, cargar_titulos

RAIZ = Path(__file__).resolve().parent.parent
DIR_EMBEDDINGS = RAIZ / "datos" / "procesados" / "embeddings"
MODELOS_CANDIDATOS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "Qwen/Qwen3-Embedding-0.6B",
)
TAM_LOTE = 32


def texto_documento(titulo, texto):
    return f"{titulo}\n{texto}"


def _nombre_fichero(modelo):
    return re.sub(r"[^A-Za-z0-9.-]+", "_", modelo)


@lru_cache(maxsize=None)
def cargar_modelo(modelo):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(modelo, device="cuda")


def codificar(modelo, textos):
    return cargar_modelo(modelo).encode(list(textos), batch_size=TAM_LOTE, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


class IndiceEmbeddings:
    def __init__(self, modelo, recalcular=False, directorio=DIR_EMBEDDINGS):
        self.modelo = modelo
        textos, titulos = cargar_textos(), cargar_titulos()
        self.ids = sorted(textos)
        self.pos = {d: k for k, d in enumerate(self.ids)}
        directorio = Path(directorio)
        ruta = directorio / f"{_nombre_fichero(modelo)}.npz"
        if ruta.exists() and not recalcular:
            datos = np.load(ruta)
            assert list(datos["ids"]) == self.ids, "embeddings guardados con otros ids"
            self.base = datos["emb"]
        else:
            self.base = codificar(modelo, [texto_documento(titulos[d], textos[d][0]) for d in self.ids])
            directorio.mkdir(parents=True, exist_ok=True)
            np.savez(ruta, ids=np.asarray(self.ids), emb=self.base)

    def embeddings_visibles(self, vista):
        """{doc: embedding} de los documentos alterados para esta consulta (texto visible)."""
        alterados = sorted(vista.documentos_alterados())
        if not alterados:
            return {}
        emb = codificar(self.modelo, [texto_documento(vista.titulo(d), vista.texto(d)) for d in alterados])
        return dict(zip(alterados, emb))

    def puntuar_todos(self, vista, q):
        """Similitud coseno de la consulta con todos los documentos (orden de self.ids)."""
        qv = self.base[self.pos[q]]
        punt = self.base @ qv
        for d, e in self.embeddings_visibles(vista).items():
            punt[self.pos[d]] = float(e @ qv)
        return punt

    def puntuar(self, vista, q, candidatos):
        """Similitud coseno solo de los candidatos; recalcula solo los alterados que estén entre ellos."""
        qv = self.base[self.pos[q]]
        alterados = sorted(set(candidatos) & set(vista.documentos_alterados()))
        visibles = {}
        if alterados:
            emb = codificar(self.modelo, [texto_documento(vista.titulo(d), vista.texto(d)) for d in alterados])
            visibles = dict(zip(alterados, emb))
        return {c: float((visibles[c] if c in visibles else self.base[self.pos[c]]) @ qv) for c in candidatos}

    def recuperar(self, vista, q, k):
        """Top-k de la colección por similitud, sin la consulta ni sus sucesores visibles en B.

        Devuelve [(doc, puntuación)], de mayor a menor puntuación; empates por idx.
        """
        punt = self.puntuar_todos(vista, q)
        # Solo los sucesores visibles en B; los predecesores visibles siguen siendo candidatos
        excluidos = {q} | set(vista.grafo("B").sucesores(q))
        ids = np.asarray(self.ids)
        orden = np.lexsort((ids, -punt))  # puntuación descendente, idx ascendente
        salida = []
        for k_pos in orden:
            d = int(ids[k_pos])
            if d in excluidos:
                continue
            salida.append((d, float(punt[k_pos])))
            if len(salida) == k:
                break
        return salida


@lru_cache(maxsize=None)
def indice(modelo):
    return IndiceEmbeddings(modelo)
