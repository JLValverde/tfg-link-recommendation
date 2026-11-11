"""Capa 2 del arnés: candidatos de cada consulta; se guardan en disco.

Modo aleatorio (control, sin sistema de recuperación):
  para cada escenario y semilla s, N = 50 candidatos por consulta =
  hasta G = 10 enlaces de "referencia" (si hay más, muestreo con la semilla,
  estratificado entre implícitos y explícitos con al menos 1 implícito) +
  negativos al azar entre los documentos válidos que no están enlazados con la
  consulta en A en ninguna dirección, distintos de la consulta. La lista va en
  orden aleatorio.
  Salida: datos/procesados/candidatos_aleatorio_{escenario}_s{s}.jsonl
  ({"idx", "candidatos": [...]}, ordenado por idx).

Modo recuperador (T1.8):
  los K = 50 primeros documentos del recuperador de embeddings, sin la consulta
  ni sus sucesores visibles en B, con la puntuación del recuperador. Los enlaces
  de referencia no se insertan. El arnés solo lee estas listas de disco: las
  genera experimentos/recuperador.py (arnes/ no importa nada de pipeline/).
  Fichero: datos/procesados/candidatos_recuperador_{escenario}.jsonl
  ({"idx", "candidatos": [...], "puntuaciones": [...]}, de mejor a peor).

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe -m arnes.candidatos     # genera el modo aleatorio
"""

import json
import math
import random
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROCESADOS = RAIZ / "datos" / "procesados"
RUTA_GRAFO = DIR_PROCESADOS / "grafo.json"

ESCENARIOS = ("caliente", "frio")
SEMILLAS = (1, 2, 3)
N_CANDIDATOS = 50
MAX_REFERENCIAS = 10


def ruta_candidatos(modo, escenario, semilla=None, directorio=DIR_PROCESADOS):
    sufijo = f"_s{semilla}" if semilla is not None else ""
    return Path(directorio) / f"candidatos_{modo}_{escenario}{sufijo}.jsonl"


def cargar_consultas(escenario, directorio=DIR_PROCESADOS):
    ruta = Path(directorio) / f"consultas_{escenario}.jsonl"
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines()]


def cargar_candidatos(modo, escenario, semilla=None, directorio=DIR_PROCESADOS):
    """idx -> lista de candidatos."""
    ruta = ruta_candidatos(modo, escenario, semilla, directorio)
    return {d["idx"]: d["candidatos"]
            for d in (json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines())}


def rng_consulta(semilla, idx):
    """Generador por consulta, determinista y distinto para cada (semilla, idx)."""
    return random.Random(1_000_003 * semilla + idx)


def muestrear_referencias(referencia, implicitas, rng):
    """Hasta MAX_REFERENCIAS enlaces de referencia, con muestreo estratificado si hay más.

    Las plazas se reparten entre implícitas y explícitas en proporción a su peso
    (redondeo al entero más próximo), con al menos 1 implícita si la consulta tiene
    alguna. Así la tarea implícita nunca se queda sin relevantes en la lista
    (decisión de T1.2: sin estratificar, la consulta 4010 de frío, semilla 2, se
    quedaba sin ninguna).
    """
    ref = sorted(referencia)
    if len(ref) <= MAX_REFERENCIAS:
        return ref
    impl = sorted(set(ref) & set(implicitas))
    expl = sorted(set(ref) - set(impl))
    n_impl = math.floor(MAX_REFERENCIAS * len(impl) / len(ref) + 0.5)
    n_impl = max(n_impl, 1 if impl else 0, MAX_REFERENCIAS - len(expl))
    n_impl = min(n_impl, len(impl))
    n_expl = MAX_REFERENCIAS - n_impl
    return sorted(rng.sample(impl, n_impl) + rng.sample(expl, n_expl))


def candidatos_aleatorio(consultas, nodos, semilla):
    """idx -> lista de N candidatos (referencias muestreadas + negativos), en orden aleatorio."""
    validos = sorted(nodos)
    entrantes_A = {i: set() for i in nodos}
    for i, d in nodos.items():
        for j in d["A"]:
            entrantes_A[j].add(i)
    salida = {}
    for c in consultas:
        q = c["idx"]
        rng = rng_consulta(semilla, q)
        ref = muestrear_referencias(c["referencia"], c["referencia_implicita"], rng)
        enlazados = set(nodos[q]["A"]) | entrantes_A[q] | {q}
        pool = [v for v in validos if v not in enlazados]
        negativos = rng.sample(pool, N_CANDIDATOS - len(ref))
        lista = ref + negativos
        rng.shuffle(lista)
        salida[q] = lista
    return salida


K_RECUPERADOR = 50


def escribir_candidatos_recuperador(listas, escenario, directorio=DIR_PROCESADOS):
    """Escribe las listas del modo recuperador ({idx: [(doc, puntuación)]}, de mejor a peor)."""
    ruta = ruta_candidatos("recuperador", escenario, None, directorio)
    with ruta.open("w", encoding="utf-8", newline="\n") as f:
        for q in sorted(listas):
            f.write(json.dumps({"idx": q, "candidatos": [d for d, _ in listas[q]],
                                "puntuaciones": [round(p, 6) for _, p in listas[q]]}) + "\n")
    return ruta


def generar_aleatorio(directorio=DIR_PROCESADOS, ruta_grafo=RUTA_GRAFO):
    g = json.loads(Path(ruta_grafo).read_text(encoding="utf-8"))
    nodos = {int(i): d for i, d in g["nodos"].items()}
    rutas = []
    for escenario in ESCENARIOS:
        consultas = cargar_consultas(escenario)
        for s in SEMILLAS:
            cands = candidatos_aleatorio(consultas, nodos, s)
            ruta = ruta_candidatos("aleatorio", escenario, s, directorio)
            with ruta.open("w", encoding="utf-8", newline="\n") as f:
                for q in sorted(cands):
                    f.write(json.dumps({"idx": q, "candidatos": cands[q]}) + "\n")
            rutas.append(ruta)
    return rutas


if __name__ == "__main__":
    for r in generar_aleatorio():
        print(f"Escrito {r.relative_to(RAIZ)}")
