"""Consultas de los dos escenarios, caliente y frío (T0.5-T0.8).

Lee datos/procesados/grafo.json y escribe datos/procesados/consultas_caliente.jsonl
y datos/procesados/consultas_frio.jsonl: una consulta por línea, ordenadas por idx,
con las mismas 2000 consultas en los dos escenarios.

- Elegibles: al menos 3 enlaces de B con destino implícito (B \\ explicitos) y una
  introducción de al menos 30 palabras.
- Muestra principal: 2000 elegibles con random.Random(42). Desarrollo: 200 de esas
  2000 con random.Random(43) (es_desarrollo).
- Caliente (grafo visible = B): se oculta la mitad de los enlaces implícitos de B
  (ceil) y la mitad de los explícitos (floor), con random.Random(44 + idx).
  referencia = salientes ocultados; ocultos_B = ocultados + sus recíprocos en B;
  ocultos_A = ocultados + sus recíprocos en A.
- Frío: referencia = todos los salientes de B; ocultos_B = todos los enlaces del
  documento en B, en las dos direcciones; ocultos_A = lo mismo en A.
- En los dos: explicitos, explicitos_solo_ancla y neutros_plantilla (A \\ B salientes).

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe datos\\generar_consultas.py
"""

import json
import math
import random
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTA_GRAFO = RAIZ / "datos" / "procesados" / "grafo.json"
DIR_SALIDA = RAIZ / "datos" / "procesados"

MIN_IMPLICITOS = 3
MIN_PALABRAS = 30
N_CONSULTAS = 2000
N_DESARROLLO = 200
SEMILLA_MUESTRA = 42
SEMILLA_DESARROLLO = 43
SEMILLA_OCULTACION = 44  # + idx


def cargar_grafo(ruta=RUTA_GRAFO):
    g = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return {int(i): d for i, d in g["nodos"].items()}


def elegibles(nodos):
    return sorted(i for i, d in nodos.items()
                  if len(set(d["B"]) - set(d["explicitos"])) >= MIN_IMPLICITOS
                  and (d["n_palabras_intro"] or 0) >= MIN_PALABRAS)


def entrantes(nodos, clave):
    """destino -> conjunto de orígenes en el grafo indicado (A o B)."""
    pred = {i: set() for i in nodos}
    for i, d in nodos.items():
        for j in d[clave]:
            pred[j].add(i)
    return pred


def pares(conjunto):
    return [list(p) for p in sorted(conjunto)]


def generar(nodos):
    """Devuelve (consultas_caliente, consultas_frio), listas ordenadas por idx."""
    candidatos = elegibles(nodos)
    principal = sorted(random.Random(SEMILLA_MUESTRA).sample(candidatos, N_CONSULTAS))
    desarrollo = set(random.Random(SEMILLA_DESARROLLO).sample(principal, N_DESARROLLO))
    pred_A, pred_B = entrantes(nodos, "A"), entrantes(nodos, "B")

    caliente, frio = [], []
    for q in principal:
        d = nodos[q]
        A, B, exp = set(d["A"]), set(d["B"]), set(d["explicitos"])
        comun = {
            "idx": q,
            "es_desarrollo": q in desarrollo,
            "explicitos": sorted(exp),
            "explicitos_solo_ancla": sorted(d["explicitos_solo_ancla"]),
            "neutros_plantilla": sorted(A - B),
        }
        b_impl, b_expl = sorted(B - exp), sorted(B & exp)

        # Caliente
        rng = random.Random(SEMILLA_OCULTACION + q)
        ocultos_impl = rng.sample(b_impl, math.ceil(len(b_impl) / 2))
        ocultos_expl = rng.sample(b_expl, len(b_expl) // 2)
        ref = set(ocultos_impl) | set(ocultos_expl)
        ocultos_B = {(q, j) for j in ref} | {(j, q) for j in ref if j in pred_B[q]}
        ocultos_A = {(q, j) for j in ref} | {(j, q) for j in ref if j in pred_A[q]}
        caliente.append({
            "idx": q, "escenario": "caliente", "es_desarrollo": comun["es_desarrollo"],
            "referencia": sorted(ref), "referencia_implicita": sorted(ocultos_impl),
            "ocultos_B": pares(ocultos_B), "ocultos_A": pares(ocultos_A),
            "explicitos": comun["explicitos"],
            "explicitos_solo_ancla": comun["explicitos_solo_ancla"],
            "neutros_plantilla": comun["neutros_plantilla"],
        })

        # Frío
        ocultos_B = {(q, j) for j in B} | {(j, q) for j in pred_B[q]}
        ocultos_A = {(q, j) for j in A} | {(j, q) for j in pred_A[q]}
        frio.append({
            "idx": q, "escenario": "frio", "es_desarrollo": comun["es_desarrollo"],
            "referencia": sorted(B), "referencia_implicita": b_impl,
            "ocultos_B": pares(ocultos_B), "ocultos_A": pares(ocultos_A),
            "explicitos": comun["explicitos"],
            "explicitos_solo_ancla": comun["explicitos_solo_ancla"],
            "neutros_plantilla": comun["neutros_plantilla"],
        })
    return caliente, frio, len(candidatos)


def escribir(consultas, ruta):
    with Path(ruta).open("w", encoding="utf-8", newline="\n") as f:
        for c in consultas:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main(dir_salida=DIR_SALIDA, ruta_grafo=RUTA_GRAFO):
    nodos = cargar_grafo(ruta_grafo)
    caliente, frio, n_elegibles = generar(nodos)
    dir_salida = Path(dir_salida)
    escribir(caliente, dir_salida / "consultas_caliente.jsonl")
    escribir(frio, dir_salida / "consultas_frio.jsonl")
    print(f"Elegibles: {n_elegibles}; consultas: {len(caliente)} "
          f"({sum(c['es_desarrollo'] for c in caliente)} de desarrollo)")
    return n_elegibles


if __name__ == "__main__":
    main()
