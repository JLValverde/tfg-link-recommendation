"""Muestra de revisión manual de la regla estricta del detector de explícitos (T0.5).

Lee datos/procesados/explicitos.jsonl (no recalcula el detector) y elige 100
pares (i, j) al azar (random.Random(43)) entre los que la regla ESTRICTA marca
como explícitos por la vía NOMBRE y que no están en A. Escribe
experimentos/resultados/muestra_explicitos_estricta.csv con la columna
"correcto" vacía.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe datos\\muestra_explicitos_estricta.py
"""

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTA_EXPLICITOS = RAIZ / "datos" / "procesados" / "explicitos.jsonl"
RUTA_TEXTOS = RAIZ / "datos" / "procesados" / "textos.jsonl"
RUTA_DATA = RAIZ / "datos" / "raw" / "wikics" / "data.json"
RUTA_MUESTRA = RAIZ / "experimentos" / "resultados" / "muestra_explicitos_estricta.csv"
TAM_MUESTRA = 100
SEMILLA = 43
CONTEXTO = 60


def main():
    filas = [json.loads(l) for l in RUTA_TEXTOS.read_text(encoding="utf-8").splitlines()]
    links = json.loads(RUTA_DATA.read_text(encoding="utf-8"))["links"]
    valido = [f["estado"] == "ok" for f in filas]
    titulos = [f["titulo"] for f in filas]

    candidatos = {}
    for linea in RUTA_EXPLICITOS.read_text(encoding="utf-8").splitlines():
        e = json.loads(linea)
        i = e["idx"]
        A = {j for j in links[i] if j != i and valido[j]}
        for d in e["estricta"]:
            if d["via"] == "nombre" and d["destino"] not in A:
                candidatos[(i, d["destino"])] = d
    poblacion = sorted(candidatos)
    print(f"Pares candidatos: {len(poblacion)}")
    print(f"Por fuente: {dict(Counter(candidatos[p]['fuente'] for p in poblacion).most_common())}")

    muestra = sorted(random.Random(SEMILLA).sample(poblacion, TAM_MUESTRA))
    with RUTA_MUESTRA.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx_origen", "titulo_origen", "titulo_destino", "nombre_detectado",
                    "fuente_nombre", "contexto", "correcto"])
        for i, j in muestra:
            d = candidatos[(i, j)]
            texto = filas[i]["texto_intro"]
            contexto = (texto[max(0, d["inicio"] - CONTEXTO):d["inicio"]] + "["
                        + texto[d["inicio"]:d["fin"]] + "]" + texto[d["fin"]:d["fin"] + CONTEXTO])
            w.writerow([i, titulos[i], titulos[j], d["nombre"], d["fuente"],
                        re.sub(r"\s+", " ", contexto), ""])
    print(f"Muestra por fuente: {dict(Counter(candidatos[p]['fuente'] for p in muestra).most_common())}")
    print(f"Escrito {RUTA_MUESTRA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
