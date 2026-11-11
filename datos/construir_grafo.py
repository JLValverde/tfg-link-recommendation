"""Grafo limpio de WikiCS (T0.5-T0.8).

Lee datos/raw/wikics/data.json, datos/procesados/textos.jsonl y
datos/procesados/explicitos.jsonl (regla estricta del detector) y escribe
datos/procesados/grafo.json.

Solo quedan los nodos válidos (estado == ok); los demás se quitan con todos sus
enlaces. Los nodos conservan su índice de WikiCS. Para cada nodo válido:
- A: enlaces de WikiCS sin duplicados ni autobucles.
- B: A ∩ enlaces escritos en el wikitexto (destinos_cuerpo ∪ destinos_cuerpo_redir).
- explicitos: destinos explícitos para el documento (enlazados o no).
- explicitos_solo_ancla: destinos con ancla en su introducción.
- n_palabras_intro.
Ningún índice apunta a un nodo no válido.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe datos\\construir_grafo.py
"""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATA = RAIZ / "datos" / "raw" / "wikics" / "data.json"
RUTA_TEXTOS = RAIZ / "datos" / "procesados" / "textos.jsonl"
RUTA_EXPLICITOS = RAIZ / "datos" / "procesados" / "explicitos.jsonl"
RUTA_GRAFO = RAIZ / "datos" / "procesados" / "grafo.json"
REGLA = "estricta"


def construir():
    filas = [json.loads(l) for l in RUTA_TEXTOS.read_text(encoding="utf-8").splitlines()]
    links = json.loads(RUTA_DATA.read_text(encoding="utf-8"))["links"]
    n = len(links)
    assert [f["idx"] for f in filas] == list(range(n))
    valido = [f["estado"] == "ok" for f in filas]
    explicitos = {}
    for linea in RUTA_EXPLICITOS.read_text(encoding="utf-8").splitlines():
        e = json.loads(linea)
        explicitos[e["idx"]] = e[REGLA]
    validos = [i for i in range(n) if valido[i]]
    assert sorted(explicitos) == validos, "explicitos.jsonl no cubre exactamente los nodos válidos"

    nodos = {}
    for i in validos:
        A = {j for j in links[i] if j != i and valido[j]}
        escritos = set(filas[i]["destinos_cuerpo"]) | set(filas[i]["destinos_cuerpo_redir"])
        exp = {d["destino"] for d in explicitos[i]}
        solo_ancla = {d["destino"] for d in explicitos[i] if d["via"] == "ancla"}
        nodos[str(i)] = {
            "A": sorted(A),
            "B": sorted(A & escritos),
            "explicitos": sorted(exp),
            "explicitos_solo_ancla": sorted(solo_ancla),
            "n_palabras_intro": filas[i]["n_palabras_intro"],
        }

    # Comprobaciones
    for i, d in nodos.items():
        for clave in ("A", "B", "explicitos", "explicitos_solo_ancla"):
            assert all(valido[j] and j != int(i) for j in d[clave]), (i, clave)
        assert set(d["B"]) <= set(d["A"])
        assert set(d["explicitos_solo_ancla"]) <= set(d["explicitos"])

    grafo = {
        "descripcion": "Grafo limpio de WikiCS: solo nodos válidos (estado ok), índices de WikiCS.",
        "regla_explicitos": REGLA,
        "n_nodos": len(nodos),
        "n_enlaces_A": sum(len(d["A"]) for d in nodos.values()),
        "n_enlaces_B": sum(len(d["B"]) for d in nodos.values()),
        "nodos": nodos,
    }
    return grafo


def main():
    grafo = construir()
    RUTA_GRAFO.write_text(json.dumps(grafo, ensure_ascii=False, separators=(",", ":")) + "\n",
                          encoding="utf-8")
    print(f"Nodos: {grafo['n_nodos']}; enlaces A: {grafo['n_enlaces_A']}; enlaces B: {grafo['n_enlaces_B']}")
    print(f"Escrito {RUTA_GRAFO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
