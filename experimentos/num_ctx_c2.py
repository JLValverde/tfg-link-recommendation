"""num_ctx de C2 (T2.4): menor potencia de 2 >= 1,2 x el prompt más largo posible.

El prompt más largo posible de cada variante de experimentos/variantes_c2.py es su
plantilla con:
- la consulta y el candidato más largos de la colección con sus topes de palabras
  (de resultados/num_ctx_c1.json: no dependen de la plantilla);
- en cada lista (vecinos de A, de B y comunes), los MAX_VECINOS títulos más largos
  en tokens de la colección. Se miden en tokens los N_TITULOS títulos más largos en
  caracteres y se toman los MAX_VECINOS con más tokens.
Se mide con el propio modelo (prompt_eval_count por /api/chat), en caliente y en frío.

Con --modelo se mide con otro modelo (T2.6: comprobar que el num_ctx vale para las
réplicas). Escribe experimentos/resultados/num_ctx_c2[_{modelo}].json.

Uso (desde la raíz del repo, con Ollama en marcha):
    .venv/Scripts/python.exe experimentos/num_ctx_c2.py [--modelo qwen2.5:7b-instruct-q4_K_M]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.vista import cargar_textos, cargar_titulos  # noqa: E402
from pipeline import contexto  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402
from variantes_c2 import VARIANTES  # noqa: E402

DIR = RAIZ / "experimentos" / "resultados"
NUM_CTX_MEDIDA = 8192
FACTOR = 1.2
N_TITULOS = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=ro.MODELO)
    args = ap.parse_args()

    def tokens(contenido):
        return ro.llamar(contenido, modelo=args.modelo, num_ctx=NUM_CTX_MEDIDA).prompt_eval_count

    textos, titulos = cargar_textos(), cargar_titulos()
    base = json.loads((DIR / "num_ctx_c1.json").read_text(encoding="utf-8"))["prompt_mas_largo"]
    q, c = base["consulta"], base["candidato"]
    fijo = tokens("")
    largos = sorted(titulos, key=lambda d: (-len(titulos[d]), d))[:N_TITULOS]
    por_titulo = {d: tokens(titulos[d]) - fijo for d in largos}
    elegidos = sorted(largos, key=lambda d: (-por_titulo[d], d))[:contexto.MAX_VECINOS]
    lista = contexto.SEPARADOR.join(titulos[d] for d in elegidos)
    campos = {"titulo_consulta": titulos[q], "texto_consulta": ro.recortar(textos[q][0], ro.MAX_PALABRAS_CONSULTA)[0],
              "titulo_candidato": titulos[c], "texto_candidato": ro.recortar(textos[c][0], ro.MAX_PALABRAS_CANDIDATO)[0],
              "vecinos_consulta": lista, "vecinos_candidato": lista, "comunes": lista}
    variantes = {}
    for v, espec in VARIANTES.items():
        variantes[v] = {}
        for esc, plantilla in espec["plantillas"].items():
            maximo = tokens(plantilla.format(**campos))
            variantes[v][esc] = {"tokens": maximo, "factor_x_maximo": FACTOR * maximo,
                                 "num_ctx": 2 ** math.ceil(math.log2(FACTOR * maximo)),
                                 "cabe": FACTOR * maximo <= ro.NUM_CTX}
    resumen = {"modelo": args.modelo, "num_ctx_medida": NUM_CTX_MEDIDA, "consulta": q, "candidato": c,
               "titulos_medidos": N_TITULOS, "titulos_lista": [titulos[d] for d in elegidos],
               "tokens_titulos_lista": [por_titulo[d] for d in elegidos],
               "tokens_lista": tokens(lista) - fijo, "factor": FACTOR, "num_ctx_fijado": ro.NUM_CTX,
               "variantes": variantes,
               "cabe_todo": all(e["cabe"] for v in variantes.values() for e in v.values())}
    sufijo = "" if args.modelo == ro.MODELO else "_" + re.sub(r"[^A-Za-z0-9.-]+", "_", args.modelo)
    ruta = DIR / f"num_ctx_c2{sufijo}.json"
    ruta.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in resumen.items() if k != "titulos_lista"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
