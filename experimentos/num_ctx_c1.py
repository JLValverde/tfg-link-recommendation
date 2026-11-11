"""num_ctx de C1 (T2.2a): menor potencia de 2 >= 1,2 x el prompt más largo posible.

El prompt más largo posible es la plantilla de pipeline.reordenacion con la consulta
y el candidato más largos de la colección, cada uno con su tope de palabras (400 y
200). Se mide en tokens con el propio modelo, por /api/chat (prompt_eval_count;
Ollama no expone un tokenizador):
1. Coste fijo de la plantilla de chat: prompt_eval_count de un mensaje vacío.
2. Tokens de "Title: {título}\\n{introducción recortada}" de TODOS los documentos
   válidos, con el tope del candidato (200) y con el de la consulta (400); los que
   tienen 200 palabras o menos valen lo mismo con los dos topes.
3. prompt_eval_count del prompt completo con la consulta y el candidato más largos.

Se mide sobre el texto original: texto_visible solo cambia anclas por "[...]".

Escribe experimentos/resultados/num_ctx_c1.json.

Con --variante N (T2.2b), repite solo el paso 3 con la plantilla de la variante N de
experimentos/variantes_c1.py: la consulta y el candidato más largos (pasos 1 y 2) no
dependen de la plantilla, así que se leen de num_ctx_c1.json. Escribe
num_ctx_c1_vN.json.

Uso (desde la raíz del repo, con Ollama en marcha):
    .venv/Scripts/python.exe experimentos/num_ctx_c1.py [--variante N]
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.vista import cargar_textos, cargar_titulos  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402
from variantes_c1 import VARIANTES  # noqa: E402

RUTA_SALIDA = RAIZ / "experimentos" / "resultados" / "num_ctx_c1.json"
NUM_CTX_MEDIDA = 8192  # holgado: ningún texto recortado se acerca
FACTOR = 1.2


def tokens(contenido):
    return ro.llamar(contenido, num_ctx=NUM_CTX_MEDIDA).prompt_eval_count


def por_variante(variante):
    base = json.loads(RUTA_SALIDA.read_text(encoding="utf-8"))
    textos, titulos = cargar_textos(), cargar_titulos()
    q, c = base["prompt_mas_largo"]["consulta"], base["prompt_mas_largo"]["candidato"]
    ro.PLANTILLA = VARIANTES[variante]
    prompt = ro.construir_prompt(titulos[q], ro.recortar(textos[q][0], ro.MAX_PALABRAS_CONSULTA)[0],
                                 titulos[c], ro.recortar(textos[c][0], ro.MAX_PALABRAS_CANDIDATO)[0])
    maximo = tokens(prompt)
    num_ctx = 2 ** math.ceil(math.log2(FACTOR * maximo))
    resumen = {"modelo": ro.MODELO, "variante": variante, "num_ctx_medida": NUM_CTX_MEDIDA,
               "prompt_mas_largo": {"consulta": q, "candidato": c, "tokens": maximo},
               "factor": FACTOR, "factor_x_maximo": FACTOR * maximo, "num_ctx": num_ctx,
               "num_ctx_fijado": ro.NUM_CTX, "cabe": FACTOR * maximo <= ro.NUM_CTX}
    ruta = RUTA_SALIDA.with_name(f"num_ctx_c1_v{variante}.json")
    ruta.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", type=int, choices=sorted(VARIANTES), default=None)
    args = ap.parse_args()
    if args.variante is not None:
        return por_variante(args.variante)
    textos, titulos = cargar_textos(), cargar_titulos()
    fijo = tokens("")
    t0 = time.perf_counter()
    por_tope = {"candidato": {}, "consulta": {}}
    for k, (d, (texto, _)) in enumerate(sorted(textos.items()), start=1):
        x200, recortado = ro.recortar(texto, ro.MAX_PALABRAS_CANDIDATO)
        por_tope["candidato"][d] = tokens(f"Title: {titulos[d]}\n{x200}") - fijo
        if recortado:
            x400, _ = ro.recortar(texto, ro.MAX_PALABRAS_CONSULTA)
            por_tope["consulta"][d] = tokens(f"Title: {titulos[d]}\n{x400}") - fijo
        else:
            por_tope["consulta"][d] = por_tope["candidato"][d]
        if k % 1000 == 0:
            print(f"{k}/{len(textos)} documentos, {time.perf_counter() - t0:.0f} s", flush=True)
    q = max(por_tope["consulta"], key=por_tope["consulta"].get)
    c = max(por_tope["candidato"], key=por_tope["candidato"].get)
    prompt = ro.construir_prompt(titulos[q], ro.recortar(textos[q][0], ro.MAX_PALABRAS_CONSULTA)[0],
                                 titulos[c], ro.recortar(textos[c][0], ro.MAX_PALABRAS_CANDIDATO)[0])
    maximo = tokens(prompt)
    num_ctx = 2 ** math.ceil(math.log2(FACTOR * maximo))
    resumen = {
        "modelo": ro.MODELO, "num_ctx_medida": NUM_CTX_MEDIDA, "n_documentos": len(textos),
        "tokens_plantilla_chat_vacia": fijo,
        **{f"tokens_{rol}": {"max": v[max(v, key=v.get)], "idx_max": max(v, key=v.get),
                              "p50": statistics.median(v.values()),
                              "p95": sorted(v.values())[int(0.95 * len(v))]}
           for rol, v in por_tope.items()},
        "prompt_mas_largo": {"consulta": q, "candidato": c, "tokens": maximo},
        "factor": FACTOR, "num_ctx": num_ctx, "duracion_s": round(time.perf_counter() - t0, 1),
    }
    RUTA_SALIDA.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
